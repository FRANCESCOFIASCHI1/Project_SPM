#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <vector>
#include <getopt.h>
#include <future>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <omp.h>
#include <mpi.h>
#include "hpc_helpers.hpp"

using namespace std;

// --- Struct dei record ---
struct Record {
    unsigned long key;
    uint32_t len;
    char payload[];
};

struct RecordHeader {
    unsigned long key;
    size_t original_index;
};

// --- Funzione per creare record casuali ---
Record* createRandomRecord(unsigned int payload_min, unsigned int payload_max) {
    static thread_local std::mt19937_64 rng(std::random_device{}() ^ (std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_int_distribution<unsigned long> key_dist(1, 100);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<unsigned int> size_dist(payload_min, payload_max);

    unsigned int payload_size = size_dist(rng);
    Record* rec = (Record*) malloc(sizeof(Record) + payload_size);
    if(!rec) return nullptr;

    rec->key = key_dist(rng);
    rec->len = payload_size;
    for(unsigned int i=0; i<payload_size; i++)
        rec->payload[i] = static_cast<char>(byte_dist(rng));

    return rec;
}

// --- Funzione per generare e salvare record in buffer locale con OpenMP ---
std::vector<char> saveRecordsToBuffer(int n, unsigned int payload_max, int num_threads) {
    int num_threads_effettivi;
    #pragma omp parallel num_threads(num_threads)
    #pragma omp single
        num_threads_effettivi = omp_get_num_threads();

    std::vector<std::vector<char>> thread_buffers(num_threads_effettivi);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        auto& local_buffer = thread_buffers[id];

        #pragma omp for
        for(int i=0; i<n; i++) {
            Record* rec = createRandomRecord(8, payload_max);

            size_t offset = local_buffer.size();
            local_buffer.resize(offset + sizeof(rec->key) + sizeof(rec->len) + rec->len);

            std::memcpy(local_buffer.data() + offset, &rec->key, sizeof(rec->key));
            offset += sizeof(rec->key);

            std::memcpy(local_buffer.data() + offset, &rec->len, sizeof(rec->len));
            offset += sizeof(rec->len);

            std::memcpy(local_buffer.data() + offset, rec->payload, rec->len);
            free(rec);
        }
    }

    // Concateno i buffer dei thread
    std::vector<char> final_buffer;
    size_t total_size = 0;
    for(auto& buf : thread_buffers)
        total_size += buf.size();
    final_buffer.reserve(total_size);
    for(auto& buf : thread_buffers)
        final_buffer.insert(final_buffer.end(), buf.begin(), buf.end());

    return final_buffer;
}

// --- Funzione per leggere i record da file ---
std::vector<Record*> loadRecordsFromFile(const std::string& filename) {
    std::vector<Record*> records;
    std::ifstream in(filename, std::ios::binary);
    if(!in) { cerr << "Errore apertura file " << filename << "\n"; return records; }

    while(true) {
        unsigned long key;
        unsigned int len;

        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        if(!in) break;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));

        Record* rec = (Record*) malloc(sizeof(Record) + len);
        rec->key = key;
        rec->len = len;
        in.read(reinterpret_cast<char*>(rec->payload), len);

        records.push_back(rec);
    }
    return records;
}

// --- MergeSort parallelo per RecordHeader ---
void mergeMPI(vector<RecordHeader>& arr, int left, int mid, int right, vector<RecordHeader>& temp) {
    int i = left, j = mid+1, k = left;
    while(i <= mid && j <= right) {
        if(arr[i].key <= arr[j].key) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while(i <= mid) temp[k++] = arr[i++];
    while(j <= right) temp[k++] = arr[j++];
    #pragma omp parallel for
    for(int idx=left; idx<=right; idx++) arr[idx] = temp[idx];
}

void mergeSortParMPI(vector<RecordHeader>& arr, int left, int right, vector<RecordHeader>& temp) {
    if(left >= right) return;
    int mid = left + (right-left)/2;
    #pragma omp task shared(arr, temp)
        mergeSortParMPI(arr, left, mid, temp);
    #pragma omp task shared(arr, temp)
        mergeSortParMPI(arr, mid+1, right, temp);
    #pragma omp taskwait
    mergeMPI(arr, left, mid, right, temp);
}

// --- Main ---
int main(int argc, char* argv[]) {
    srand(time(NULL));
    unsigned int PAYLOAD_MAX = 1024;
    unsigned long array_size = 1000;
    unsigned int num_threads = 16;

    int opt;
    while((opt = getopt(argc, argv, "s:t:p:")) != -1) {
        switch(opt) {
            case 's': {
                std::string s(optarg);
                char last_char = s.back();
                unsigned long multiplier = 1;
                if(last_char=='M'||last_char=='m'){ multiplier=1e6; s.pop_back();}
                else if(last_char=='K'||last_char=='k'){ multiplier=1e3; s.pop_back();}
                array_size = std::stoul(s) * multiplier;
                break;
            }
            case 't': num_threads = std::stoi(optarg); break;
            case 'p': PAYLOAD_MAX = std::stoul(optarg); break;
            default: 
                cout << "Usage: ./program -s array_size -t num_threads -p PAYLOAD_MAX\n";
                return 1;
        }
    }

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ==================== GENERAZIONE RECORD ====================
    int base = array_size / size;
    int resto = array_size % size;

    int my_records = base + (rank==0 ? resto : 0);

    auto local_buffer = saveRecordsToBuffer(my_records, PAYLOAD_MAX, num_threads);

    if(rank==0) {
        std::ofstream out("records.bin", std::ios::binary);
        if(!out){ cerr << "Errore apertura file\n"; MPI_Abort(MPI_COMM_WORLD,1);}
        out.write(local_buffer.data(), local_buffer.size());

        // ricezione dai worker
        for(int src=1; src<size; src++){
            int n_send;
            MPI_Recv(&n_send, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<char> recv_buf(n_send);
            MPI_Recv(recv_buf.data(), n_send, MPI_BYTE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            out.write(recv_buf.data(), recv_buf.size());
        }
        out.close();
    } else {
        int buf_size = local_buffer.size();
        MPI_Send(&buf_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_buffer.data(), buf_size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    }

    // ==================== ORDINAMENTO MPI+OpenMP ====================
    TIMERSTART(_SORT_)
    std::vector<Record*> records;
    int mpi_records_dim = 0;
    if(rank==0){
        records = loadRecordsFromFile("records.bin");
        mpi_records_dim = records.size();
    }
    MPI_Bcast(&mpi_records_dim,1,MPI_INT,0,MPI_COMM_WORLD);

    // calcolo chunk per ogni rank
    int chunk_size = mpi_records_dim / size;
    int remainder = mpi_records_dim % size;
    std::vector<int> sendcounts(size), displs(size);
    for(int i=0;i<size;i++){
        sendcounts[i] = chunk_size + (i==0 ? remainder : 0);
        displs[i] = (i==0) ? 0 : displs[i-1]+sendcounts[i-1];
    }
    int my_chunk_size = sendcounts[rank];

    // Rank 0 crea headers globali
    std::vector<RecordHeader> all_headers;
    if(rank==0){
        all_headers.resize(mpi_records_dim);
        #pragma omp parallel for
        for(int i=0;i<mpi_records_dim;i++){
            all_headers[i].key = records[i]->key;
            all_headers[i].original_index = i;
        }
    }

    // Scatterv per distribuire chunk di RecordHeader
    std::vector<RecordHeader> local_headers(my_chunk_size);
    MPI_Datatype MPI_RecordHeader;
    MPI_Type_contiguous(sizeof(RecordHeader), MPI_BYTE, &MPI_RecordHeader);
    MPI_Type_commit(&MPI_RecordHeader);
    MPI_Scatterv(all_headers.data(), sendcounts.data(), displs.data(),
                 MPI_RecordHeader, local_headers.data(), my_chunk_size,
                 MPI_RecordHeader, 0, MPI_COMM_WORLD);

    // --- Ordinamento locale con OpenMP ---
    std::vector<RecordHeader> tempMPIHead(my_chunk_size);
    #pragma omp parallel
    #pragma omp single
        mergeSortParMPI(local_headers,0,my_chunk_size-1,tempMPIHead);

    // --- Creazione vettore indici ordinati locali ---
    std::vector<int> local_indices(my_chunk_size);
    for(int i=0;i<my_chunk_size;i++)
        local_indices[i] = local_headers[i].original_index;

    // --- Gatherv indici ordinati sul rank 0 ---
    std::vector<int> gathered_indices;
    if(rank==0) gathered_indices.resize(mpi_records_dim);
    MPI_Gatherv(local_indices.data(), my_chunk_size, MPI_INT,
                gathered_indices.data(), sendcounts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    // --- Merge finale sul rank 0 ---
    if(rank==0){
        std::vector<Record*> sorted_records(mpi_records_dim);
        for(int i=0;i<mpi_records_dim;i++)
            sorted_records[i] = records[gathered_indices[i]];

        std::vector<Record*> final_sorted;
        final_sorted.insert(final_sorted.end(), sorted_records.begin(), sorted_records.begin()+sendcounts[0]);
        int offset = sendcounts[0];

        for(int i=1;i<size;i++){
            int current_chunk = sendcounts[i];
            std::vector<Record*> merged;
            merged.reserve(final_sorted.size()+current_chunk);

            std::merge(final_sorted.begin(), final_sorted.end(),
                       sorted_records.begin()+offset,
                       sorted_records.begin()+offset+current_chunk,
                       std::back_inserter(merged),
                       [](Record* a, Record* b){ return a->key < b->key; });

            final_sorted.swap(merged);
            offset += current_chunk;
        }

        // Controllo ordinamento
        for(size_t i=1;i<final_sorted.size();i++)
            if(final_sorted[i-1]->key>final_sorted[i]->key)
                cerr<<"Errore ordinamento MPI+OpenMP!"<<endl;

        std::cout<<"====== ORDINAMENTO MPI+OpenMP OK! ======"<<std::endl;

        for(Record* r:records) free(r);
    }
    TIMERSTOP(_SORT_)

    MPI_Type_free(&MPI_RecordHeader);
    MPI_Finalize();
    
    return 0;
}
