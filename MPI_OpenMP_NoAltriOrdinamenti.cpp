#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <vector>
#include <getopt.h>
#include <future>
#include <iostream>
#include <fstream>
#include <random>
#include "hpc_helpers.hpp"   

// Direttive OpenOMP -> usare (-fopenmp) per compilare
#include <omp.h>
#include <mpi.h>


using namespace std;

struct Record {
    unsigned long key; // 8 byte
    uint32_t len;      // lunghezza payload -> quanti byte sono usati per il payload
    char payload[];    // flessibile
};

struct RecordHeader {
    unsigned long key;
    size_t original_index;  // posizione nel vettore originale
};


    // Alloco e Creo il singolo record in memoria -> per ogni record devo usarla
Record* createRandomRecord(unsigned int payload_min, unsigned int payload_max) {
    // Ogni thread ha il suo rng indipendente
    static thread_local std::mt19937_64 rng(std::random_device{}() ^ (std::hash<std::thread::id>{}(std::this_thread::get_id())));

    std::uniform_int_distribution<unsigned long> key_dist(1, 100);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<unsigned int> size_dist(payload_min, payload_max);

    unsigned int payload_size = size_dist(rng);

    Record* rec = (Record*) malloc(sizeof(Record) + payload_size);
    if (!rec) return nullptr;

    rec->key = key_dist(rng);
    rec->len = payload_size;

    for (unsigned int i = 0; i < payload_size; i++) {
        rec->payload[i] = static_cast<char>(byte_dist(rng));
    }
    return rec;
}


std::vector<char> saveRecordsToFile(int n, unsigned int payload_max, int num_thread) {

    // Ogni thread costruisce un buffer locale
    std::vector<std::vector<char>> thread_buffers(num_thread);

    #pragma omp parallel num_threads(num_thread)
    {
        
        int id = omp_get_thread_num();
        // int n_threads = omp_get_num_threads();  // Numero di thread attivi in questo team

        // #pragma omp single
        //     std::cout << "Numero di thread in uso: " << n_threads << std::endl;
        auto& local_buffer = thread_buffers[id];

        #pragma omp for
        for (int i = 0; i < n; i++) {
            Record* rec = createRandomRecord(8, payload_max); // Record creato

            size_t offset = local_buffer.size(); // Creo offset per sapere da dove partire a scrivere i dati e ad ogni iterazione sposto l'offset della dimensione dei dati contenuti in local buffer
            local_buffer.resize(offset + sizeof(rec->key) + sizeof(rec->len) + rec->len);

            std::memcpy(local_buffer.data() + offset, &rec->key, sizeof(rec->key)); // Scrivo la chiave nel buffer
            offset += sizeof(rec->key);

            std::memcpy(local_buffer.data() + offset, &rec->len, sizeof(rec->len)); // Scrivo la lunghezza del payload nel buffer
            offset += sizeof(rec->len);

            std::memcpy(local_buffer.data() + offset, rec->payload, rec->len); // Scrivo 

            free(rec);
        }
    }
    // Concateno i buffer dei thread
    std::vector<char> final_buffer;
    size_t total_size = 0;
    for (auto& buf : thread_buffers) {
        total_size += buf.size();
    }
    final_buffer.reserve(total_size);

    for (auto& buf : thread_buffers) {
        final_buffer.insert(final_buffer.end(), buf.begin(), buf.end());
    }

    return final_buffer;
}

std::vector<Record*> loadRecordsFromFile(const std::string& filename) {
    std::vector<Record*> records;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Errore apertura file " << filename << "\n";
        return records;
    }

    while (true) {
        unsigned long key;
        unsigned int len;

        // Leggo header
        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!in) break; // fine file

        in.read(reinterpret_cast<char*>(&len), sizeof(len));

        // Alloco record in memoria
        Record* rec = (Record*) malloc(sizeof(Record) + len);
        rec->key = key;
        rec->len = len;

        // Leggo payload
        in.read(reinterpret_cast<char*>(rec->payload), len);

        records.push_back(rec);
    }

    in.close();
    return records;
}

void standardSeqSort(vector<Record*>& arr) {
    std::sort(arr.begin(), arr.end(), [](Record* a, Record* b) {
        return a->key < b->key;
    });
}
    
// --- Merge per array di Record* ---
void merge(vector<Record*>& arr, int left, int mid, int right, vector<Record*>& temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i]->key <= arr[j]->key)
        {
            temp[k] = arr[i];
            k++;
            i++;
        }
        else
        {
            temp[k] = arr[j];
            k++;
            j++;
        }
    }
    while (i <= mid) 
    {
        temp[k] = arr[i];
        k++;
        i++;
    }
    while (j <= right)
    {
        temp[k] = arr[j];
        k++;
        j++;
    }
    #pragma omp parallel for
    for (int idx = left; idx <= right; idx++)
        arr[idx] = temp[idx];
}

void mergeSortPar(vector<Record*>& arr, int left, int right, vector<Record*>& temp) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    // Crea un task che utilizza i thread creati quando chiamo #pragma omp parrallel
    // Se lo utilizzo qui ogni volta che effettuo una ricorsione creo un team di thread

    // Devo condividere le variabili arr e temp dato che le due parti eseguite dai thread modificano l'array e temp
    #pragma omp task shared(arr, temp)
    mergeSortPar(arr, left, mid, temp);
    #pragma omp task shared(arr, temp)
    mergeSortPar(arr, mid + 1, right, temp);
    // Aspetta che tutte le task precedenti siano completate
    // Aspetta che tutte le due parti dell'array siano ordinate prima di fare il merge altrimenti il merge non funziona
    #pragma omp taskwait
    merge(arr, left, mid, right, temp);
}

// --- MergeSort SEQ ----
void mergeSortSeq(vector<Record*>& arr, int left, int right, vector<Record*>& temp) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    #pragma omp parallel sections
    {
        #pragma omp section
        mergeSortSeq(arr, left, mid, temp);
        #pragma omp section
        mergeSortSeq(arr, mid + 1, right, temp);
    }
    merge(arr, left, mid, right, temp);
}

#include <iomanip> // per std::hex e std::setw

void printRecords(const std::vector<Record*>& records, size_t max_payload_bytes = 16) {
    for (size_t i = 0; i < records.size(); i++) {
        Record* rec = records[i];
        std::cout << "Record " << i
                  << " | key: " << rec->key
                  << " | len: " << rec->len
                  << " | payload (primi " << max_payload_bytes << " byte): ";

        // stampo i primi max_payload_bytes in esadecimale
        for (size_t j = 0; j < rec->len && j < max_payload_bytes; j++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << (static_cast<unsigned int>(static_cast<unsigned char>(rec->payload[j]))) << " ";
        }

        std::cout << std::dec << "\n"; // reset a decimale
    }
}


void saveRecordsToFile2(const std::string& filename, int n, unsigned int payload_max) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Errore apertura file " << filename << "\n";
        return;
    }

    for (int i = 0; i < n; i++) {
        Record* rec = createRandomRecord(8, payload_max);

        // Buffer temporaneo per header + payload
        std::vector<char> buffer(sizeof(rec->key) + sizeof(rec->len) + rec->len);

        size_t offset = 0;

        // Copio la key nel buffer
        std::memcpy(buffer.data() + offset, &rec->key, sizeof(rec->key));
        offset += sizeof(rec->key);

        // Copio la lunghezza
        std::memcpy(buffer.data() + offset, &rec->len, sizeof(rec->len));
        offset += sizeof(rec->len);

        // Copio il payload
        std::memcpy(buffer.data() + offset, rec->payload, rec->len);

        // Scrivo il buffer nel file in un colpo solo
        out.write(buffer.data(), buffer.size());

        free(rec); // libero subito la memoria
    }

    
    std::cout << "File creato con " << n << " record.\n";
}

    // --- Merge per array di Record* ---
void mergeMPI(vector<RecordHeader>& arr, int left, int mid, int right, vector<RecordHeader>& temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i].key <= arr[j].key)
        {
            temp[k] = arr[i];
            k++;
            i++;
        }
        else
        {
            temp[k] = arr[j];
            k++;
            j++;
        }
    }
    while (i <= mid) 
    {
        temp[k] = arr[i];
        k++;
        i++;
    }
    while (j <= right)
    {
        temp[k] = arr[j];
        k++;
        j++;
    }
    #pragma omp parallel for
    for (int idx = left; idx <= right; idx++)
        arr[idx] = temp[idx];
}

void mergeSortParMPI(vector<RecordHeader>& arr, int left, int right, vector<RecordHeader>& temp) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    // Crea un task che utilizza i thread creati quando chiamo #pragma omp parrallel
    // Se lo utilizzo qui ogni volta che effettuo una ricorsione creo un team di thread

    // Devo condividere le variabili arr e temp dato che le due parti eseguite dai thread modificano l'array e temp
    #pragma omp task shared(arr, temp)
    mergeSortParMPI(arr, left, mid, temp);
    #pragma omp task shared(arr, temp)
    mergeSortParMPI(arr, mid + 1, right, temp);
    // Aspetta che tutte le task precedenti siano completate
    // Aspetta che tutte le due parti dell'array siano ordinate prima di fare il merge altrimenti il merge non funziona
    #pragma omp taskwait
    mergeMPI(arr, left, mid, right, temp);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    unsigned int PAYLOAD_MAX = 1024; // default

    unsigned long array_size = 1000; 
    unsigned int num_threads = 16;

    int opt;
    while ((opt = getopt(argc, argv, "s:t:p:")) != -1) {
        switch (opt) {
            case 's': {
                std::string s(optarg);
                char last_char = s.back();
                unsigned long multiplier = 1;
                if (last_char == 'M' || last_char == 'm') {
                    multiplier = 1000 * 1000;
                    s.pop_back();
                } else if (last_char == 'K' || last_char == 'k') {
                    multiplier = 1000;
                    s.pop_back();
                }
                array_size = std::stoul(s) * multiplier;
                break;
            }
            case 't':
                num_threads = std::stoi(optarg);
                break;
            case 'p': // massimo payload consentito
                PAYLOAD_MAX = std::stoul(optarg);
                break;
            default:
                std::cout << "Usage: ./program -s array_size -t num_threads -p PAYLOAD_MAX\n";
                return 1;
        }
    }

    MPI_Init(&argc, &argv);

    int rank, size;    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Posso dividere anche la scrittura del file in più processi
    // Dentro ognuno genero chunk_size record che scrivo nel file
    // La scrittura su file però va effettuata nel rank 0 sennò ho comportamenti indefiniti
    
    if (rank == 0)
    {
        TIMERSTART(saveRecordsToFile_MainProcess);
        // APERTURA FILE
        std::ofstream out("records.bin", std::ios::binary);
        if (!out) {
            std::cerr << "Errore apertura file - records.bin\n";
            return 0;
        }

        // Divido i valori da generare per il numero di processi MPI
        int base = array_size / size;
        int resto = array_size % size;

        for (int i = 1; i < size; i++) {
            int n_records = base;
            MPI_Send(&n_records, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Record creati dal rank 0
        int my_records = base+resto;
        // Genero my_records numero di record
        auto local_buffer = saveRecordsToFile(my_records, PAYLOAD_MAX, num_threads);
        out.write(local_buffer.data(), local_buffer.size());

        // Ricevo dai worker
        for (int src = 1; src < size; src++) {
            int buf_size;
            // Ricevo la dimensione del buffer che mi sta per mandare il worker
            MPI_Recv(&buf_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Alloco il buffer giusto
            std::vector<char> recv_buffer(buf_size);

            // Ricevo i dati
            MPI_Recv(recv_buffer.data(), buf_size, MPI_BYTE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Scrivo direttamente su file
            out.write(recv_buffer.data(), recv_buffer.size());
        }
        out.close();
        TIMERSTOP(saveRecordsToFile_MainProcess);

        // TIMERSTART(saveRecordsToFile2);
        // // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
        // // Generazione record casuali
        // saveRecordsToFile2("records2.bin", array_size, PAYLOAD_MAX);
        // TIMERSTOP(saveRecordsToFile2);
    }
    else {
        TIMERSTART(saveRecordsToFile_SubProcess);
        // Altri processi non principale
        // Devono creare i record e mandarli al rank 0
        int n_records;
        // Ricevo dal rank 0 quanti record devo generare
        MPI_Recv(&n_records, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto local_buffer = saveRecordsToFile(n_records, PAYLOAD_MAX, num_threads);

        int buf_size = local_buffer.size();
        // 1. invio la dimensione del buffer
        MPI_Send(&buf_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // 2. invio i dati
        MPI_Send(local_buffer.data(), buf_size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        TIMERSTOP(saveRecordsToFile_SubProcess);
    }


    // TIMERSTART(saveRecordsToFile);
    // // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
    // // Generazione record casuali
    // saveRecordsToFile("records.bin", array_size, PAYLOAD_MAX);
    // TIMERSTOP(saveRecordsToFile);

    MPI_Datatype MPI_RecordHeader;
    MPI_Type_contiguous(sizeof(RecordHeader), MPI_BYTE, &MPI_RecordHeader);
    MPI_Type_commit(&MPI_RecordHeader);

    int mpi_records_dim = 0;
    std::vector<Record*> records;
    if (rank == 0) {
        records = loadRecordsFromFile("records.bin"); // solo root legge
        mpi_records_dim = records.size();
        // eventualmente crea all_headers qui
    }

    // Tutti i rank ricevono la dimensione totale -> senza caricare il file
    MPI_Bcast(&mpi_records_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // =======================
    // Calcolo chunk
    int base_chunk = mpi_records_dim / size;
    int remainder  = mpi_records_dim % size;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_chunk + (i == 0 ? remainder : 0); // rank 0 prende resto
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    // =======================
    // Distribuzione chunk size a tutti i rank
    int my_chunk_size = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &my_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::vector<Record*> recordsMPI_OpenMP = records;
        // ============================================
        // =============== MERGE MPI ==================
        // ============================================
    int mpi_records_dim = recordsMPI_OpenMP.size();
    //std::cout << "Dimensione totale dei record MPI: " << mpi_records_dim << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "========= MPI+OpenMP ==========" << std::endl;
    std::cout << "===============================" << std::endl;
    // ------------------ CALCOLO CHUNK ------------------
    int my_chunk_size = sendcounts[0];  // rank 0 elabora chunk + resto

    // Creazione array globale di header
    std::vector<RecordHeader> all_headers(mpi_records_dim);
    #pragma omp parallel for
    for (int i = 0; i < mpi_records_dim; i++) {
        all_headers[i].key = recordsMPI_OpenMP[i]->key;
        all_headers[i].original_index = i;
    }

    TIMERSTART(MPI_Send_Headers);


    std::vector<RecordHeader> local_headers(my_chunk_size);
    MPI_Scatterv(
        all_headers.data(),             // root buffer
        sendcounts.data(),              // numero elementi per rank
        displs.data(),                  // offset per rank
        MPI_RecordHeader,               // tipo dati
        local_headers.data(),           // buffer locale
        my_chunk_size,                  // numero elementi locali
        MPI_RecordHeader,               // tipo dati
        0,                              // root
        MPI_COMM_WORLD
    );


    TIMERSTOP(MPI_Send_Headers);

// ------------------ ORDINE LOCALE ------------------
    // Costruzione vettore globale
    std::vector<Record*> sorted_records(mpi_records_dim);
    // Per fare la ricezione dei dati ordinati -> con Irecv
    std::vector<MPI_Request> requests(size-1);
    std::vector<std::vector<int>> recv_buffers(size-1);

    TIMERSTART(MPI_Recv_OriginalIndex_eOrdinamento);
    for(int i = 1; i < size; i++) {
        int recv_chunk = sendcounts[i];
        std::vector<int> original_indices_Sorted(recv_chunk);
        MPI_Irecv(original_indices_Sorted.data(), recv_chunk, MPI_INT, i, 2, MPI_COMM_WORLD, &requests[i-1]);recv_buffers[i-1] = std::move(original_indices_Sorted);
        //std::cout << "============ DATI RICEVUTI ================ "<<" DATI: " << original_indices_Sorted.size()<<"\n";
    }

    TIMERSTART(Oridnamento_RANK_0)
    // Ordina chunk locale rank 0
    std::vector<RecordHeader> headers0(my_chunk_size);
    for(int i=0; i<my_chunk_size; i++)
        headers0[i] = all_headers[i];

    std::vector<RecordHeader> tempMPIHead(my_chunk_size);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp single nowait
        mergeSortParMPI(headers0, 0, my_chunk_size-1, tempMPIHead);
    }

    for(int i=0; i<my_chunk_size; i++)
        sorted_records[i] = recordsMPI_OpenMP[headers0[i].original_index];
    TIMERSTOP(Oridnamento_RANK_0);


    // Poi aspetti che tutte le ricezioni siano completate
    MPI_Waitall(size - 1, requests.data(), MPI_STATUSES_IGNORE);

    // Copi nei vettori globali solo dopo che sei sicuro che i dati sono arrivati
    int start = my_chunk_size;
    for(int i = 1; i < size; i++) {
        for(int j = 0; j < sendcounts[i]; j++)
            sorted_records[start + j] = recordsMPI_OpenMP[recv_buffers[i-1][j]];
        start += sendcounts[i];
    }


    TIMERSTOP(MPI_Recv_OriginalIndex_eOrdinamento);

    TIMERSTART(Oridnamento_FINALE);
    std::vector<Record*> final_sorted;

    // copia iniziale del primo chunk
    final_sorted.insert(final_sorted.end(),
                        sorted_records.begin(),
                        sorted_records.begin() + my_chunk_size);

    // fusione iterativa
    int offset = sendcounts[0];
    for (int i = 1; i < size; i++) {

        std::vector<Record*> merged;
        merged.reserve(final_sorted.size() + sendcounts[i]);

        std::merge(final_sorted.begin(), final_sorted.end(),
                sorted_records.begin() + offset,
                sorted_records.begin() + offset + sendcounts[i],
                std::back_inserter(merged),
                [](Record* a, Record* b) { return a->key < b->key; });

        final_sorted.swap(merged);
        offset += sendcounts[i];
    }
    TIMERSTOP(Oridnamento_FINALE);

    for (size_t i = 1; i < final_sorted.size(); ++i)
        if (final_sorted[i-1]->key > final_sorted[i]->key) {
            cerr << "Errore ordinamento MPI+OpenMP!" << endl;
        }
    std::cout<<"====== ORDINAMENTO MPI+OpenMP OK! ======"<<std::endl;

    //printRecords(final_sorted);

    for (Record* r : records) free(r);
}

    if (rank != 0) {
        // int T_start = MPI_Wtime();
        TIMERSTART(Ordinamento_MPI_SubProcess);
        // Altri processi non principale
        // Devono ordinare i record dell'array di dimensione chunk_size
        // rank ricevente
        std::vector<RecordHeader> local_headers(my_chunk_size);
        MPI_Scatterv(
            nullptr,     // valido solo per rank 0
            sendcounts.data(),      
            displs.data(),          
            MPI_RecordHeader,
            local_headers.data(),   // ognuno riceve qui
            my_chunk_size,          // numero elementi locali
            MPI_RecordHeader,
            0,
            MPI_COMM_WORLD  
        );



        // Creiamo array di puntatori agli headers per mergeSortPar
        // std::vector<RecordHeader*> header_ptrs(recv_chunk);
        // #pragma omp parallel for
        // for(int i=0; i<recv_chunk; i++)
        //     header_ptrs[i] = &headers[i];

        std::vector<RecordHeader> tempMPIHead(my_chunk_size);

        #pragma omp parallel
        {
            // Questo è necessario per eseguire la funzione una sola volta
            #pragma omp single
            mergeSortParMPI(local_headers, 0, local_headers.size() - 1, tempMPIHead);
        }

        // INVIO solo il vettore ordinato degli indici ordinati
        std::vector<int> original_indices(my_chunk_size);
        for(int i = 0; i < my_chunk_size; i++)
            original_indices[i] = local_headers[i].original_index;

        MPI_Send(original_indices.data(), my_chunk_size, MPI_INT, 0, 2, MPI_COMM_WORLD);

        TIMERSTOP(Ordinamento_MPI_SubProcess);
        // int T_end = MPI_Wtime();
        // std::cout << "Tempo Ordinamento MPI SubProcess (rank " << rank << "): " << T_end - T_start << " secondi" << std::endl;
    }

    MPI_Type_free(&MPI_RecordHeader);
    MPI_Finalize();
    return 0;
}
