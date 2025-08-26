#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <vector>
#include <getopt.h>
#include <future>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm> 
#include <cstring> 
#include "fastflow/ff/ff.hpp"
#include "fastflow/ff/dc.hpp"

#include "hpc_helpers.hpp"   

// Direttive OpenOMP -> usare (-fopenmp) per compilare
#include <omp.h>
#include <mpi.h>

using namespace ff;
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

        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!in) break;

        in.read(reinterpret_cast<char*>(&len), sizeof(len));

        // Alloco solo l'header, senza payload
        Record* rec = (Record*) malloc(sizeof(Record));
        rec->key = key;
        rec->len = len;
        // rec->payload non viene letto, risparmio memoria e tempo

        records.push_back(rec);

        // Salto i byte del payload
        in.seekg(len, std::ios::cur);
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


// ========= FAST FLOW =========

struct workersFileCreation : ff_node {
    int record;
    unsigned int payload_max;

    workersFileCreation(int record, unsigned int payload_max)
        : record(record), payload_max(payload_max) {}

    void* svc(void* task) override {
        std::vector<Record*> createdRecord;
        for (int i = 0; i < record; i++) {
            Record* rec = createRandomRecord(8, payload_max);
            if (!rec) {
                std::cerr << "Errore: malloc fallito\n";
                continue;
            }
            if (rec->len > payload_max) {
                std::cerr << "Errore: payload fuori range (" << rec->len << ")\n";
                free(rec);
                continue;
            }

            // Buffer temporaneo per header + payload
            auto* buffer = new std::vector<char>(sizeof(rec->key) + sizeof(rec->len) + rec->len);

            size_t offset = 0;

            // Copio la key
            std::memcpy(buffer->data() + offset, &rec->key, sizeof(rec->key));
            offset += sizeof(rec->key);

            // Copio la lunghezza
            std::memcpy(buffer->data() + offset, &rec->len, sizeof(rec->len));
            offset += sizeof(rec->len);

            // Copio il payload
            std::memcpy(buffer->data() + offset, rec->payload, rec->len);
            offset += rec->len;
            // std::cout << "[Worker " << ff_node::get_my_id()
            //           << "] generato record " << i
            //           << " con payload " << rec->len << "\n";
            free(rec);
            // Invio tutto al collector per scrivere in maniera sequenziale sul file
            // Passo il puntatore al buffer già creato nell'heap -> risparmio, NussunaCopia
            ff_send_out(buffer);
        }
        return GO_ON; // worker non finisce, continua a elaborare altri task
    }
};

struct CreationCollector : ff_node {
    MPI_File mpi_file;
    MPI_Offset offset = 0;   // Offset corrente nel file
    MPI_Win win;             // (opzionale per gestione condivisa)

    CreationCollector(MPI_File file) : mpi_file(file) {}

    void* svc(void* task) override {
        auto* partial = (std::vector<char>*) task;

        // Scrittura ordinata (MPI mantiene l’ordine fra processi)
        MPI_Status status;
        MPI_File_write_ordered(mpi_file, partial->data(), 
                               partial->size(), MPI_CHAR, &status);

        delete partial;
        return GO_ON;
    }
};



// --- Emitter: divide l'array in blocchi di chunk_size ---
struct EmitterSortParallel : ff_node {
    std::vector<RecordHeader>& arr;
    int chunk_size;

    EmitterSortParallel(std::vector<RecordHeader>& arr, int chunk_size)
        : arr(arr), chunk_size(chunk_size) {}

    void* svc(void*) override {
        for (size_t i = 0; i < arr.size(); i += chunk_size) {
            // Per gestire anche il caso in cui l'array non sia divisibile per chunk_size
            // end ha dimensione il minimo tra la posizione nell'array + chunk_size e la dimensione massima rimasta nell'array nel caso non sia divisibile
            size_t end = std::min(i + chunk_size, arr.size());
            // In questa riga creo il vettore di dimensione chunk_size
            // Sommando i ottengo un iteratore all'elemento di indice i.
            // arr.begin() restituisce un iteratore al primo elemento sommandoci i mi sposto tra tutti gli elementi fino a arr.size
            // arr.begin() + end restituisce invece l'iteratore alla posizione successiva all'ultimo elemento del blocco
            // In questo modo costruisco un nuovo vettore copiando gli elementi da indice i fino a end-1.
            auto* block = new std::vector<RecordHeader>(arr.begin() + i, arr.begin() + end);
            // Manda il blocco da ordinare ai worker
            ff_send_out(block);
        }
        return EOS;
    }
};

// --- Worker: ordina un blocco ---
struct WorkerSortParallel : ff_node {
    void* svc(void* task) override {
        std::vector<RecordHeader>* block = (std::vector<RecordHeader>*) task;

        std::sort(block->begin(), block->end(), [](RecordHeader a, RecordHeader b) {
            return a.key < b.key;
        });

        return block; // restituisce il blocco ordinato
    }
};

// --- Collector: merge incrementale due-a-due ---
struct CollectorSortParallel : ff_node {
    std::vector<RecordHeader>& result;

    CollectorSortParallel(std::vector<RecordHeader>& result)
        : result(result) {}

    void* svc(void* task) override {
        auto* block = static_cast<std::vector<RecordHeader>*>(task);

        if (result.empty()) {
            result = *block;
        } else {
            std::vector<RecordHeader> merged;
            merged.reserve(result.size() + block->size());

            size_t i = 0, j = 0;
            while (i < result.size() && j < block->size()) {
                if (result[i].key <= (*block)[j].key)
                    merged.push_back(result[i++]);
                else
                    merged.push_back((*block)[j++]);
            }
            while (i < result.size()) merged.push_back(result[i++]);
            while (j < block->size()) merged.push_back((*block)[j++]);

            result.swap(merged);
        }

        delete block;
        return GO_ON;
    }
};



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

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "records_mpi.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // Posso dividere anche la scrittura del file in più processi
    // Dentro ognuno genero chunk_size record che scrivo nel file
    // La scrittura su file però va effettuata nel rank 0 sennò ho comportamenti indefiniti

    int base = array_size / size;
    int resto = array_size % size;

    int n_records = (rank == 0 ? base + resto : base);

    TIMERSTART(saveRecordsToFile_MPI);

    std::vector<ff_node*> workerVectorCreation;

    ff_farm farmCreation;

    for (int i = 0; i < num_threads; ++i) {
        // Per gestire casi in cui array size non è divisibile
        workerVectorCreation.push_back(new workersFileCreation(n_records, PAYLOAD_MAX));
    }
    CreationCollector* collectorCreation = new CreationCollector(fh);
    farmCreation.add_workers(workerVectorCreation);
    farmCreation.add_collector(collectorCreation);
    printf("-----------------\n");
    
    //TIMERSTART(farmCreationParallel);
    if (farmCreation.run_and_wait_end() < 0) {
        std::cerr << "Errore avvio farm\n";
        return -1;
    }
    //TIMERSTOP(farmCreationParallel);
    MPI_File_close(&fh);

    if(rank == 0) std::cout << "Scrittura completata!" << std::endl;

    TIMERSTOP(saveRecordsToFile_MPI);
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
    int remainder_copy = remainder;

    for (int i = 0; i < size; i++) {
        if (remainder_copy > 0) {
            sendcounts[i] = base_chunk + 1;
            remainder_copy--;
        } else {
            sendcounts[i] = base_chunk;
        }
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    // =======================
    // Distribuzione chunk size a tutti i rank
    int my_chunk_size = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &my_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::vector<Record*> recordsMPI_FF = records;
        // ============================================
        // =============== MERGE MPI ==================
        // ============================================
        int mpi_records_dim = recordsMPI_FF.size();
        //std::cout << "Dimensione totale dei record MPI: " << mpi_records_dim << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "========= MPI+OpenMP ==========" << std::endl;
        std::cout << "===============================" << std::endl;
        // ------------------ CALCOLO CHUNK ------------------
        int my_chunk_size = sendcounts[0];  // rank 0 elabora chunk + resto

        // Creazione array globale di header
        std::vector<RecordHeader> all_headers(mpi_records_dim);
        for (int i = 0; i < mpi_records_dim; i++) {
            all_headers[i].key = recordsMPI_FF[i]->key;
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

        TIMERSTART(MPI_Recv_OriginalIndex_e_Ordinamento);
        for(int i = 1; i < size; i++) {
            std::vector<int> original_indices_Sorted(sendcounts[i]);
            MPI_Irecv(original_indices_Sorted.data(), sendcounts[i], MPI_INT, i, 2, MPI_COMM_WORLD, &requests[i-1]);recv_buffers[i-1] = std::move(original_indices_Sorted);
            //std::cout << "============ DATI RICEVUTI ================ "<<" DATI: " << original_indices_Sorted.size()<<"\n";
        }

        //TIMERSTART(Oridnamento_RANK_0)
        // Ordina chunk locale rank 0
        std::vector<RecordHeader> headers0(my_chunk_size);
        for(int i=0; i<my_chunk_size; i++)
            headers0[i] = all_headers[i];


        ff_farm mergeSortParalleloFarm;
        EmitterSortParallel* EmitterSort = new EmitterSortParallel(headers0, my_chunk_size);

        std::vector<ff_node*> workerSort;
        for (int i = 0; i < num_threads; ++i) {
            workerSort.push_back(new WorkerSortParallel());
        }

        std::vector<RecordHeader> result;
        CollectorSortParallel* CollectorSort = new CollectorSortParallel(result);
        // CollectorMergeTree* CollectorSort = new CollectorMergeTree(result);
        mergeSortParalleloFarm.add_emitter(EmitterSort);
        mergeSortParalleloFarm.add_workers(workerSort);
        mergeSortParalleloFarm.add_collector(CollectorSort);

        if (mergeSortParalleloFarm.run_and_wait_end() < 0) {
            std::cerr << "Errore avvio farm\n";
            return -1;
        }

        for(int i=0; i<my_chunk_size; i++)
            sorted_records[i] = recordsMPI_FF[result[i].original_index];
        //TIMERSTOP(Oridnamento_RANK_0);


        // Poi aspetti che tutte le ricezioni siano completate
        MPI_Waitall(size - 1, requests.data(), MPI_STATUSES_IGNORE);

        // Copi nei vettori globali solo dopo che sei sicuro che i dati sono arrivati
        int start = sendcounts[0];
        for(int i = 1; i < size; i++) {
            for(int j = 0; j < sendcounts[i]; j++)
                sorted_records[start + j] = recordsMPI_FF[recv_buffers[i-1][j]];
            start += sendcounts[i];
        }


        TIMERSTOP(MPI_Recv_OriginalIndex_e_Ordinamento);

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
                cerr << "Errore ordinamento MPI+FF!" << endl;
            }
        std::cout<<"====== ORDINAMENTO MPI+FF OK! ======"<<std::endl;

        //printRecords(final_sorted);

        for (Record* r : records) free(r);
    }

    if (rank != 0) {
        // int T_start = MPI_Wtime();
        
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

        TIMERSTART(Ordinamento_MPI_FF_SubProcess);

        ff_farm mergeSortParalleloFarm;
        EmitterSortParallel* EmitterSort = new EmitterSortParallel(local_headers, my_chunk_size);

        std::vector<ff_node*> workerSort;
        for (int i = 0; i < num_threads; ++i) {
            workerSort.push_back(new WorkerSortParallel());
        }

        std::vector<RecordHeader> result;
        CollectorSortParallel* CollectorSort = new CollectorSortParallel(result);
        // CollectorMergeTree* CollectorSort = new CollectorMergeTree(result);
        mergeSortParalleloFarm.add_emitter(EmitterSort);
        mergeSortParalleloFarm.add_workers(workerSort);
        mergeSortParalleloFarm.add_collector(CollectorSort);

        if (mergeSortParalleloFarm.run_and_wait_end() < 0) {
            std::cerr << "Errore avvio farm\n";
            return -1;
        }

        // INVIO solo il vettore ordinato degli indici ordinati
        std::vector<int> original_indices(result.size());
        for (size_t i = 0; i < result.size(); i++)
            original_indices[i] = result[i].original_index;

        MPI_Send(original_indices.data(), original_indices.size(), MPI_INT, 0, 2, MPI_COMM_WORLD);


        TIMERSTOP(Ordinamento_MPI_FF_SubProcess);
    }

    MPI_Type_free(&MPI_RecordHeader);
    MPI_Finalize();
    return 0;
}
