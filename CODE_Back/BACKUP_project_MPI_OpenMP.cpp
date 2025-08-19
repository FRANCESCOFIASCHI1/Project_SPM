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

    // Alloco e Creo il singolo record in memoria -> per ogni record devo usarla
Record* createRandomRecord(unsigned int payload_min, unsigned int payload_max) {
    // Ogni thread ha il suo rng indipendente
    static thread_local std::mt19937_64 rng(std::random_device{}() ^ 
                                            (std::hash<std::thread::id>{}(std::this_thread::get_id())));

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


std::vector<char> saveRecordsToFile(int n, unsigned int payload_max) {
    
    int num_threads = 1;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    // Ogni thread costruisce un buffer locale
    std::vector<std::vector<char>> thread_buffers(num_threads);

    #pragma omp parallel
    {
        
        int tid = omp_get_thread_num();
        auto& local_buffer = thread_buffers[tid];

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

// // --- MergeSort Parallelizzato MALE -> creazione ed eliminazione di thread crea overhead elevato ---
// void mergeSortSeq(vector<Record*>& arr, int left, int right, vector<Record*>& temp) {
//     if (left >= right) return;
//     int mid = left + (right - left) / 2;
//     #pragma omp parallel sections
//     {
//         #pragma omp section
//         mergeSortSeq(arr, left, mid, temp);
//         #pragma omp section
//         mergeSortSeq(arr, mid + 1, right, temp);
//     }
//     merge(arr, left, mid, right, temp);
// }
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

std::vector<Record*> mergeMPI_Tree(std::vector<Record*>& local_data, int rank, int size, MPI_Comm comm, int num_threads) {
    int step = 1;
    while (step < size) {
        // Se sono il processo pari, ricevo dal vicino per fare il merge
        if (rank % (2*step) == 0) {
            // Se il processo vicino esiste e non rimane un solo processo o dispari
            if (rank + step < size) {
                // Ricevo dal vicino
                // Prima ricevo la dimensione dei dati
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank+step, 0, comm, MPI_STATUS_IGNORE);
                std::vector<Record> recv_data(recv_size);

                MPI_Recv(recv_data.data(), recv_size * sizeof(Record), MPI_BYTE, rank+step, 1, comm, MPI_STATUS_IGNORE);
                
                // Creazione puntatori
                std::vector<Record*> recv_ptrs(recv_size);
                for (int i = 0; i < recv_size; i++)
                    recv_ptrs[i] = &recv_data[i];

                // Merge locale
                local_data = mergeTwoParallel(local_data, recv_ptrs, num_threads);
            }
        // Se sono il processo dispari, mando al vicino
        } else {
            // Mando al vicino
            int dest = rank - step;
            int send_size = local_data.size();
            MPI_Send(&send_size, 1, MPI_INT, dest, 0, comm);
            MPI_Send(local_data.data(), send_size * sizeof(Record), MPI_BYTE, dest, 1, comm);
            break; // Dopo aver inviato, esco dal ciclo
        }
        step *= 2;
    }
    return local_data;
}

std::vector<Record> mergeTwoParallel(const std::vector<Record>& first, const std::vector<Record>& second, int num_threads) {
    size_t n = first.size(), m = second.size();
    std::vector<Record> result(n + m);

    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        // Dato che non è garantito che vengano generati esattamente num_threads come richiesto 
        int n_thread  = omp_get_num_threads();

        // Ogni thread prende un blocco del risultato
        // Chunk_size ha dimensione size primo array + size secodo /num_threads
        // n_thread - 1 -> compensa il risultato della divisione rendendola intera
        size_t chunk_size = (n + m + n_thread - 1) / n_thread;
        // Ogni thread deve lavorare su un blocco diverso quindi lo prendiamo in base all'id del thread
        size_t start = id * chunk_size;
        // Con il minimo evitiamo di superare i limiti e prendiamo il resto che avanza
        size_t end   = std::min((id + 1) * chunk_size, n + m);

        if (start >= end) {
            // Trova la "frontiera" negli array FIRST e SECOND
            // Usa binary search per sapere quanti elementi di FIRST vanno prima del risultato[start]
            size_t first_low  = std::max((size_t)0, start > m ? start - m : (size_t)0);
            size_t first_high = std::min(start, n);

            while (first_low < first_high) {
                size_t mid = (first_low + first_high) / 2;
                if (start - mid > m || (start - mid < m && mid < n && first[mid] < second[start - mid]))
                    first_low = mid + 1;
                else
                    first_high = mid;
            }

            size_t i = first_low;
            size_t j = start - i;

            // Merge fino a riempire il blocco del risultato
            for (size_t k = start; k < end; k++) {
                if (i < n && (j >= m || first[i] < second[j]))
                    result[k] = first[i++];
                else
                    result[k] = second[j++];
            }
        }
    }

    return result;
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
        int my_records = resto;
        // Genero my_records numero di record
        auto local_buffer = saveRecordsToFile(my_records, PAYLOAD_MAX);
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

        TIMERSTART(saveRecordsToFile2);
        // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
        // Generazione record casuali
        saveRecordsToFile2("records2.bin", array_size, PAYLOAD_MAX);
        TIMERSTOP(saveRecordsToFile2);
    }
    else {
        TIMERSTART(saveRecordsToFile_SubProcess);
        // Altri processi non principale
        // Devono creare i record e mandarli al rank 0
        int n_records;
        // Ricevo dal rank 0 quanti record devo generare
        MPI_Recv(&n_records, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto local_buffer = saveRecordsToFile(n_records, PAYLOAD_MAX);

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
    if(rank == 0) {
    // Caricamento dei record dal file
    auto records = loadRecordsFromFile("records.bin");
    //printRecords(records);
    // Copia dei record per il confronto
    std::vector<Record*> recordsCopySeq = records;
    std::vector<Record*> recordsCopyPar = records;
    std::vector<Record*> recordsMPI_OpenMP = records;
    // Invio i
    // Buffer temporaneo per il merge
    vector<Record*> tempSeq(records.size());
    vector<Record*> tempPar(records.size());
    // MERGE SORT SEQUENZIALE
    TIMERSTART(mergeSortSeq);
    mergeSortSeq(recordsCopySeq, 0, recordsCopySeq.size() - 1, tempSeq);
    TIMERSTOP(mergeSortSeq);

    TIMERSTART(mergeSortPar);
    // Utilizzo questa direttiva per creare il pool di thread che verra utilizzato dal mergeSort
    // Riutilizzo i thread così non ho overhead per crearli e distruggerli
    #pragma omp parallel
    {
        // Questo è necessario per eseguire la funzione una sola volta
        #pragma omp single
        mergeSortPar(recordsCopyPar, 0, recordsCopyPar.size() - 1, tempPar);
    }
    TIMERSTOP(mergeSortPar);

    // Controllo ordinamento
    for (size_t i = 1; i < recordsCopySeq.size(); ++i)
        if (recordsCopySeq[i-1]->key > recordsCopySeq[i]->key) {
            cerr << "Errore ordinamento Sequenziale!" << endl;
        }

    for (size_t i = 1; i < recordsCopyPar.size(); ++i)
        if (recordsCopyPar[i-1]->key > recordsCopyPar[i]->key) {
            cerr << "Errore ordinamento Parallelo!" << endl;
    }

    cout << "======== MergeSort completato correttamente su " << array_size << " record. ==========" << endl;
    //printRecords(records);
}

    // =============================================
    // ============= MERGE MPI TREE ================
    // =============================================

    // Merge MPI Tree
    // Chiamato da ogni processo così effettuo il merge due alla volta così evito di fare tutto il merge in un solo passaggio e con solo il processo rank 0
    auto mergedRecords = mergeMPI_Tree(recordsMPI_OpenMP, rank, size, MPI_COMM_WORLD, num_threads);

    if (rank == 0) {
        // Solo il processo 0 stampa i risultati
        printRecords(mergedRecords);
    }

    MPI_Finalize();
    // Pulizia memoria
    for (Record* r : records) free(r);
    recordsCopySeq.clear();
    recordsCopyPar.clear();
    recordsMPI_OpenMP.clear();
    return 0;
}
