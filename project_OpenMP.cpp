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


void saveRecordsToFile(const std::string& filename, int n, unsigned int payload_max) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Errore apertura file " << filename << "\n";
        return;
    }

    #pragma omp parallel
    {
        std::vector<char> local_buffer;
        // Riserva spazio per il buffer locale così garantisco il gisto spazio
        local_buffer.reserve((sizeof(unsigned long) + sizeof(unsigned int) + payload_max) * (n / omp_get_thread_num() + 1));

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

        #pragma omp critical
        {
            out.write(local_buffer.data(), local_buffer.size());
        }   
    }
    out.close();
    std::cout << "File " << filename << " creato con " << n << " record in parallelo.\n";
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
    if (right - left > 400) { // solo grandi segmenti diventano task
        #pragma omp task shared(arr, temp)
        mergeSortPar(arr, left, mid, temp);
        #pragma omp task shared(arr, temp)
        mergeSortPar(arr, mid + 1, right, temp);
        #pragma omp taskwait
    } else {
        mergeSortPar(arr, left, mid, temp);
        mergeSortPar(arr, mid + 1, right, temp);
    }

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


int main(int argc, char *argv[]) {
    srand(time(NULL));

    unsigned int PAYLOAD_MAX = 1024; // default

    unsigned long array_size = 1000; 
    unsigned int num_threads = 0;

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
    TIMERSTART(saveRecordsToFile);
    // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
    // Generazione record casuali
    saveRecordsToFile("records.bin", array_size, PAYLOAD_MAX);
    TIMERSTOP(saveRecordsToFile);

    // Caricamento dei record dal file
    auto records = loadRecordsFromFile("records.bin");
    //printRecords(records);
    // Copia dei record per il confronto
    std::vector<Record*> recordsCopySeq = records;
    std::vector<Record*> recordsCopyPar = records;
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
    if (num_threads > 0) {
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp single
            mergeSortPar(recordsCopyPar, 0, recordsCopyPar.size() - 1, tempPar);
        }
        } else {
            #pragma omp parallel
            {
                #pragma omp single
                mergeSortPar(recordsCopyPar, 0, recordsCopyPar.size() - 1, tempPar);
            }
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

    cout << "========MergeSort completato correttamente su " << array_size << " record.==========" << endl;
    //printRecords(records);
    

    // Pulizia memoria
    for (Record* r : records) free(r);

    return 0;
}
