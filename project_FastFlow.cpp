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
#include <fstream>
#include "fastflow/ff/ff.hpp"
#include "fastflow/ff/dc.hpp"

#include <omp.h>

using namespace ff;
using namespace std;

struct Record {
    unsigned long key; // 8 byte
    uint32_t len;      // lunghezza payload -> quanti byte sono usati per il payload
    char payload[];    // flessibile
};


// Alloco e Creo il singolo record in memoria -> per ogni record devo usarla
Record* createRandomRecord(unsigned int payload_min, unsigned int payload_max) {
    // Ogni thread ha il suo generatore indipendente
    static thread_local std::mt19937_64 rng(std::random_device{}());
    
    std::uniform_int_distribution<unsigned long> key_dist(1, 100);        // chiavi 1-100
    std::uniform_int_distribution<int> byte_dist(0, 255);                 // byte payload 0-255
    std::uniform_int_distribution<unsigned int> size_dist(payload_min, payload_max); // lunghezza payload

    unsigned int payload_size = size_dist(rng);

    // Alloco record + payload
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

        // Leggo payload -> Posso evitare così risparmio memoria dato che ordino per la chiave
        in.read(reinterpret_cast<char*>(rec->payload), len); // Leggo comunque così posso stamparlo

        records.push_back(rec);
    }

    in.close();
    return records;
}

    // --- Merge per array di Record* ---
void mergeSeq(vector<Record*>& arr, int left, int mid, int right, vector<Record*>& temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i]->key <= arr[j]->key) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int idx = left; idx <= right; ++idx)
        arr[idx] = temp[idx];
}

// --- MergeSort sequenziale ---
void mergeSortSeq(vector<Record*>& arr, int left, int right, vector<Record*>& temp) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    mergeSortSeq(arr, left, mid, temp);
    mergeSortSeq(arr, mid + 1, right, temp);
    mergeSeq(arr, left, mid, right, temp);
}

void standardSeqSort(vector<Record*>& arr) {
    std::sort(arr.begin(), arr.end(), [](Record* a, Record* b) {
        return a->key < b->key;
    });
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
    std::ofstream& out;   // riferimento al file aperto

    // costruttore prende il file già aperto
    CreationCollector(std::ofstream& f) : out(f) {}

    void* svc(void* task) override {
        auto* buffer = static_cast<std::vector<char>*>(task);
        // std::cout << "Ricevuti " << buffer->size() << " byte dal worker\n";

        out.write(buffer->data(), buffer->size());

        delete buffer;  // libero il vettore
        return GO_ON;
    }

};


// --- Emitter: divide l'array in blocchi di chunk_size ---
struct EmitterSortParallel : ff_node {
    std::vector<Record*>& arr;
    int chunk_size;

    EmitterSortParallel(std::vector<Record*>& arr, int chunk_size)
        : arr(arr), chunk_size(chunk_size) {}

    void* svc(void*) override {
        for (size_t i = 0; i < arr.size(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, arr.size());
            auto* block = new std::vector<Record*>(arr.begin() + i, arr.begin() + end);
            ff_send_out(block);
        }
        return EOS;
    }
};

// --- Worker: ordina un blocco ---
struct WorkerSortParallel : ff_node {
    void* svc(void* task) override {
        std::vector<Record*>* block = (std::vector<Record*>*) task;

        std::sort(block->begin(), block->end(), [](Record* a, Record* b) {
            return a->key < b->key;
        });

        return block; // restituisce il blocco ordinato
    }
};

// --- Collector: merge incrementale due-a-due ---
struct CollectorSortParallel : ff_node {
    std::vector<Record*> result; // array ordinato accumulato

    void* svc(void* task) override {
        std::vector<Record*>* block = (std::vector<Record*>*) task;

        if (result.empty()) {
            // primo blocco: copia diretta
            result = *block;
        } else {
            // merge con il risultato parziale
            std::vector<Record*> merged;
            merged.reserve(result.size() + block->size());

            size_t i = 0, j = 0;
            while (i < result.size() && j < block->size()) {
                if (result[i]->key <= (*block)[j]->key)
                    merged.push_back(result[i++]);
                else
                    merged.push_back((*block)[j++]);
            }
            // copia rimanenti
            while (i < result.size()) merged.push_back(result[i++]);
            while (j < block->size()) merged.push_back((*block)[j++]);

            result.swap(merged);
        }

        delete block; // libera memoria del blocco ricevuto
        return GO_ON; // continua a ricevere altri blocchi
    }

    void svc_end() override {
        std::cout << "Ordinamento completato, array finale ha "
                  << result.size() << " record\n";
    }
};

int main(int argc, char *argv[]) {
    srand(time(NULL));

    unsigned int PAYLOAD_MAX = 1000; // default

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
    
    auto filename = "records.bin";
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Errore apertura file " << filename << "\n";
        return 0;
    }
    

    std::vector<ff_node*> workerVectorCreation;
    int chunk_size = array_size / num_threads;
    int resto  = array_size % num_threads;  // Non divisibile

    ff_farm farmCreation;

    for (int i = 0; i < num_threads; ++i) {
        // Per gestire casi in cui array size non è divisibile
        int local_chunk = chunk_size;
        if (i == num_threads - 1) {
            local_chunk += resto;   // l'ultimo worker prende in più
        }
        workerVectorCreation.push_back(new workersFileCreation(local_chunk, PAYLOAD_MAX));
    }
    

    CreationCollector* collectorCreation = new CreationCollector(out);
    farmCreation.add_workers(workerVectorCreation);    
    farmCreation.add_collector(collectorCreation);
    printf("-----------------\n");
    TIMERSTART(farmCreationParallel);
    if (farmCreation.run_and_wait_end() < 0) {
        std::cerr << "Errore avvio farm\n";
        return -1;
    }
    TIMERSTOP(farmCreationParallel);

    // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
    // Generazione record casuali
    TIMERSTART(saveRecordsToFile);
    // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
    // Generazione record casuali
    saveRecordsToFile("records2.bin", array_size, PAYLOAD_MAX);
    TIMERSTOP(saveRecordsToFile);
    
    out.close();
    // Caricamento dei record dal file
    auto records = loadRecordsFromFile("records.bin");


    //printRecords(records);

    // Buffer temporaneo per il merge
    vector<Record*> temp(records.size());
    // MERGE SORT SEQUENZIALE
    // Copie per confronto
    std::vector<Record*> recordsCopySeqMerge = records;
    std::vector<Record*> recordsCopySeqStandard = records;
    std::vector<Record*> recordsCopyPar = records;

    TIMERSTART(mergeSortSeq)
    mergeSortSeq(recordsCopySeqMerge, 0, recordsCopySeqMerge.size() - 1, temp);
    TIMERSTOP(mergeSortSeq)

    TIMERSTART(standardSeqSort)
    standardSeqSort(recordsCopySeqStandard);
    TIMERSTOP(standardSeqSort);

    // PARALLELO
    ff_farm mergeSortParalleloFarm;
    EmitterSortParallel* EmitterSort = new EmitterSortParallel(recordsCopyPar, chunk_size);

    std::vector<ff_node*> workerSort;
    for (int i = 0; i < num_threads; ++i) {
        workerSort.push_back(new WorkerSortParallel());
    }

    CollectorSortParallel* CollectorSort = new CollectorSortParallel();
    mergeSortParalleloFarm.add_emitter(EmitterSort);
    mergeSortParalleloFarm.add_workers(workerSort);
    mergeSortParalleloFarm.add_collector(CollectorSort);

    TIMERSTART(mergeParSort)
    if (mergeSortParalleloFarm.run_and_wait_end() < 0) {
        std::cerr << "Errore avvio farm\n";
        return -1;
    }
    TIMERSTOP(mergeParSort)

    // Controllo ordinamento
    for (size_t i = 1; i < recordsCopySeqMerge.size(); ++i)
        if (recordsCopySeqMerge[i-1]->key > recordsCopySeqMerge[i]->key) {
            cerr << "Errore ordinamento Sequenziale Merge!" << endl;
        }

    for (size_t i = 1; i < recordsCopySeqStandard.size(); ++i)
        if (recordsCopySeqStandard[i-1]->key > recordsCopySeqStandard[i]->key) {
            cerr << "Errore ordinamento Sequenziale Standard!" << endl;
        }
    
    for (size_t i = 1; i < recordsCopySeqStandard.size(); ++i)
    if (recordsCopySeqStandard[i-1]->key > recordsCopySeqStandard[i]->key) {
        cerr << "Errore ordinamento Sequenziale Standard!" << endl;
    }

    // cout << "========MergeSort completato correttamente su " << array_size << " record.==========" << endl;
    // //printRecords(records);
    
    // Pulizia memoria
    //for (Record* r : records) free(r);

    return 0;
}
