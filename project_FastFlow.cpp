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
            // Per gestire anche il caso in cui l'array non sia divisibile per chunk_size
            // end ha dimensione il minimo tra la posizione nell'array + chunk_size e la dimensione massima rimasta nell'array nel caso non sia divisibile
            size_t end = std::min(i + chunk_size, arr.size());
            // In questa riga creo il vettore di dimensione chunk_size
            // Sommando i ottengo un iteratore all'elemento di indice i.
            // arr.begin() restituisce un iteratore al primo elemento sommandoci i mi sposto tra tutti gli elementi fino a arr.size
            // arr.begin() + end restituisce invece l'iteratore alla posizione successiva all'ultimo elemento del blocco
            // In questo modo costruisco un nuovo vettore copiando gli elementi da indice i fino a end-1.
            auto* block = new std::vector<Record*>(arr.begin() + i, arr.begin() + end);
            // Manda il blocco da ordinare ai worker
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
    std::vector<Record*>& result; // array ordinato accumulato

    CollectorSortParallel(std::vector<Record*>& result)
        : result(result) {}

    void* svc(void* task) override {
        std::vector<Record*>* block = (std::vector<Record*>*) task;

        if (result.empty()) {
            // primo blocco che arriva -> copio direttamente
            result = *block;
        } else {
            // Eseguo il merge mano a mano che arriano dati 
            std::vector<Record*> merged;
            merged.reserve(result.size() + block->size());

            size_t i = 0, j = 0;
            // Finche uno dei due array è esaurito
            while (i < result.size() && j < block->size()) {
                // Confronto le chiavi dei due array e inserisco l'elemento nel vettore del merge
                if (result[i]->key <= (*block)[j]->key)
                    merged.push_back(result[i++]);
                else
                    merged.push_back((*block)[j++]);
            }
            // copia rimanenti
            // Quando ho finito uno dei due array copio i rimanenti elementi dell'array non esaurito nel blocco finale
            // Se esaurisco prima l'array block allora il primo while sposta tutti gli elementi nell'array merged
            while (i < result.size()) merged.push_back(result[i++]); // Duplica il riferimento ai dati -> puntaotri
            while (j < block->size()) merged.push_back((*block)[j++]);

            // SCambia il contenuto dei due buffer
            // Result contiene merged e merged contiene result
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

struct CollectorMergeTree : ff_node {
    std::vector<Record*>& finalResult;      // risultato finale
    std::mutex mtx;
    std::queue<std::vector<Record*>*> q;    // coda dei blocchi ordinati

    CollectorMergeTree(std::vector<Record*>& result)
        : finalResult(result) {}

    // funzione di merge di due blocchi
    static std::vector<Record*>* mergeBlocks(std::vector<Record*>* a, std::vector<Record*>* b) {
        std::vector<Record*>* merged = new std::vector<Record*>();
        merged->reserve(a->size() + b->size());

        size_t i = 0, j = 0;
        while (i < a->size() && j < b->size()) {
            if ((*a)[i]->key <= (*b)[j]->key) merged->push_back((*a)[i++]);
            else merged->push_back((*b)[j++]);
        }
        while (i < a->size()) merged->push_back((*a)[i++]);
        while (j < b->size()) merged->push_back((*b)[j++]);

        delete a;
        delete b;
        return merged;
    }

    void* svc(void* task) override {
        auto* block = static_cast<std::vector<Record*>*>(task);

        std::vector<Record*>* toMerge = nullptr;

        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push(block);

            // se ci sono almeno due blocchi, li prendiamo e li fondiamo in parallelo
            if (q.size() >= 2) {
                auto* first = q.front(); q.pop();
                auto* second = q.front(); q.pop();
                toMerge = mergeBlocks(first, second);  // merge sequenziale qui, ma puoi anche lanciare thread separati
                q.push(toMerge);
            }
        }

        return GO_ON;
    }

    void svc_end() override {
        // Alla fine ci dovrebbe rimanere un solo blocco nella coda: il risultato finale
        if (!q.empty()) {
            finalResult = *(q.front());
            delete q.front();
            q.pop();
        }
        std::cout << "Ordinamento completato, array finale ha "
                << finalResult.size() << " record\n";
        // niente return
    }
};


int main(int argc, char *argv[]) {
    srand(time(NULL));

    unsigned int PAYLOAD_MAX = 1000; // default

    unsigned long array_size = 1000; 
    int num_threads = 16;

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
    // TIMERSTART(saveRecordsToFile);
    // // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
    // // Generazione record casuali
    // saveRecordsToFile("records2.bin", array_size, PAYLOAD_MAX);
    // TIMERSTOP(saveRecordsToFile);
    
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

    // TIMERSTART(mergeSortSeq)
    // mergeSortSeq(recordsCopySeqMerge, 0, recordsCopySeqMerge.size() - 1, temp);
    // TIMERSTOP(mergeSortSeq)

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

    std::vector<Record*> result;
    CollectorSortParallel* CollectorSort = new CollectorSortParallel(result);
    // CollectorMergeTree* CollectorSort = new CollectorMergeTree(result);
    mergeSortParalleloFarm.add_emitter(EmitterSort);
    mergeSortParalleloFarm.add_workers(workerSort);
    mergeSortParalleloFarm.add_collector(CollectorSort);

    TIMERSTART(mergeParSort)
    if (mergeSortParalleloFarm.run_and_wait_end() < 0) {
        std::cerr << "Errore avvio farm\n";
        return -1;
    }
    TIMERSTOP(mergeParSort)

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
