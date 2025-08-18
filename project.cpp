#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <vector>
#include <getopt.h>
#include <future>
#include <iostream>
#include <fstream>

#include "hpc_helpers.hpp"   

// Direttive OpenOMP -> usare (-fopenmp) per compilare


using namespace std;

struct Record {
    unsigned long key; // 8 byte
    uint32_t len;      // lunghezza payload -> quanti byte sono usati per il payload
    char payload[];    // flessibile
};

    // Alloco e Creo il singolo record in memoria -> per ogni record devo usarla
    Record* createRandomRecord(unsigned int payload_size) {
        // Alloco memoria per header + payload
        Record* rec = (Record*) malloc(sizeof(Record) + payload_size);
        rec->key = (rand() % 100) + 1;
        rec->len = payload_size;

        for (unsigned int i = 0; i < payload_size; i++) {
            rec->payload[i] = (char)(rand() % 256); // blob di byte
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
        unsigned int size = 8 + (rand() % (payload_max - 8 + 1));
        Record* rec = createRandomRecord(size);

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

    out.close();
    std::cout << "File " << filename << " creato con " << n << " record.\n";
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

    // CREARE VARIABILE DA AFFIDARE A PAYLOAD_MAX da linea di comando
    // Generazione record casuali
    TIMERSTART(saveRecordsToFile);
    saveRecordsToFile("records.bin", array_size, PAYLOAD_MAX);
    TIMERSTOP(saveRecordsToFile);
    // Caricamento dei record dal file
    auto records = loadRecordsFromFile("records.bin");
    //printRecords(records);

    // Buffer temporaneo per il merge
    vector<Record*> temp(records.size());
    // MERGE SORT SEQUENZIALE
    mergeSortSeq(records, 0, records.size() - 1, temp);

    // Controllo ordinamento
    for (size_t i = 1; i < records.size(); ++i)
        if (records[i-1]->key > records[i]->key) {
            cerr << "Errore ordinamento!" << endl;
        }

    cout << "========MergeSort completato correttamente su " << array_size << " record.==========" << endl;
    //printRecords(records);
    

    // Pulizia memoria
    for (Record* r : records) free(r);

    return 0;
}
