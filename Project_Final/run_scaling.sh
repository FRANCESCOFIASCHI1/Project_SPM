#!/bin/bash

# Array di parametri
ARRAY_SIZES=(100M)
PAYLOAD_SIZES=(128 256 512)     # ← corretto, niente virgole
THREADS=(2 4 8)                 # numero di thread OpenMP per MPI process
MPI_NODES=(2 4 8)               # numero di nodi MPI

# Ciclo su dimensione array
for SIZE in "${ARRAY_SIZES[@]}"; do
    # Ciclo su payload
    for PAY in "${PAYLOAD_SIZES[@]}"; do
        # Ciclo su thread
        for N in "${MPI_NODES[@]}"; do
            # Ciclo su nodi MPI
            for T in "${THREADS[@]}"; do
                echo ">>> SIZE=$SIZE, PAYLOAD=$PAY, THREADS=$T, NODES=$N"
                
                srun --mpi=pmix \
		     --ntasks=$N \
                     --cpus-per-task=$T \
                     --time=00:10:00 \
                     ./MPI_FF_MPI_IO -s $SIZE -p $PAY -t $T
                     
                echo
            done
        done
    done
done

