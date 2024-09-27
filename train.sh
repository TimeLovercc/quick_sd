export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# each 1 18g 4 20g 32 70g
python running.py -p 5 6 -b 16 16
