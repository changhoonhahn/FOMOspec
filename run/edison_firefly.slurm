#!/bin/bash -l 
#SBATCH -q debug 
#SBATCH -N 1 
#SBATCH -t 00:30:00 
#SBATCH -J firefly 
#SBATCH -o _firefly.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

#for galid in 101314 253044 10924; do 
for galid in 10924; do 
    srun -n 1 -c 1 python /global/homes/c/chahah/projects/FOMOspec/run/firefly.py $galid 'desi' 'on'
done

now=$(date +"%T") 
echo "end time ... $now"
