#!/bin/bash 
now=$(date +"%T") 
echo "start time ... $now"

for galid in 1; do # 101314 253044 10924
    ofile=/Users/chang/projects/FOMOspec/run/_prosp_desi_emcee_$galid.o
    python /Users/chang/projects/FOMOspec/run/prospector.py 'source' 'emcee' $galid 1 > $ofile
    #python /Users/chang/projects/FOMOspec/run/prospector.py 'desi' 'emcee' $galid 1 > $ofile
done

now=$(date +"%T") 
echo "end time ... $now"
