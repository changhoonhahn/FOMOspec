#!/bin/bash 
now=$(date +"%T") 
echo "start time ... $now"

for galid in 1; do #101314; do # 253044 10924; do 
    python /Users/chang/projects/FOMOspec/run/firefly.py $galid 'desi' 'hpf_only'
done

now=$(date +"%T") 
echo "end time ... $now"
