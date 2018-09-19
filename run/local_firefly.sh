#!/bin/bash 
now=$(date +"%T") 
echo "start time ... $now"

for galid in 1 101314 253044 10924; do 
    python /Users/chang/projects/FOMOspec/run/firefly.py $galid 'desi' 'hpf_only'
    python /Users/chang/projects/FOMOspec/run/firefly.py $galid 'desi' 'on'
done

now=$(date +"%T") 
echo "end time ... $now"
