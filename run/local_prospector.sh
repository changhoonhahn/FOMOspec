#!/bin/bash 
now=$(date +"%T") 
echo "start time ... $now"

for galid in 1; do # 101314 253044 10924
    python /Users/chang/projects/FOMOspec/run/prospector.py 'desi' 'dynesty' $galid 1
done

now=$(date +"%T") 
echo "end time ... $now"
