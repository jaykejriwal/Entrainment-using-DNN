#!/usr/bin/env bash
cmddir=/home/jay_kejriwal/Fisher/Programs/LLD
featdir=/home/jay_kejriwal/Fisher/Programs/LLD/raw_features
raw_featdir=/home/jay_kejriwal/Fisher/Programs/LLD/raw_features
audiodirroot=/home/jay_kejriwal/Fisher/fisher03_audio

numParallelJobs=28
ctr=1

for f in $audiodirroot/*.sph;
do
	echo $f;
	(
	 python $cmddir/0feat_extract_nopre.py --audio_file $f 
	 ) &
if [ $(($ctr % $numParallelJobs)) -eq 0 ]
	then
#		
	wait
fi
ctr=`expr $ctr + 1`	
done