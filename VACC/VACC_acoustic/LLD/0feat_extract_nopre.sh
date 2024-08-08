#!/usr/bin/env bash
cmddir=/home/jay_kejriwal/Fisher/Programs/VACC/LLD
featdir=/home/jay_kejriwal/Fisher/Programs/VACC/LLD/raw_features
raw_featdir=/home/jay_kejriwal/Fisher/Programs/VACC/LLD/raw_features
audiodirroot=/home/jay_kejriwal/Fisher/Processed/VACC_audio_resampled

numParallelJobs=28
ctr=1

for f in $audiodirroot/*.wav;
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