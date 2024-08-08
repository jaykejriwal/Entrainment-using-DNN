import os
import fnmatch
import sys
import subprocess
import glob

list_of_wav_files = glob.glob('/home/jay_kejriwal/Fisher/Processed/Audio/*.wav',recursive=True)
new_path_resampled = '/home/jay_kejriwal/Fisher/Processed/Audio-resampled'

def resample_wavfiles(filename):
    output1=os.path.basename(filename)
    print("processing file", output1)
    outfile2 = os.path.join(new_path_resampled, output1)
    cmd2wav2 = 'ffmpeg  -i ' + filename + ' ' + "-ac 1" + ' ' + "-ar 16000" + ' ' + outfile2
    print(cmd2wav2)
    subprocess.call(cmd2wav2, shell=True)
    
for file in list_of_wav_files:
    resample_wavfiles(file)
    print(file)
