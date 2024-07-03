import os
import fnmatch
import sys
import subprocess
import glob

list_of_files = glob.glob('/home/jay_kejriwal/Fisher/fisher03_audio/*.sph',recursive=True)
new_path = '/home/jay_kejriwal/Fisher/Processed/Audio'

def convert_sphfiles(filename):
    output=os.path.basename(filename)
    print("processing file", output)
    outfile1 = os.path.join(new_path, output).split('.')[-2]+'.wav'
    cmd2wav1 = 'sph2pipe -f rif ' + filename +' '+ outfile1
    print(cmd2wav1)
    subprocess.call(cmd2wav1, shell=True)
    
for file in list_of_files:
    convert_sphfiles(file)
    print(file)

