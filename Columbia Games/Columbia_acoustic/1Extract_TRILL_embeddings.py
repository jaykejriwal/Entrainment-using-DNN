import pandas as pd
import torch
from functools import reduce
import csv
import random
import subprocess
import re
import glob
import os,sys
import numpy as np
import soundfile as sf
import wave
import json
import tensorflow as tf1
import tensorflow_hub as hub
# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
#import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
modulev3=None
modulev3_graph=None

tf.enable_v2_behavior()
assert tf.executing_eagerly()
   
def get_TRILLv3_signal(signal,samplerate):
    global modulev3
    if modulev3==None:
        print('******************\nLoading model ...\n******************')    
        modulev3 = hub.load('/home/jay_kejriwal/Fisher/trill_extraction_v2/v3')
    
    
    max_int16 = 2**15
    chunks_cnt=int(signal.shape[0]/(samplerate*10.0))#10 seconds max in chunk
    if chunks_cnt==0:
        chunks=[signal]
    else:
        chunks=np.array_split(signal, chunks_cnt)
    
    trillv3_emb_all=np.empty(shape=(0,512))
    
    for chunk in chunks:
        trillv3 = modulev3(samples=chunk, sample_rate=samplerate)
        trillv3_emb = trillv3['embedding']
        trillv3_emb_all=np.concatenate((trillv3_emb_all, trillv3_emb))

    trillv3_emb_avg = np.mean(trillv3_emb_all, axis=0, keepdims=False)

    return (trillv3_emb_avg.tolist())    

def check_wav_format(wav_file, start, end):
    wf = wave.open(wav_file)
    nchannels, sampwidth, framerate, nframes, comptype, compname = wf.getparams()
    print(nchannels, sampwidth, framerate, nframes, comptype, compname)
    wav_length = float(nframes) / float(framerate)
    print(wav_length)						
    return(framerate)

def get_TRILLv3_audiofile_from_to(wav_file,start,end):
    print('get_TRILLv3_signal:',wav_file,start,end)
    samplerate=check_wav_format(wav_file, start, end)
    if samplerate<0:
        return(null)    
    startsample=int(start*samplerate)
    endsample=int(end*samplerate)
    signal, samplerate = sf.read(wav_file,start=startsample, stop=endsample)
    print(len(signal),samplerate)    

    trill=get_TRILLv3_signal(signal,samplerate)
    return(trill)

output_path = '/home/jay_kejriwal/Fisher/Processed/Embeddings/CGC_audio'
audio_path = '/home/jay_kejriwal/Fisher/Processed/CGC_audio'
text_path= '/home/jay_kejriwal/Fisher/Processed/CGC_text'
all_files = os.listdir(text_path)
for root, dirs, files in os.walk(text_path):
    for file in files:
        if file.endswith('.TURN_txt'):
            with open(os.path.join(root, file), 'r') as f:
                out_name= os.path.basename(file)
                output=os.path.join(output_path, out_name)
                audioembeddings=[]
                text = f.readlines()
                for line in text:
                    n1=line.split('\t')[0]
                    n2=line.split('\t')[1]
                    if(float(n1)-float(n1)<0.5):
                        n2=float(n2)+0.5
                    n3=os.path.join(audio_path, out_name).split('.TURN_txt')[0]+'.wav'
                    print(n3,n1,n2)
                    x=get_TRILLv3_audiofile_from_to(n3,float(n1),float(n2))
                    audioembeddings.append(x)

            #Merge consecutive utterance of Speaker A and B
            out = reduce(lambda x, y: x+y, audioembeddings)

            #Each consecutive utterance is of size 1024 i.e 512 for each utterance
            chunks = [out[x:x+1024] for x in range(0, len(out)-512, 512)]

            #Convert list to array
            arr = np.asarray(chunks)
            with open(output, 'w') as fcsv:
                writer = csv.writer(fcsv)
                writer.writerows(arr)
            audioembeddings = []
            audio_vectors = []
            arr=None
