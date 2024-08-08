import pandas as pd
import glob
import os
import sys
import torch
import numpy as np
from functools import reduce
import csv
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
sen_w_feats = []
sentence_embeddings = []


# Load the Google model.
print('Loading Googles model...')
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input).numpy()


list_of_files = glob.glob('/home/jay_kejriwal/Fisher/Processed/VACC_text/*.txt',recursive=True) 
output_path = '/home/jay_kejriwal/Fisher/Processed/Embeddings/VACC_text_Google'

for file_name in list_of_files:
    out_name= os.path.join(output_path, os.path.basename(file_name))
    csv_input = pd.read_csv(file_name, usecols=[3], names=['utterance'],delimiter='\t',header=None)
    for index, row in csv_input.iterrows():
        sen_w_feats.append(row["utterance"])
        
    #Convert sentence to list
    sentence_embeddings = embed(sen_w_feats)
    sentence_vectors1=sentence_embeddings.tolist()

    #Merge consecutive utterance of Speaker A and B
    out = reduce(lambda x, y: x+y, sentence_vectors1)

    #Each consecutive utterance is of size 1536 i.e 768 for each utterance
    chunks = [out[x:x+1024] for x in range(0, len(out)-512, 512)]

    #Convert list to array
    arr = np.asarray(chunks)
    with open(out_name, 'w') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerows(arr)
    sen_w_feats = []
    sentence_embeddings = []
    sentence_vectors1=None
    arr=None
    model_output=None