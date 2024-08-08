import pandas as pd
import glob
import os
import sys
import torch
import numpy as np
from functools import reduce
import csv
from sentence_transformers import SentenceTransformer
sen_w_feats = []
sentence_embeddings = []


# Load the BERT tokenizer.
print('Loading Transformer...')
model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

list_of_files = glob.glob('/home/jay_kejriwal/Fisher/Processed/VACC_text/*.txt',recursive=True) 
output_path = '/home/jay_kejriwal/Fisher/Processed/Embeddings/VACC_text_BERT'

for file_name in list_of_files:
    out_name= os.path.join(output_path, os.path.basename(file_name))
    csv_input = pd.read_csv(file_name, usecols=[3], names=['utterance'],delimiter='\t',header=None)
    for index, row in csv_input.iterrows():
        sen_w_feats.append(row["utterance"])
        
    #Convert sentence to list
    sentence_embeddings = model.encode(sen_w_feats)
    sentence_vectors1=sentence_embeddings.tolist()

    #Merge consecutive utterance of Speaker A and B
    out = reduce(lambda x, y: x+y, sentence_vectors1)

    #Each consecutive utterance is of size 1536 i.e 768 for each utterance
    chunks = [out[x:x+1536] for x in range(0, len(out)-768, 768)]

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