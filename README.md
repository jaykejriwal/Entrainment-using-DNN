# Training models for detecting entrainment using DNN

Python program for DNN models for detecting entrainment at auditory and semantic linguistic levels.

## Dataset

We utilized state-of-the-art DNN embeddings such as BERT and TRIpLet Loss network (TRILL) vectors to extract features for measuring semantic and auditory similarities of turns within dialogues in three spoken corpora, namely Columbia Games corpus, Voice Assistant conversation corpus, and Fisher corpus.


## Required Software

ffmpeg (Download from https://www.ffmpeg.org/download.html)

sph2pipe (Download from https://www.openslr.org/3/)

opensmile (https://github.com/audeering/opensmile)

sentence-transformers (pip install sentence-transformers)

tensorflow (pip install tensorflow)

textgrid (Install textgrid from https://github.com/kylebgorman/textgrid)

TRILL vectors model (Download from https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3)

## Execution instruction

Firstly, train models with Fisher corpus. The programs need to be executed in a sequential format. 
Firstly, LLD features can be extracted using shell script file 0feat_extract_nopre.sh
Next, the 1create_h5data.py file allows the creation of embeddings in h5 data format.
Lastly, models can be trained using different distance measures, such as L1 and cos, which are mentioned in the file.

For CGC and VAC corpus, two Jupyter Notebook files are provided. These files need to be executed first for feature extraction.

## Citation

Kejriwal, J., Beňuš, Š., Rojas-Barahona, L.M. (2023) Unsupervised Auditory and Semantic Entrainment Models with Deep Neural Networks. Proc. INTERSPEECH 2023, 2628-2632, doi: 10.21437/Interspeech.2023-1929
