import glob
import os
import pandas as pd

list_of_files = glob.glob('/home/jay_kejriwal/Fisher/fe_03_p1_tran/data/trans/**/*.txt',recursive=True)
new_path = '/home/jay_kejriwal/Fisher/Processed/Text'

def read_lexicon(filename):
    output=os.path.basename(filename)
    print("processing file", output)
    outfiletxt = os.path.join(new_path, output)
    trans = open(filename).readlines()
    spk_list=[]
    for line in trans:
        if line!='\n':
            if line[0] !='#':
                start, stop, spk, utt = line.rstrip('\n').replace(':','').split(' ',3)
                spk_list.append([start, stop, spk, utt])
    df = pd.DataFrame (spk_list, columns = ['start','end','speaker','utterance'])
    df['sentences']=(df['speaker'].ne(df['speaker'].shift())).cumsum()
    df3=df.groupby('sentences').agg(lambda x: ' '.join(x))
    df3['start'] = df3['start'].map(lambda x: x.split(" ")[0])
    df3['end'] = df3['end'].map(lambda x: x.split(" ")[-1])
    df3['speaker'] = df3['speaker'].map(lambda x: x.split(" ")[-1])
    if df3.iloc[-1]['speaker'] == df3.iloc[0]['speaker']:
        df4=df3.drop(df3.index[len(df3)-1])
        df4.to_csv(outfiletxt, index=False,header=False,sep='\t')
        del df4
    else:
        df3.to_csv(outfiletxt, index=False,header=False,sep='\t')
    del output,outfiletxt,filename,line,spk_list,df,df3    


for file in list_of_files:
    read_lexicon(file)
    print(file)
    
