import numpy as np
import os
import pandas as pd
from Bio import SeqIO
from nltk import bigrams
from nltk import trigrams
import gensim
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-M", "--ModelInput", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-R", "--row", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-C", "--column", required=True, default=200, help="Path to target tsv file")
parser.add_argument("-I", "--InputFasta", required=True, default=6, help="Path to target tsv file")
parser.add_argument("-O", "--OutFile", required=False, default='Out.tsv', help="Path to target tsv file")
parser.add_argument("-P", "--positive", required=True, help="Path to target tsv file")
parser.add_argument("-N", "--negative", required=True, help="Path to target tsv file")

args = parser.parse_args()

seed = 42
np.random.seed(seed)


new_model = gensim.models.Word2Vec.load(args.ModelInput)

import time
t0 = time.time()

temp_word = np.zeros(shape=(int(args.row), int(args.column)))

for index, seqs in enumerate(SeqIO.parse(args.InputFasta, 'fasta')):
    seq_sum = 0
    tri_seq = trigrams(seqs.seq)
    for item in ((tri_seq)):
        tri_str = item[0] + item[1] + item[2]
        if tri_str not in list(new_model.wv.vocab):
            continue
        seq_sum = seq_sum + new_model[tri_str]
    temp_word[index] = seq_sum

t1 = time.time()

#temp_scaler = preprocessing.StandardScaler().fit(temp_word)
#temp_word = temp_scaler.transform(temp_word)

temp_word = temp_word


clm = [x for x in range(0,temp_word.shape[1])]
y_temp_word = np.vstack((np.ones((int(args.positive), 1)), np.zeros((int(args.negative),1))))

c, r = y_temp_word.shape
y_temp_word = y_temp_word.reshape(c,)

class_label = pd.DataFrame(y_temp_word, columns=["Class_label"])

df = pd.DataFrame(temp_word, columns=clm)
df = pd.concat([df,class_label], axis=1)

df.to_csv(args.OutFile, index=None, sep="\t")