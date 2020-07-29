import numpy as np
seed = 42
np.random.seed(seed)
import os
import pandas as pd
from Bio import SeqIO
from nltk import bigrams
from nltk import trigrams
import gensim, logging
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-M", "--ModelInput", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-R", "--row", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-C", "--column", required=False, default=200, help="Path to target tsv file")
parser.add_argument("-I", "--InputFasta", required=True, default=6, help="Path to target tsv file")
parser.add_argument("-O", "--OutFile", required=False, default='Out.tsv', help="Path to target tsv file")
parser.add_argument("-P", "--positive", required=True, default='Out.tsv', help="Path to target tsv file")
parser.add_argument("-N", "--negative", required=True, default='Out.tsv', help="Path to target tsv file")

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

new_model = gensim.models.Word2Vec.load(args.ModelInput)

import time
t0 = time.time()

temp_word = np.zeros(shape=(int(args.row), int(args.column)))

for index, record in enumerate(SeqIO.parse(args.InputFasta, 'fasta')):
    sum_of_sequence = 0
    tri_tokens = trigrams(record.seq)
    for item in ((tri_tokens)):
        tri_str = item[0] + item[1] + item[2]
        if tri_str not in list(new_model.wv.vocab):
            continue
        sum_of_sequence = sum_of_sequence + new_model[tri_str]
    temp_word[index] = sum_of_sequence

t1 = time.time()

temp_scaler = preprocessing.StandardScaler().fit(temp_word)
temp_word_scaled = temp_scaler.transform(temp_word)

clm = [x for x in range(0,temp_word_scaled.shape[1])]

y_temp_word = np.vstack((np.ones((args.positive, 1)), 
                    np.zeros((args.negative,1))))

c, r = y_temp_word.shape
y_temp_word = y_temp_word.reshape(c,)

df = pd.DataFrame(temp_word_scaled, columns=clm)
df.to_csv(args.OutFile, index=None, sep="\t")