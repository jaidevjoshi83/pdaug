import nltk
from nltk import trigrams
import pandas as pd
from Bio import SeqIO
import gensim, logging
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-I", "--Input", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-M", "--min_count", required=False, default=0, help="Path to target tsv file")
parser.add_argument("-S", "--size", required=False, default=200, help="Path to target tsv file")
parser.add_argument("-W", "--window", required=False, default=5, help="Path to target tsv file")
parser.add_argument("-g", "--sg", required=False, default=1, help="Path to target tsv file")
parser.add_argument("-O", "--OutFile", required=False, default='word2vec_model', help="Path to target tsv file")

args = parser.parse_args()

class ProteinSeq(object):
    def __init__(self):
        pass
    def __iter__(self):
        for index, record in enumerate(SeqIO.parse(args.Input, 'fasta')):
            for loop_num in range(0, 3):
                Ngram_list = []
                tri_tokens = trigrams(record.seq)
                for index1, item in enumerate(tri_tokens):
                    if index1 % 3 == loop_num:
                        tri_pep = item[0] + item[1] + item[2]
                        Ngram_list.append(tri_pep)
                yield Ngram_list
#min_count = 0
#size = 200
#window = 5
#sg = 1
sentences = ProteinSeq() 
model = gensim.models.Word2Vec(sentences, min_count=int(args.min_count), size=int(args.size), window=int(args.window), sg = int(args.sg), workers = 10)
model.save(args.OutFile)







