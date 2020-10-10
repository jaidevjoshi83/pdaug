from modlamp.core import BaseDescriptor
from modlamp.descriptors import PeptideDescriptor
import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()

parser.add_argument("-I", "--InFile", required=True, default=None, help="Input file")
parser.add_argument("-O", "--OutFile", required=True, default=None, help="Output file")
parser.add_argument("-N", "--Ngrams", required=True, default=None, help="ngrams")

args = parser.parse_args()

file = open(args.InFile)
lines = file.readlines()

Index = []
Pep = []

#print (lines)

for line in lines:
    if '>' in line:
        Index.append(line.strip('\n'))
    else:
        line = line.strip('\n')
        line = line.strip('\r')
        #print (line)
        Pep.append(line)

df =    pd.DataFrame()

for i, l in enumerate(Pep):

    D = PeptideDescriptor(l)
    D.count_ngrams([int(args.Ngrams)])

    df1 = pd.DataFrame(D.descriptor, index=["sequence"+str(i),])
    df = pd.concat([df, df1], axis=0)

df =  df.fillna(0)
df.to_csv(args.OutFile, sep='\t', index=None)



