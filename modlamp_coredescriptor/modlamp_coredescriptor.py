from modlamp.core import BaseDescriptor
from modlamp.descriptors import PeptideDescriptor
import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()

parser.add_argument("-I", "--InFile", required=True, default=None, help=".fasta or .tsv")
parser.add_argument("-O", "--OutFile", required=True, default=None, help=".fasta or .tsv")
#parser.add_argument("-M", "--Method", required=True, default=None, help="Path to target tsv file")
#parser.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")


args = parser.parse_args()

if args.InFile.split('.')[1] == 'fasta':

    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    Pep = Peptides

elif args.InFile.split('.tsv') :
    df1 =  pd.read_csv(args.InFile, sep="\t")
    l = df1[df1.columns.tolist()[0]].tolist()

    Pep = l

else:
    pass

df =    pd.DataFrame()

for i, l in enumerate(Pep):

    D = PeptideDescriptor([l])
    D.count_ngrams([2])

    df1 = pd.DataFrame(D.descriptor, index=["sequence"+str(i),])
    df = pd.concat([df, df1], axis=0)

df =  df.fillna(0)
df.to_csv(args.OutFile, sep='\t', index=None)







