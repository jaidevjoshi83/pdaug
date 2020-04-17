from modlamp.core import BaseDescriptor
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-I", "--InFile", required=True, default=None, help=".fasta or .tsv")
parser.add_argument("-O", "--OutFile", required=True, default=None, help=".fasta or .tsv")
parser.add_argument("-M", "--Method", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-W","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

args = parser.parse_args()


if args.InFile.split('.')[1] == 'fasta':
    OutFile = "OutFile.tsv"
else:
    OutFile = "OutFile.fasta" 

if args.Method == 'ReadFasta':

    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))

    df = pd.DataFrame(Peptides, index=Index, columns=['Peptides'])
    df.to_csv(os.path.join(args.Workdirpath, args.OutFile), sep="\t")

elif args.Method == "SaveFasta":
    df1 =  pd.read_csv(args.InFile, sep="\t")
    l = df1[df1.columns.tolist()[0]].tolist()
    b = BaseDescriptor(l)
    b.save_fasta(os.path.join(args.Workdirpath, args.OutFile), names=False)
else:
    pass



