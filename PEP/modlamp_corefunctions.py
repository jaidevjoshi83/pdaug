from modlamp.core import *
import pandas as pd
import os, sys
import argparse


parser = argparse.ArgumentParser(description='Deployment tool')
subparsers = parser.add_subparsers()

mutateAA = subparsers.add_parser('mutateAA')
mutateAA.add_argument("-I","--InFile", required=True, default=None, help="")
mutateAA.add_argument("-N","--nr", required=True, default=None, help="")
mutateAA.add_argument("-P","--Prob", required=True, default=None, help="")
mutateAA.add_argument("-F","--FastOut", required=False, default=None, help="")
mutateAA.add_argument("-T","--TsvOut", required=False, default="Out.tsv", help="")
mutateAA.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

filterduplicates = subparsers.add_parser('filterduplicates')
filterduplicates.add_argument("-I","--InFile", required=True, default=None, help="")
filterduplicates.add_argument("-F","--FastOut", required=False, default=None, help="")
filterduplicates.add_argument("-T","--TsvOut", required=False, default="Out.tsv", help="")
filterduplicates.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")


keepnaturalaa = subparsers.add_parser('keepnaturalaa')
keepnaturalaa.add_argument("-I","--InFile", required=True, default=None, help="")
keepnaturalaa.add_argument("-F","--FastOut", required=False, default=None, help="")
keepnaturalaa.add_argument("-T","--TsvOut", required=False, default="Out.tsv", help="")
keepnaturalaa.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")


filteraa = subparsers.add_parser('filteraa')
filteraa.add_argument("-I","--InFile", required=True, default=None, help="")
filteraa.add_argument("-F","--FastOut", required=False, default=None, help="")
filteraa.add_argument("-T","--TsvOut", required=False, default="Out.tsv", help="")
filteraa.add_argument("-A","--FilterAA", required=True, default=None, help="")
filteraa.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")


args = parser.parse_args()


if sys.argv[1] == 'mutateAA':

    df1 =  pd.read_csv(args.InFile, sep="\t")
    l = df1[df1.columns.tolist()[0]].tolist()
    b = BaseSequence(len(l))
    b.sequences = l
    b.names = ['sequences_'+str(i) for i in range(0,len(l))]
    df = pd.DataFrame(b.sequences, index=b.names, columns=['peptides'])
    b.mutate_AA(int(args.nr), float(args.Prob))

    df.to_csv(os.path.join(args.Workdirpath, args.TsvOut),  sep='\t', index=None)

    if args.FastOut is not None:
        b.save_fasta(os.path.join(args.Workdirpath, args.FastOut), names=True)


elif sys.argv[1] == 'filterduplicates':

    df1 =  pd.read_csv(args.InFile, sep="\t")
    l = df1[df1.columns.tolist()[0]].tolist()
    b = BaseSequence(len(l))
    b.sequences = l
    b.names = ['sequences_'+str(i) for i in range(0,len(l))]
    df = pd.DataFrame(b.sequences, index=b.names, columns=['peptides'])
    b.filter_duplicates()

    df.to_csv(os.path.join(args.Workdirpath, args.TsvOut),  sep='\t', index=None)

    if args.FastOut is not None:
        b.save_fasta(os.path.join(args.Workdirpath, args.FastOut), names=True)


elif sys.argv[1] == 'keepnaturalaa':

    df1 =  pd.read_csv(args.InFile, sep="\t")
    l = df1[df1.columns.tolist()[0]].tolist()
    b = BaseSequence(len(l))
    b.sequences = l
    b.names = ['sequences_'+str(i) for i in range(0,len(l))]
    df = pd.DataFrame(b.sequences, index=b.names, columns=['peptides'])
    b.keep_natural_aa()

    df.to_csv(os.path.join(args.Workdirpath, args.TsvOut),  sep='\t', index=None)

    if args.FastOut is not None:
        b.save_fasta(os.path.join(args.Workdirpath, args.FastOut), names=True)

elif sys.argv[1] == 'filteraa':


    df1 =  pd.read_csv(args.InFile, sep="\t")
    l = df1[df1.columns.tolist()[0]].tolist()
    b = BaseSequence(len(l))
    b.sequences = l
    b.names = ['sequences_'+str(i) for i in range(0,len(l))]
    df = pd.DataFrame(b.sequences, index=b.names, columns=['peptides'])


    FilterAA = args.FilterAA.split(',')

    b.filter_aa(FilterAA)
    
    df.to_csv(os.path.join(args.Workdirpath, args.TsvOut),  sep='\t', index=None)

    if args.FastOut is not None:
        b.save_fasta(os.path.join(args.Workdirpath, args.FastOut), names=True)



