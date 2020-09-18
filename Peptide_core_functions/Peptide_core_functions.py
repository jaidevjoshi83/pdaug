from modlamp.core import BaseSequence
import pandas as pd
import os, sys
import argparse


parser = argparse.ArgumentParser(description='Deployment tool')
subparsers = parser.add_subparsers()

mutateAA = subparsers.add_parser('mutateAA')
mutateAA.add_argument("-I","--InFile", required=True, default=None, help="Input fasta sequence")
mutateAA.add_argument("-N","--nr", required=True, default=None, help="Number of mutations to perform per sequence")
mutateAA.add_argument("-P","--Prob", required=True, default=None, help="Probability of mutating a sequence")
mutateAA.add_argument("-F","--FastOut", required=False, default='Out.fasta', help="Mutated output fasta")

filterduplicates = subparsers.add_parser('filterduplicates')
filterduplicates.add_argument("-I","--InFile", required=True, default=None, help="Input file")
filterduplicates.add_argument("-F","--FastOut", required=False, default='Out.fasta', help="Output file")


keepnaturalaa = subparsers.add_parser('keepnaturalaa')
keepnaturalaa.add_argument("-I","--InFile", required=True, default=None, help="Inputt file")
keepnaturalaa.add_argument("-F","--FastOut", required=False, default='Out.fasta', help="Output file")


filteraa = subparsers.add_parser('filteraa')
filteraa.add_argument("-I","--InFile", required=True, default=None, help="Input file")
filteraa.add_argument("-F","--FastOut", required=False, default='Out.fasta', help="Output file")
filteraa.add_argument("-A","--FilterAA", required=True, default=None, help="Filter amino acide")

args = parser.parse_args()

if sys.argv[1] == 'mutateAA':

    Pep = []
    Index = []

    f = open(args.InFile)
    lines = f.readlines()

    for line in lines:
        if '>' in line:
            line = line.strip('\n')
            line = line.strip('\r')
            Index.append(line)
        else:
            line = line.strip('\n')
            line = line.strip('\r')
            Pep.append(line)

    b = BaseSequence(len(Pep))
    b.sequences = Pep
    b.mutate_AA(int(args.nr), float(args.Prob))
    OutPep = b.sequences

    OutFasta = open(args.FastOut, 'w')

    for i,O in enumerate(OutPep):

        OutFasta.write(Index[i]+'\n')
        OutFasta.write(O+'\n')


elif sys.argv[1] == 'filterduplicates':

    Pep = []
    Index = []

    f = open(args.InFile)
    lines = f.readlines()

    for line in lines:
        if '>' in line:
            line = line.strip('\n')
            line = line.strip('\r')
            Index.append(line)
        else:
            line = line.strip('\n')
            line = line.strip('\r')
            Pep.append(line)

    b = BaseSequence(len(Pep))

    b.sequences = Pep

    b.filter_duplicates()

    OutPep = b.sequences

    OutFasta = open(args.FastOut, 'w')

    for i,O in enumerate(OutPep):

        OutFasta.write(Index[i]+'\n')
        OutFasta.write(O+'\n')


elif sys.argv[1] == 'keepnaturalaa':

    Pep = []
    Index = []

    f = open(args.InFile)
    lines = f.readlines()

    for line in lines:
        if '>' in line:
            line = line.strip('\n')
            line = line.strip('\r')
            Index.append(line)
        else:
            line = line.strip('\n')
            line = line.strip('\r')
            Pep.append(line)

    b = BaseSequence(len(Pep))
    b.sequences = Pep
    b.keep_natural_aa()

    OutFasta = open(args.FastOut, 'w')

    OutPep = b.sequences

    for i,O in enumerate(OutPep):

        OutFasta.write(Index[i]+'\n')
        OutFasta.write(O+'\n')


elif sys.argv[1] == 'filteraa':

    Pep = []
    Index = []

    f = open(args.InFile)
    lines = f.readlines()

    for line in lines:
        if '>' in line:
            line = line.strip('\n')
            line = line.strip('\r')
            Index.append(line)
        else:
            line = line.strip('\n')
            line = line.strip('\r')
            Pep.append(line)


    b = BaseSequence(len(Pep))
    b.sequences = Pep

    FilterAA = args.FilterAA.split(',')

    b.filter_aa(FilterAA)

    OutPep = b.sequences

    OutFasta = open(args.FastOut, 'w')

    for i,O in enumerate(OutPep):

        OutFasta.write(Index[i]+'\n')
        OutFasta.write(O+'\n')
    




