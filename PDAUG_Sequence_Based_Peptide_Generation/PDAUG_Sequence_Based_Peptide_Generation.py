import sys
import itertools
import pandas as pd
import random
import os
from itertools import permutations
from random import shuffle
import argparse, sys
import pandas as pd


def MutatedPeptides(input_file, index_list, AA, outputFile):


    index_list = [int(x) for x in index_list.split(',')]
    out_put = []
    AA = AA.split(',')
    l = len(index_list)

    replacements = [x for x in itertools.permutations(AA,l)]


    counter = 0
    to_modify = [x for x in input_file]

    for replacement in replacements:
        for i,index in enumerate(index_list):
            to_modify[index_list[i]-1] = replacement[i]

        counter = counter + 1
        out_put.append("".join(to_modify).upper())

    w = open(outputFile, 'w')

    for i, f in enumerate(out_put):


        w.write(">sequence_"+str(i)+'\n')
        w.write(f+'\n')

def RandomPeptides(AAs, pep_length, out_pep_num, outputFile):


    if int(pep_length) > 20:
        print ("Max peptide lenth 20")
        exit()
    else:
        pass

    if int(out_pep_num) > 10000:
        print ("Max peptide library 10000")
        exit()
    else:
        pass

    out_pep_lib = []
    raw = AAs.split(',')

    for x in range(int(out_pep_num)):
        un_seq = []
        for i in range(int(pep_length)):
            un_seq.append(random.choice(raw))
        out_pep_lib.append("".join(un_seq))


    w = open(outputFile, 'w')


    for i, f in enumerate(out_pep_lib):

        w.write(">sequence_"+str(i)+'\n')
        w.write(f+'\n')

def SlidingWindowPeptide(infile, window_size, frag_size, outputFile):


    if int(window_size) > 10:
        print ("Max window_size 10")
        exit()
    else:
        pass
    if int(frag_size) >  20:
        print ("Max frag size is 20")
        exit()
    else:
        pass


    pep_list = []

    f = open(infile)

    lines = f.readlines()

    flines = []

    for line in lines:
        if '>' in line:
            pass
        else:
            flines.append(line.strip('\n'))
    sequence = "".join(flines)

    for i in range(int(frag_size)):
        if int(frag_size) == len(sequence[i*int(window_size):i*int(window_size)+int(frag_size)]):
            pep_list.append(sequence[i*int(window_size):i*int(window_size)+int(frag_size)])
        else:
            break

    w = open(outputFile, 'w')


    for i, f in enumerate(pep_list):

        w.write(">sequence_"+str(i)+'\n')
        w.write(f+'\n')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Mp = subparsers.add_parser('MutatedPeptides')
    Mp.add_argument("-s","--sequence")
    Mp.add_argument("-m","--mutation_site_list")
    Mp.add_argument("-a","--AA_list")
    Mp.add_argument("-d", "--outputFile", required=None, default='out.fasta',   help="Path to out file")

    Rp = subparsers.add_parser('RandomPeptides')
    Rp.add_argument("-a","--AA_list")
    Rp.add_argument("-l","--pep_length")
    Rp.add_argument("-o","--out_pep_lenght")
    Rp.add_argument("-d", "--outputFile", required=None, default=os.path.join(os.getcwd(),'report_dirr'),   help="Path to out file")

    Sp = subparsers.add_parser('SlidingWindowPeptide')
    Sp.add_argument("-i","--InFile")
    Sp.add_argument("-w","--winSize")
    Sp.add_argument("-s","--FragSize")
    Sp.add_argument("-d", "--outputFile", required=None, default=os.path.join(os.getcwd(),'report_dirr'),   help="Path to out file")

    args = parser.parse_args()

    if sys.argv[1] == 'MutatedPeptides':
        MutatedPeptides(args.sequence, args.mutation_site_list, args.AA_list, args.outputFile)

    elif sys.argv[1] == 'RandomPeptides':
        RandomPeptides(args.AA_list, args.pep_length, args.out_pep_lenght, args.outputFile)

    elif sys.argv[1] == 'SlidingWindowPeptide':
        SlidingWindowPeptide(args.InFile, args.winSize, args.FragSize, args.outputFile)

    else:
        print("In Correct Option:")

