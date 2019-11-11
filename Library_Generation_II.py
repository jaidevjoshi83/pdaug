import sys
import itertools
import pandas as pd
import random
import os
from itertools import permutations
from random import shuffle
import argparse, sys
import pandas as pd


def MutatedPeptides(input_file, index_list, AA, output_dir):


    if not os.path.exists(os.path.join(os.getcwd(), output_dir)):
        os.makedirs(os.path.join(os.getcwd(), output_dir))

    f = open(input_file)
    line = f.readline()
    input_seq = line.strip('\n')
    to_modify = [x for x in input_seq]

    index_list = index_list.split(',')
    AA = AA.split(',')

    if len(AA) < len(index_list):
        print ("Index list should be < or = to the AA list")
        exit()
    else:
        pass

    if len(to_modify) > 20:
        print ("Max peptide lenth 20")
        exit()
    else:
        pass

    if len(index_list) > 4:
        print ("Max AA modificatin sites 4")
        exit()
    else:
        pass

    if len(AA) > 20:
        print ("Max AA 20")
        exit()
    else:
        pass

    out_put = []
    replacements = list(itertools.permutations(AA, len(index_list)))
    counter = 0
    
    for replacement in replacements:
        for i,index in enumerate(index_list):
              to_modify[int(index_list[i])-1] = replacement[i]
        counter = counter + 1
        out_put.append("".join(to_modify).upper())

    df = pd.DataFrame(out_put, columns=["Peptide"])
    df.to_csv(os.path.join(output_dir,'pep.tsv'), index=False,sep='\t')



def RandomPeptides(AAs, pep_length, out_pep_num, output_dir):

    if not os.path.exists(os.path.join(os.getcwd(), output_dir)):
        os.makedirs(os.path.join(os.getcwd(), output_dir))

    if int(pep_length) > 20:
        print "Max peptide lenth 20"
        exit()
    else:
        pass

    if int(out_pep_num) > 10000:
        print "Max peptide library 10000"
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

    df = pd.DataFrame(out_pep_lib, columns=["Peptide"])
    df.to_csv(os.path.join(output_dir,'pep.tsv'), index=False,sep='\t')


def SlidingWindowPeptide(infile, window_size, frag_size, output_dir):

    if not os.path.exists(os.path.join(os.getcwd(), output_dir)):
        os.makedirs(os.path.join(os.getcwd(), output_dir))

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

    f = open(infile)

    line = f.readline()
    sequence = line.strip('\n')
    pep_list = []

    for i in range(int(frag_size)):
        if int(frag_size) == len(sequence[i*int(window_size):i*int(window_size)+int(frag_size)]):
            pep_list.append(sequence[i*int(window_size):i*int(window_size)+int(frag_size)])
        else:
            break

    print pep_list
    df = pd.DataFrame(pep_list, columns=["Peptide"])
    df.to_csv(os.path.join(output_dir,'pep.tsv'), index=False,sep='\t')



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Ran = subparsers.add_parser('MutatedPeptides')
    Ran.add_argument("-s","--sequence")
    Ran.add_argument("-m","--mutation_site_list")
    Ran.add_argument("-a","--AA_list")
    Ran.add_argument("-d", "--out_dir_name", required=None, default=os.path.join(os.getcwd(),'report_dirr'),   help="Path to out file")

    Ran = subparsers.add_parser('RandomPeptides')
    Ran.add_argument("-a","--AA_list")
    Ran.add_argument("-l","--pep_length")
    Ran.add_argument("-o","--out_pep_lenght")
    Ran.add_argument("-d", "--out_dir_name", required=None, default=os.path.join(os.getcwd(),'report_dirr'),   help="Path to out file")

    Ran = subparsers.add_parser('SlidingWindowPeptide')
    Ran.add_argument("-i","--infile")
    Ran.add_argument("-w","--winSize")
    Ran.add_argument("-s","--FragSize")
    Ran.add_argument("-d", "--out_dir_name", required=None, default=os.path.join(os.getcwd(),'report_dirr'),   help="Path to out file")

    args = parser.parse_args()

    if sys.argv[1] == 'MutatedPeptides':
        MutatedPeptides(args.sequence, args.mutation_site_list, args.AA_list, args.out_dir_name)

    elif sys.argv[1] == 'RandomPeptides':
        RandomPeptides(args.AA_list, args.pep_length, args.out_pep_lenght, args.out_dir_name)

    elif sys.argv[1] == 'SlidingWindowPeptide':
        SlidingWindowPeptide(args.infile, args.winSize, args.FragSize, args.out_dir_name)

    else:
        print"You entered Wrong Values: "

