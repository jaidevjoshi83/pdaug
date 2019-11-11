import sys
import itertools
import pandas as pd
import random
import os
from itertools import permutations
from random import shuffle
import argparse, sys


def MutatedPeptides(input_file, index_list, AA):

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
    return out_put


#print MutatedPeptides('jai.txt', '1,4,5', 'A,I')

def RandomPeptides(AAs, pep_length, out_pep_num):

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
    return out_pep_lib


def SlidingWindowPeptide(infile, window_size, frag_size):

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
    return pep_list

  
#a = "MGSSHHHHHHSSGLVPRGSHMARVTLVLRYAARSDRGLVRANNEDSVYAGARLLALADGMGGHAAGEVASQLVIAALAHLDDDEPGGDLLAKLDAAVRAGNSAIAAQVEMEPDLEGMGTTLTAILFAGNRLGLVHIGDSRGYLLRDGELTQITKDDTFVQTLVDEGRITPEEAHSHPQRSLIMRALTGHEVEPTLTMREARAGDRYLLCSDGLSDPVSDETILEALQIPEVAESAHRLIELALRGGGPDNVTVVVADVVD"
#AAs =ILKMFPSTWYV'
#AA = ['A','R','N','D','B','C','E','Q','Z','G','H','I','L','K','M','F','P','S','T','W','Y','V']
#to_modify = '--------------------'
#s = [3,5,9,14]
#print MutatedPeptides('jai.txt', [3,4,5,6,7] , ['A,T,Y,I'] )

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Ran = subparsers.add_parser('MutatedPeptides')
    Ran.add_argument("-s","--sequence")
    Ran.add_argument("-m","--mutation_site_list")
    Ran.add_argument("-a","--AA_list")

    Ran = subparsers.add_parser('RandomPeptides')
    Ran.add_argument("-a","--AA_list")
    Ran.add_argument("-l","--pep_length")
    Ran.add_argument("-o","--out_pep_lenght")

    Ran = subparsers.add_parser('SlidingWindowPeptide')
    Ran.add_argument("-i","--infile")
    Ran.add_argument("-w","--winSize")
    Ran.add_argument("-s","--FragSize")

    args = parser.parse_args()

    if sys.argv[1] == 'MutatedPeptides':
        print MutatedPeptides(args.sequence, args.mutation_site_list, args.AA_list)

    elif sys.argv[1] == 'RandomPeptides':
        print RandomPeptides(args.AA_list, args.pep_length, args.out_pep_lenght)

    elif sys.argv[1] == 'SlidingWindowPeptide':
        print SlidingWindowPeptide(args.infile, args.winSize, args.FragSize)

    else:
        print"You entered Wrong Values: "

