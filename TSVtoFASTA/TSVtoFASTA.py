from modlamp.core import BaseDescriptor
import pandas as pd
import os
import argparse


def TSVtoFASTA(InFile, Method):

    if Method == 'WithClassLable':

        f = open(InFile)
        lines = f.readlines()

        of1 = open('Positive.fasta','w')
        of2 = open('Negative.fasta','w')

        n = 0
        m = 0

        for line in lines:

            if '1' in line.split('\t')[1].strip('\n'):
                n= n+1
                of1.write('>peptide_'+str(n)+'\n')
                of1.write(line.split('\t')[0]+'\n')

            if '0' in line.split('\t')[1].strip('\n'):
                m= m+1
                of2.write('>peptide_'+str(m)+'\n')
                of2.write(line.split('\t')[0]+'\n')

    elif Method == 'NoClassLable':

        f = open(InFile)
        lines = f.readlines()
        of1 = open('Out.fasta','w')

        for i, line in enumerate(lines[1:]):
            of1.write('>peptide_'+str(i)+'\n')
            of1.write(line.split('\t')[0])

    else:
        pass

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-I", "--InFile", required=True, default=None, help=".fasta or .tsv")
    parser.add_argument("-M", "--Method", required=True, default=None, help="Path to target tsv file")
    args = parser.parse_args()

    TSVtoFASTA(args.InFile, args.Method)



