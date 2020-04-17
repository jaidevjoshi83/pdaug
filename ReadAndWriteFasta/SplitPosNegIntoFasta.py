from modlamp.core import BaseDescriptor
import pandas as pd
import os
import argparse

"""
parser = argparse.ArgumentParser()
parser.add_argument("-I", "--InFile", required=True, default=None, help=".fasta or .tsv")
parser.add_argument("-O", "--OutFile", required=True, default=None, help=".fasta or .tsv")
parser.add_argument("-M", "--Method", required=True, default=None, help="Path to target tsv file")
parser.add_argument("-W","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
args = parser.parse_args()
"""


f = open('IO.tsv')
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
  







