import Levenshtein
from Bio import SeqIO

seq_list = []

f = open("positive.fasta")

lines = f.readlines()

for line in lines:
    if ">" not in line:
        print(line)
        seq_list.append(line.strip('\n'))

print(seq_list)