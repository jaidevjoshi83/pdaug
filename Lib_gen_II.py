a = "MGSSHHHHHHSSGLVPRGSHMARVTLVLRYAARSDRGLVRANNEDSVYAGARLLALADGMGGHAAGEVASQLVIAALAHLDDDEPGGDLLAKLDAAVRAGNSAIAAQVEMEPDLEGMGTTLTAILFAGNRLGLVHIGDSRGYLLRDGELTQITKDDTFVQTLVDEGRITPEEAHSHPQRSLIMRALTGHEVEPTLTMREARAGDRYLLCSDGLSDPVSDETILEALQIPEVAESAHRLIELALRGGGPDNVTVVVADVVD"


import sys
import itertools
import itertools
import pandas as pd
import random
import os
from itertools import permutations
from random import shuffle


def frg_gen(ofs, fs, a):

    frg_list = []

    if ofs > fs:
        print "Offset count be greater then frg size"
        sys.exit()

    else:
        pass
    for i in range(0,len(a)):

        n = 0
        i = (n+i)*ofs
        e = i+fs
        if e < len(a):
            frg_list.append(a[i:e])
            #print i, e
    print frg_list
    return set(frg_list)



AA = ['A','R','N','D','B','C','E','Q','Z','G','H','I','L','K','M','F','P','S','T','W','Y','V']
to_modify = '--------------------'
s = [3,5,9, 14]


def mut_pep_gene(in_seq, index_list, AA):

    out_put = []
    replacements = [x for x in itertools.permutations(AA,4)]
    counter = 0
    to_modify = [x for x in in_seq]

    for replacement in replacements:
        for i,index in enumerate(index_list):
              to_modify[index_list[i]-1] = replacement[i]
        counter = counter + 1
        out_put.append("".join(to_modify).upper())

    return out_put

m = mut_pep_gene(to_modify, s, AA)


AAs = 'ARNDCEQGHILKMFPSTWYV'


def Random_peptides(AAs, pep_len, pep_num):

    output_peptide_library = []
    raw = [x for x in AA]
    un_seq = []

    print raw

    for x in range(5):
        for i in range(15):
            un_seq.append(random.choice(raw))




    #for i in pep_len:
     #   print un_seq.append(random.choice(raw))

    #for x in range(pep_num):
     #   for a in range(pep_len):
      #      un_seq.append(random.choice(raw))
       # print un_seq
        

    #output_peptide_library.append("".join(un_seq))


Random_peptides(AAs, 15, 10)




