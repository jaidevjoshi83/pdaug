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
        print ("Offset count be greater then frg size")
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
    print (frg_list)
    return set(frg_list)


#print (frg_gen(4,10,a))


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

    out_pep_lib = []
    raw = [x for x in AA]

    for x in range(pep_num):
        un_seq = []
        for i in range(pep_len):
            un_seq.append(random.choice(raw))
        
        out_pep_lib.append("".join(un_seq))
    return out_pep_lib

l = Random_peptides(AAs, 15, 1000)

#print (len(l))
#print (len(set(l))) 



w = 4
f = 10

a = "MGSSHHHHHHSSGLVPRGSHMARVTLVLRYAARSDRGLVRANNEDSVYAGARLLA"
     

print (a[0:10])
print (a[10:20])
print (a[20:30])
print (a[30:40])
print (a[40:50])

for i in range((int(len(a)/f))):

    i = i+10
    print (i, i)


s:s+10








