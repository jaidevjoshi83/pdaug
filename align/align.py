
import Levenshtein

from Bio import Align

from itertools import combinations

f = open('seq.txt')

lines = f.readlines()

pep = []

for line in lines:
    if ">" in line:
        pass 
    else:

        line =  line.strip('\r\n')
        line = line.strip(' ')
        pep.append(line)

from itertools import combinations

comb = combinations(pep, 2)


pep1 = ",".join(pep)


outfile = open("out1.tsv", 'w')
outfile2 = open("lev.tsv", 'w')

outfile.write("source"+"\t"+'target'+"\t"+'weight'+"\n")

aligner = Align.PairwiseAligner()

for x, i in enumerate(comb):

    print i

    seq1 =  i[0].strip('\r\n')
    seq1 =  i[0].strip(' ')
    seq2 =  i[1].strip('\r\n')
    seq2 =  i[1].strip(' ')
    alignment = aligner.align(seq1,seq2)
    outfile.write(seq1+"\t"+seq2+"\t"+str(alignment.score)+"\n")
    outfile2.write(seq1+"\t"+seq2+"\t"+str(Levenshtein.ratio(seq1, seq2 ))+"\n")
        #print p, x, alignment.score
outfile.close()


"""
from Bio import Align


f = open('seq1.txt')

lines = f.readlines()

pep = []

for line in lines:
    if ">" in line:
        pass 
    else:

        line =  line.strip('\r\n')
        line = line.strip(' ')
        pep.append(line)

print pep



#print pep

pep1 = ",".join(pep)

#pep1 = "KKVVEKNADPET,ADPETTLLVYLR,RKLGLCGTKLGCGEG"

#pep = pep1.split(',')

outfile = open("out1.csv", 'w')

outfile.write("index,"+str(pep1)+'\n')

aligner = Align.PairwiseAligner()
for p in pep:
    outfile.write(p)
    for i, x in  enumerate(pep):
        alignment = aligner.align(p,x)
        outfile.write(','+str(alignment.score))
    outfile.write('\n')
outfile.close()

"""
"""
import Levenshtein

from Bio import Align

f = open('seq.txt')

lines = f.readlines()

pep = []

for line in lines:
    if ">" in line:
        pass 
    else:

        line =  line.strip('\r\n')
        line = line.strip(' ')
        pep.append(line)

pep1 = ",".join(pep)





outfile = open("out1.tsv", 'w')
outfile2 = open("lev.tsv", 'w')

outfile.write("source"+"\t"+'target'+"\t"+'weight'+"\n")

aligner = Align.PairwiseAligner()
for p in pep:
    outfile.write(p)
    for i, x in  enumerate(pep):
        alignment = aligner.align(p,x)
        Levenshtein.ratio(p, x )
        outfile.write(p+"\t"+x+"\t"+str(alignment.score)+"\n")
        outfile2.write(p+"\t"+x+"\t"+str(Levenshtein.ratio(p, x ))+"\n")
        #print p, x, alignment.score
outfile.close()
"""

