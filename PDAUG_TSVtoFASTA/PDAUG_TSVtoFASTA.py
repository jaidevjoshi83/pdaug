import os
import argparse


def TSVtoFASTA(InFile, Method, Positive, Negative, OutFile):

    if Method == 'WithClassLabel':

        f = open(InFile)
        lines = f.readlines()

        of1 = open(Positive,'w')
        of2 = open(Negative,'w')

        n = 0
        m = 0
        
        l = []

        for line in lines[1:]:
            l.append(line.split('\t')[1].strip('\n').strip('\r'))
        l = list(set(l))

        print(l)

        for line in lines:

            if l[1] in line.split('\t')[1].strip('\n').strip('\r'):
                n= n+1
                of1.write('>peptide_'+str(n)+'_'+str(l[0])+'\n')
                of1.write(line.split('\t')[0]+'\n')

            if l[0] in line.split('\t')[1].strip('\n').strip('\r'):
                m= m+1
                of2.write('>peptide_'+str(m)+'_'+str(l[1])+'\n')
                of2.write(line.split('\t')[0]+'\n')

    elif Method == 'NoClassLabel':

        f = open(InFile)
        lines = f.readlines()
        of1 = open(OutFile,'w')

        for i, line in enumerate(lines[1:]):
            of1.write('>peptide_'+str(i)+'\n')
            of1.write(line.split('\t')[0]+'\n')

    else:
        pass

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-I", "--InFile", required=True, default=None, help=".fasta or .tsv")
    parser.add_argument("-P", "--Postvs", required=False, default='FirstDataFile.fasta', help="Path to target tsv file")
    parser.add_argument("-N", "--Negtvs", required=False, default='SecondDataFile.fasta', help="Path to target tsv file")
    parser.add_argument("-O", "--OutFile", required=False, default='OutFile.fasta', help="Path to target tsv file")
    parser.add_argument("-M", "--Method", required=True, default=None, help="Path to target tsv file")
    args = parser.parse_args()

    TSVtoFASTA(args.InFile, args.Method, args.Postvs, args.Negtvs, args.OutFile)