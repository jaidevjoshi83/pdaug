from modlamp.analysis import GlobalAnalysis
from modlamp.analysis import *
import pandas as pd
import os, sys
import argparse

parser = argparse.ArgumentParser(description='Deployment tool')
subparsers = parser.add_subparsers()

CalcAAFreq = subparsers.add_parser('CalcAAFreq')
CalcAAFreq.add_argument("-I","--InFile", required=True, default=None, help="")
CalcAAFreq.add_argument("-T","--PlotFile", required=False, default='Out.pdf', help="out.png")
CalcAAFreq.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")

H = subparsers.add_parser('H')
H.add_argument("-I","--InFile", required=True, default=None, help="")
H.add_argument("-S","--Scale", required=False, default='eisenberg', help="hydrophobicity scale to use. For available scales, see modlamp.descriptors.PeptideDescriptor.")
H.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")

uH = subparsers.add_parser('uH')
uH.add_argument("-I","--InFile", required=True, default=None, help="")
uH.add_argument("-S","--Scale", required=False, default='eisenberg', help="hydrophobicity scale to use. For available scales, see modlamp.descriptors.PeptideDescriptor.")
uH.add_argument("-W", "--Window", required=False, default=1000, help="")
uH.add_argument("-A", "--Angle", required=False, default=100, help="")
uH.add_argument("-M", "--Modality", required=False, default='max', help="")
uH.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")

charge = subparsers.add_parser('charge')
charge.add_argument("-I","--InFile", required=True, default=None, help="")
charge.add_argument("-p", "--ph", required=False, default=7.0, help="")
charge.add_argument("-A", "--Amide", required=False, default=True, help="")
charge.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")

Len = subparsers.add_parser('Len')
Len.add_argument("-I","--InFile", required=True, default=None, help="")
Len.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")

PlotSaummary = subparsers.add_parser('PlotSummary')
PlotSaummary.add_argument("-I1","--InFile1", required=True, default=None, help="")
PlotSaummary.add_argument("-I2", "--InFile2", required=True, default=None, help="Out.tsv")
PlotSaummary.add_argument("--OutFile", required=False, default='Out.pdf', help="out.pdf")
PlotSaummary.add_argument("--ImageFile", required=False, help="HTML out file", default="out.png")

args = parser.parse_args()


def ReturnPeptide(Infile):

    file = open(Infile)
    lines = file.readlines()

    Index = []
    Pep = []

    for line in lines:
        if '>' in line:
            line = line.strip('\n')
            line = line.strip('\r')
            Index.append(line.strip('\n'))
        else:
            line = line.strip('\n')
            line = line.strip('\r')
            Pep.append(line)
    return Pep, Index

if sys.argv[1] == 'CalcAAFreq':

    Pep, Index = ReturnPeptide(args.InFile)
    g = GlobalAnalysis(Pep)
    g.calc_aa_freq(plot=True, color='#83AF9B', filename='out.png')
    df1 =  pd.DataFrame(g.aafreq[0], columns=['aa_freq'])
    df1.to_csv(args.OutFile,  sep='\t', index=None)
    os.system('ls')

elif sys.argv[1] == 'H':

    Pep, _ = ReturnPeptide(args.InFile)
    g = GlobalAnalysis(Pep)
    g.calc_H(args.Scale)
    df1 = pd.DataFrame(g.H[0].T, columns=['H'])
    df1.to_csv(args.OutFile,  sep='\t', index=None)

elif sys.argv[1] == 'uH':

    Pep, _ = ReturnPeptide(args.InFile)

    g = GlobalAnalysis(Pep)
    g.calc_uH(int(args.Window), int(args.Angle), args.Modality)
    df1 = pd.DataFrame(g.uH[0].T, columns=['uH'])
    df1.to_csv(args.OutFile,  sep='\t', index=None)
  

elif sys.argv[1] == 'charge':

    Pep, _ = ReturnPeptide(args.InFile)

    for p in Pep:
        print (p)

    g = GlobalAnalysis(Pep)

    if args.Amide == 'true':
        amide = True
    else:
      amide = False

    g.calc_charge(float(args.ph), amide)
    df1 = pd.DataFrame(g.charge[0].T, columns=['charge'])
    df1.to_csv(args.OutFile,  sep='\t', index=None)

elif sys.argv[1] == 'Len':

    Pep, _ = ReturnPeptide(args.InFile)
    df1 = pd.DataFrame([len(x) for x in Pep], columns=['c'])
    df1.to_csv( args.OutFile,  sep='\t', index=None)

elif sys.argv[1] == "PlotSummary":

    seqs1, _ = ReturnPeptide(args.InFile1)
    seqs2, _ = ReturnPeptide(args.InFile2)

    g = GlobalAnalysis([seqs1, seqs2])

    g.plot_summary(filename='out.png', colors=None, plot=True)






