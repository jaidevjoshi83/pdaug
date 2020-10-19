from modlamp.plot import helical_wheel
from modlamp.plot import plot_pde
from modlamp.plot import plot_violin
from modlamp.plot import plot_aa_distr
import pandas as pd
import argparse
import sys, os



parser = argparse.ArgumentParser(description='Deployment tool')
subparsers = parser.add_subparsers()

HelWhl = subparsers.add_parser('HelWhl')

HelWhl.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
HelWhl.add_argument("-C","--colorcoding", required=False, default='rainbow', help="available: , charge, polar, simple, amphipathic, none")
HelWhl.add_argument("-L","--lineweights", required=False, default=True, help="(boolean) defines whether connection lines decrease in thickness along the sequence")
HelWhl.add_argument("-F","--filename", required=False, default="out.png", help="")
HelWhl.add_argument("-s","--seq", required=False, default=False, help="")
HelWhl.add_argument("-M","--movment", required=False, default=False, help="")
HelWhl.add_argument("-O", "--OutFile", required=False, default="out.png", help="OutFile")


PltPde = subparsers.add_parser('PltPde')
PltPde.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
PltPde.add_argument("-l", "--ClmList", required=True, default=None, help="")
PltPde.add_argument("-F","--filename", required=False, default="out.png", help="filename where to safe the plot. default = None > show the plot")
PltPde.add_argument("-O", "--OutFile", required=False, default="out.png", help="OutFile")

PltVio = subparsers.add_parser('PltVio')
PltVio.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
PltVio.add_argument("-l", "--ClmList", required=True, default=None, help="Column list")
PltVio.add_argument("-C","--colors", required=False, default=None, help='Data to be plotted')
PltVio.add_argument("-B","--bp", required=False, default=False, help="Print a box blot inside violin")
PltVio.add_argument("-T","--title", required=False, default=None, help="Title of the plot.")
PltVio.add_argument("-a","--axlabels", required=False, default=None, help="list containing the axis labels for the plot")
PltVio.add_argument("-M","--y_max", required=False, default=1, help='y-axis maximum.')
PltVio.add_argument("-m","--y_min", required=False, default=0, help="y-axis minimum.")
PltVio.add_argument("-O", "--OutFile", required=False, default="out.png", help="OutFile")


PltAaDis = subparsers.add_parser('PltAaDis')
PltAaDis.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
PltAaDis.add_argument("-O", "--OutFile", required=False, default="out.png", help="OutFile")

args = parser.parse_args()


if sys.argv[1] == 'HelWhl':

    f = open(args.InFile)
    lines = f.readlines()
    sequence = lines[1].strip('\n')

    if args.movment == 'true':
        mvt = True
    else:
      mvt = False

    if args.seq == 'true':
        sq = True
    else:
      sq = False

    if args.lineweights == 'true':
        lw = True
    else:
      lw = False

    helical_wheel(sequence, colorcoding=args.colorcoding, lineweights=args.lineweights, filename=args.OutFile, seq=args.seq, moment=mvt)


elif sys.argv[1] == 'PltPde':

    df = pd.read_csv(args.InFile, sep="\t")

    data = df[args.ClmList.split(',')].as_matrix().T

    plot_pde(data, filename=args.OutFile)

elif sys.argv[1] == 'PltVio':

    df = pd.read_csv(args.InFile, sep="\t")

    data = df[args.ClmList.split(',')].as_matrix().T

    c = ['#0B486B']*len(args.ClmList.split(','))

    plot_violin(data, colors=c, bp=True, filename=args.OutFile)


elif sys.argv[1] == 'PltAaDis':

    f = open(args.InFile)
    lines = f.readlines()

    sequences = []

    for line in lines:
        if '>' in line:
            pass
        else:
            sequences.append(line.strip('\n'))

    plot_aa_distr(sequences, color='#0B486B', filename=args.OutFile) 









