from modlamp.analysis import GlobalAnalysis
from modlamp.analysis import *
import pandas as pd
import os, sys
import argparse


def HTML_Gen(html):

    out_html = open(html,'w')             
    part_1 =  """

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Bootstrap Example</title>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <body>
    <style>
    div.container_1 {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    div.table {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    </style>
    </head>
    <div class="jumbotron text-center">
      <h1> Plot summary </h1>
    </div>
    <div class="container">
      <div class="row">
        <div class="col-sm-4">
          <img src="out.png" alt="Smiley face" height="800" width="1000">
        </div>
      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()


parser = argparse.ArgumentParser(description='Deployment tool')

subparsers = parser.add_subparsers()

CalcAAFreq = subparsers.add_parser('CalcAAFreq')
CalcAAFreq.add_argument("-I","--InFile", required=True, default=None, help="")
CalcAAFreq.add_argument("-T","--PlotFile", required=False, default='Out.png', help="Out.png")
CalcAAFreq.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")
CalcAAFreq.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
CalcAAFreq.add_argument("--htmlFname", required=False, help="HTML out file", default="jai.html")
CalcAAFreq.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

H = subparsers.add_parser('H')
H.add_argument("-I","--InFile", required=True, default=None, help="")
H.add_argument("-S","--Scale", required=False, default='eisenberg', help="hydrophobicity scale to use. For available scales, see modlamp.descriptors.PeptideDescriptor.")
H.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")
H.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

uH = subparsers.add_parser('uH')
uH.add_argument("-I","--InFile", required=True, default=None, help="")
uH.add_argument("-S","--Scale", required=False, default='eisenberg', help="hydrophobicity scale to use. For available scales, see modlamp.descriptors.PeptideDescriptor.")
uH.add_argument("-W", "--Window", required=False, default=1000, help="")
uH.add_argument("-A", "--Angle", required=False, default=100, help="")
uH.add_argument("-M", "--Modality", required=False, default='max', help="")
uH.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")
uH.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

charge = subparsers.add_parser('charge')
charge.add_argument("-I","--InFile", required=True, default=None, help="")
charge.add_argument("-p", "--ph", required=False, default=7.0, help="")
charge.add_argument("-A", "--Amide", required=False, default=True, help="")
charge.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")
charge.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

Len = subparsers.add_parser('Len')
Len.add_argument("-I","--InFile", required=True, default=None, help="")
Len.add_argument("--OutFile", required=False, default='Out.tsv', help="Out.tsv")
Len.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

PlotSaummary = subparsers.add_parser('PlotSummary')
PlotSaummary.add_argument("-I1","--InFile1", required=True, default=None, help="")
PlotSaummary.add_argument("-I2", "--InFile2", required=True, default=None, help="Out.tsv")
PlotSaummary.add_argument("-I3","--InFile3", required=False, default=None, help="")
PlotSaummary.add_argument("--OutFile", required=False, default='Out.png', help="out.png")
PlotSaummary.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="HTML Out Dir")
PlotSaummary.add_argument("--htmlFname", required=False, help="HTML out file", default="jai.html")
PlotSaummary.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

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

    if not os.path.exists(args.htmlOutDir):
        os.makedirs(args.htmlOutDir)

    Pep, Index = ReturnPeptide(args.InFile)

    print (len(Index))

    print (len(Pep))


    g = GlobalAnalysis(Pep)

    g.calc_aa_freq(plot=True, color='#83AF9B', filename=os.path.join(args.Workdirpath, args.htmlOutDir, args.PlotFile))
    df1 =  pd.DataFrame(g.aafreq[0], columns=['aa_freq'])
    df1.to_csv(os.path.join(args.Workdirpath, args.OutFile),  sep='\t', index=None)
    HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname))

elif sys.argv[1] == 'H':

    Pep, _ = ReturnPeptide(args.InFile)

    g = GlobalAnalysis(Pep)
    g.calc_H(args.Scale)
    df1 = pd.DataFrame(g.H[0].T, columns=['H'])
    df1.to_csv(os.path.join(args.Workdirpath, args.OutFile),  sep='\t', index=None)

elif sys.argv[1] == 'uH':

    Pep, _ = ReturnPeptide(args.InFile)

    g = GlobalAnalysis(Pep)

    g.calc_uH(int(args.Window), int(args.Angle), args.Modality)
    df1 = pd.DataFrame(g.uH[0].T, columns=['uH'])

    df1.to_csv(os.path.join(args.Workdirpath, args.OutFile),  sep='\t', index=None)
  

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
    df1.to_csv(os.path.join(args.Workdirpath, args.OutFile),  sep='\t', index=None)

elif sys.argv[1] == 'Len':

    Pep, _ = ReturnPeptide(args.InFile)

    df1 = pd.DataFrame([len(x) for x in Pep], columns=['c'])
    df1.to_csv(os.path.join(args.Workdirpath, args.OutFile),  sep='\t', index=None)

elif sys.argv[1] == "PlotSummary":
    if not os.path.exists(args.htmlOutDir):
        os.makedirs(args.htmlOutDir)

    seqs1, _ = ReturnPeptide(args.InFile1)
    seqs2, _ = ReturnPeptide(args.InFile2)

    if args.InFile3 == None:
        g = GlobalAnalysis([seqs1, seqs2])
    else:
        seqs3, _ = ReturnPeptide(args.InFile3)
        g = GlobalAnalysis([seqs1, seqs2, seqs3])

    g.plot_summary(filename=os.path.join(args.Workdirpath, args.htmlOutDir, 'Out.png'), colors=None, plot=True)
    HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname))





