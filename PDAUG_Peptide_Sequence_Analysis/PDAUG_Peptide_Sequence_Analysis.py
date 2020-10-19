import  modlamp
from modlamp.analysis import *

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from modlamp.analysis import GlobalAnalysis
from modlamp.analysis import *

import pandas as pd
import os, sys
import argparse

parser = argparse.ArgumentParser(description='Deployment tool')
subparsers = parser.add_subparsers()

CalcAAFreq = subparsers.add_parser('CalcAAFreq')
CalcAAFreq.add_argument("-I","--InFile", required=True, default=None, help="")
CalcAAFreq.add_argument("-T","--PlotFile", required=False, default='out.pdf', help="out.pdf")
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
PlotSaummary.add_argument("--PlotFile", required=False, default='Out.pdf', help="out.pdf")
PlotSaummary.add_argument("--htmlFname", required="False", default='report.html', help="Output file")
PlotSaummary.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
PlotSaummary.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
PlotSaummary.add_argument("-fn", "--First_lib_name", required=True, help="Name of the fist peptide data")
PlotSaummary.add_argument("-sn", "--Second_lib_name", required=True, help="Name of the second peptide data")


args = parser.parse_args()



def SummaryPlot(Lib_1, Lib_2, First_lib_name, Second_lib_Name, Workdirpath, htmlOutDir, htmlFname):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)


    AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    Pep1, Index1 = ReturnPeptide(Lib_1)
    Pep2, Index2 = ReturnPeptide(Lib_2)

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "xy"}, {"type": "histogram"},   {"type": "box"}  ],[{"type": "violin"}, {"type": "violin"}, {"type": "scatter3d"} ]],
    subplot_titles=(" Amino Acid Fraction", "Global Charge", "Length Distribution", "Global Hydrophobicity", "Global Hydrophobic Movement", "Scatter Plot"))


    #########################################
    g = GlobalAnalysis([Pep1, Pep2])
    df = g.calc_aa_freq(plot=False)

    data1 = g.aafreq[0]
    data2 = g.aafreq[1] 

    fig.add_trace(go.Bar(x=AA, y=data1, name=First_lib_name, marker_color='#1F77B4'),  row=1, col=1)
    fig.add_trace(go.Bar( x=AA, y=data2,name=Second_lib_Name, marker_color='#FF7F0E'), row=1, col=1)
    fig.update_layout(showlegend=True)
    ##########################################


    #########################################
    d1  = GlobalDescriptor(Pep1)
    d1.calculate_charge(ph=7.4, amide=True)
    charge1 = [x[0] for x in d1.descriptor]

    d2  = GlobalDescriptor(Pep2)
    d2.calculate_charge(ph=7.4, amide=True)
    charge2 = [x[0] for x in d2.descriptor]

    fig.add_trace(go.Histogram(x=charge1, histnorm='probability', marker_color='#1F77B4', name=First_lib_name,  xbins=dict(  start=min(charge1), end=max(charge1), ), opacity=0.75), row=1, col=2)
    fig.add_trace(go.Histogram( x=charge2, histnorm='probability', marker_color='#FF7F0E', name=Second_lib_Name, xbins=dict( start=min(charge2), end=max(charge2), ),  opacity=0.75), row=1, col=2 )
    #fig.update_layout( ,  xaxis_title_text='Charge',  yaxis_title_text='Fraction',  bargap=0.1, bargroupgap=0.1 )
    ###########################################

    ##############################################################################
    Length1 = [len(x) for x in Pep1]
    Length2 = [len(x) for x in Pep2]

    fig.add_trace(go.Box(y=Length1, name=First_lib_name, marker_color='#1F77B4'), row=1, col=3)
    fig.add_trace(go.Box(y=Length2, name=Second_lib_Name, marker_color='#FF7F0E'), row=1, col=3)
    #############################################################################

    ########################################################################
    g = GlobalAnalysis([Pep1, Pep2])

    g.calc_H()

    h1 = g.H[0]
    h2 = g.H[1]

    fig.add_trace(go.Violin( y=h1,box_visible=True, name =First_lib_name, marker_color='#1F77B4', meanline_visible=True), row=2, col=1)
    fig.add_trace(go.Violin(y=h2,box_visible=True, name=Second_lib_Name, marker_color='#FF7F0E', meanline_visible=True), row=2, col=1)
    #################################################################

    #####################################
    uH = GlobalAnalysis([Pep1, Pep2])
    uH.calc_uH()

    uh1 = uH.uH[0]
    uh2 = uH.uH[1]

    fig.add_trace(go.Violin( y=uh1,box_visible=True, name =First_lib_name, marker_color='#1F77B4', meanline_visible=True), row=2, col=2)
    fig.add_trace(go.Violin(y=uh2,box_visible=True, name=Second_lib_Name, marker_color='#FF7F0E', meanline_visible=True), row=2, col=2)
    #######################################


    ############################################
    fig.add_trace(go.Scatter3d(x=h1, y=uh1, z=charge1, marker_color='#1F77B4', mode='markers', name=First_lib_name, marker_size=3.0),row=2, col=3)
    fig.add_trace(go.Scatter3d(x=h2, y=uh2, z=charge2, marker_color='#FF7F0E', mode='markers', name=Second_lib_Name, marker_size=3.0), row=2, col=3)
    fig.update_layout(scene = dict(xaxis_title='Hydrophobicity', yaxis_title='Hydrophobic Movement', zaxis_title='Charge'),uniformtext_minsize=4, font=dict(
            family="Times New Roman",
            size=12,
            color="black"))
    ###########################################

    fig.update_xaxes(title_text="Amino Acid", row=1, col=1)
    fig.update_xaxes(title_text="Global Charge", row=1, col=2)
    fig.update_xaxes(title_text="Peptide dataset", showgrid=False, row=1, col=3)
    fig.update_xaxes(title_text="Peptide dataset",  row=2, col=1)
    fig.update_xaxes(title_text="Peptide dataset",  row=2, col=2)

    fig.update_yaxes(title_text="Fraction", row=1, col=1)
    fig.update_yaxes(title_text="Fraction",  row=1, col=2)
    fig.update_yaxes(title_text="Length",  row=1, col=3)
    fig.update_yaxes(title_text="Global hydrophobicity", row=2, col=1)
    fig.update_yaxes(title_text="Global hydrophobic Movement", row=2, col=2)
    fig.write_html(os.path.join(Workdirpath, htmlOutDir, htmlFname))
    #fig.show()

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
    g.calc_aa_freq(plot=True, color='#83AF9B', filename='out.pdf')
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

    SummaryPlot(args.InFile1, args.InFile2, args.First_lib_name, args.Second_lib_name, args.Workdirpath, args.htmlOutDir, args.htmlFname)






