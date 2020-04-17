from modlamp.plot import helical_wheel
from modlamp.plot import plot_pde
from modlamp.plot import plot_violin
from modlamp.plot import plot_aa_distr
import pandas as pd
import argparse
import sys, os



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
      <h1> Machine Learning Algorithm Assessment Report </h1>
    </div>
    <div class="container">
      <div class="row">
        <div class="col-sm-4">
          <img src="out.png" alt="Smiley face" height="500" width="400">
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

HelWhl = subparsers.add_parser('HelWhl')

HelWhl.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
HelWhl.add_argument("-C","--colorcoding", required=False, default='rainbow', help="available: , charge, polar, simple, amphipathic, none")
HelWhl.add_argument("-L","--lineweights", required=False, default=True, help="(boolean) defines whether connection lines decrease in thickness along the sequence")
HelWhl.add_argument("-F","--filename", required=False, default="out.png", help="")
HelWhl.add_argument("-s","--seq", required=False, default=False, help="")
HelWhl.add_argument("-M","--moment", required=False, default=False, help="")
HelWhl.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
HelWhl.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
HelWhl.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

PltPde = subparsers.add_parser('PltPde')
PltPde.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
PltPde.add_argument("-l", "--ClmList", required=True, default=None, help="")
PltPde.add_argument("-T","--Title", required=False, default='rbf', help="(str) plot title")
PltPde.add_argument("-A","--axlabels", required=False, default=None, help="(list of str) list containing the axis labels for the plot")
PltPde.add_argument("-F","--filename", required=False, default="out.png", help="filename where to safe the plot. default = None > show the plot")
PltPde.add_argument("-L","--legendloc", required=False, default=2, help="location of the figures legend. 1 = top right, 2 = top left")
PltPde.add_argument("-M","--x_max", required=False, default=1, help="x-axis minimum")
PltPde.add_argument("-m","--x_min", required=False, default=0, help="x-axis maximum")
#PltPde.add_argument("-c","--Colors", required=False, default=None, help="list of colors (readable by matplotlib, e.g. hex) to be used to plot different data classes")
PltPde.add_argument("-a","--alpha", required=False, default=0.2, help="color alpha for filling pde curve")
PltPde.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
PltPde.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
PltPde.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

PltVio = subparsers.add_parser('PltVio')
PltVio.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
PltVio.add_argument("-l", "--ClmList", required=True, default=None, help="")
PltVio.add_argument("-C","--colors", required=False, default=None, help='data to be plotted')
PltVio.add_argument("-B","--bp", required=False, default=False, help="print a box blot inside violin")
PltVio.add_argument("-F","--filename", required=False, default="jai.png", help="location / filename where to save the plot to. default = None > show the plot")
PltVio.add_argument("-T","--title", required=False, default=None, help="Title of the plot.")
PltVio.add_argument("-a","--axlabels", required=False, default=None, help="list containing the axis labels for the plot")
PltVio.add_argument("-M","--y_max", required=False, default=1, help='y-axis maximum.')
PltVio.add_argument("-m","--y_min", required=False, default=0, help="y-axis minimum.")
PltVio.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
PltVio.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
PltVio.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

PltAaDis = subparsers.add_parser('PltAaDis')
PltAaDis.add_argument("-I","--InFile", required=True, default=None, help="Input data file")
PltAaDis.add_argument("-C", "--color", required=False, default='#83AF9B', help="color to be used (matplotlib style / hex)")
PltAaDis.add_argument("-F", "--filename", required=False, default="jai.png", help="location / filename where to save the plot to. default = None > show the plot")
PltAaDis.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
PltAaDis.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
PltAaDis.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")

args = parser.parse_args()


if sys.argv[1] == 'HelWhl':

    if not os.path.exists(args.htmlOutDir):
        os.makedirs(args.htmlOutDir)

    F_Path  = os.path.join(args.Workdirpath, args.htmlOutDir, "out.png")

    f = open(args.InFile)

    lines = f.readlines()

    sequence = lines[1]

    helical_wheel(sequence, colorcoding=args.colorcoding, lineweights=args.lineweights, filename=F_Path, seq=args.seq, moment=args.moment)
    HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname))

elif sys.argv[1] == 'PltPde':

    if not os.path.exists(args.htmlOutDir):
        os.makedirs(args.htmlOutDir)

    F_Path  = os.path.join(args.Workdirpath, args.htmlOutDir, "out.png")

    df = pd.read_csv(args.InFile, sep="\t")
    data = df[args.ClmList.split(',')].as_matrix().T
  
    plot_pde(data, title=args.Title, axlabels=args.axlabels, filename=F_Path, legendloc=args.legendloc, x_min=args.x_min, x_max=args.x_max,  alpha=args.alpha)
    HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname))

elif sys.argv[1] == 'PltVio':

    if not os.path.exists(args.htmlOutDir):
        os.makedirs(args.htmlOutDir)

    F_Path  = os.path.join(args.Workdirpath, args.htmlOutDir, "out.png")

    df = pd.read_csv(args.InFile, sep="\t")
    data = df[args.ClmList.split(',')].as_matrix().T

    plot_violin(data, colors=args.colors, bp=args.bp, filename=F_Path, title=args.title, axlabels=args.axlabels, y_min=args.y_min, y_max=args.y_max)
    HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname))

elif sys.argv[1] == 'PltAaDis':

    if not os.path.exists(args.htmlOutDir):
        os.makedirs(args.htmlOutDir)

    F_Path  = os.path.join(args.Workdirpath, args.htmlOutDir, "out.png")

    f = open(args.InFile)

    lines = f.readlines()

    sequences = []

    for line in lines:
        if '>' in line:
            pass
        else:
            sequences.append(line.strip('\n'))

    plot_aa_distr(sequences, color=args.color, filename=F_Path)
    HTML_Gen(os.path.join(args.Workdirpath, args.htmlOutDir, args.htmlFname))








