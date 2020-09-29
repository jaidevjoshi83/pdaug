import sys
import matplotlib
matplotlib.use('Agg')
import glob, os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt;plt.interactive(True)
import seaborn as sns, numpy as np, pandas as pd, random
from mpl_toolkits.mplot3d import Axes3D


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
          <img src="Out.png" alt="Smiley face" height="500" width="400">
        </div>

      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()

def HeatMapPlot(Infile,  FigHight, FigWidth, Rotation, Workdirpath, htmlOutDir, htmlFname):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(Infile, sep="\t")
    #print (df.columns[1:])

    plt.figure(figsize=(float(FigHight),float(FigWidth)))

    x_axis_labels = df.columns[1:]
    y_axis_labels = df[df.columns[0]].tolist()

    sns.set(font_scale=2)
    sns.heatmap(df[df.columns[1:]], center=0, vmin=0, vmax=1, yticklabels=y_axis_labels, xticklabels=x_axis_labels)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    plt.savefig(os.path.join(Workdirpath, htmlOutDir, "Out.png"))

    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))


def BoxPlot(InFile, Feature, RotationX, RotationY, FigHight, FigWidth,  Workdirpath, htmlOutDir, htmlFname):

    Workdirpath = os.path.join(os.getcwd(),'report_dir')

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(InFile, sep="\t")

    f = Feature.split(',')
    plt.figure(figsize=(int(FigHight),int(FigWidth)))
    sns.boxplot(data=df[f])
    plt.xticks(rotation=int(RotationX))
    plt.yticks(rotation=int(RotationY))
    plt.savefig(os.path.join(Workdirpath, htmlOutDir, "Out.png"), dpi=600)

    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))

def ScatterPlot(InFile, Feature1, Feature2, Feature3, Label, PlotType, RotationX, RotationY, FigHight, FigWidth,  Workdirpath, htmlOutDir, htmlFname):


    Workdirpath = os.path.join(os.getcwd(),'report_dir')

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(InFile, sep="\t")

    fig = plt.figure(figsize=(int(FigHight),int(FigWidth)))


    if PlotType== "2D":
        sns.scatterplot(x=Feature1, y=Feature2, hue=Label, data=df)
        plt.xticks(rotation=int(RotationX))
        plt.yticks(rotation=int(RotationY))
        plt.savefig(os.path.join(Workdirpath, htmlOutDir, 'Out.png'))

    elif PlotType == "3D":

        sns.set_style("whitegrid", {'axes.grid' : False})
        ax = Axes3D(fig)

        g = ax.scatter(df[Feature1], df[Feature2], df[Feature3], c=df[Label], marker='o', depthshade=False, cmap='Paired')
     
        ax.set_xlabel(Feature1)
        ax.set_ylabel(Feature2)
        ax.set_zlabel(Feature3)

        plt.savefig(os.path.join(Workdirpath, htmlOutDir, 'Out.png'))

    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))


if __name__=="__main__":


    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    HM = subparsers.add_parser('HeatMap')
    HM.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    HM.add_argument("-H", "--FigHight",  required=False, default=4, help="Figure hight")
    HM.add_argument("-W", "--FigWidth", required=False, default=6, help="Figure width")
    HM.add_argument("-R", "--Rotation", required=False, default=1, help="Rotation of the label")
    HM.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="Path to html directory")
    HM.add_argument("--htmlFname", required=False, help="HTML out file", default="report.html")
    HM.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Path to working directory")

    BP = subparsers.add_parser('BoxPlot')
    BP.add_argument("-I", "--InFile", required=True, default=None, help="Input file")
    BP.add_argument("-Rx", "--RotationX", required=False, default=0, help="Roatate xticks")
    BP.add_argument("-Ry", "--RotationY", required=False, default=0, help="Roatate yticks")
    BP.add_argument("-H", "--FigHight", required=False,  default=6,  help="Figure Hight")
    BP.add_argument("-W", "--FigWidth", required=False, default=4, help="Figure Width")
    BP.add_argument("-F", "--Features", required=True, default=None, help="Feature list")
    BP.add_argument("-O", "--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="Path to html dir")
    BP.add_argument("-Hf", "--htmlFname", required=False, help="HTML out file", default="report.html")
    BP.add_argument("-Wp", "--Workdirpath", required=False, default=os.getcwd(), help="Path to Working Directory")
    
    SP = subparsers.add_parser('ScatterPlot')
    SP.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    SP.add_argument("-F1", "--Feature1", required=True, default=True, help="Path to target tsv file")   
    SP.add_argument("-F2", "--Feature2", required=True, default=True, help="Roatate ticks")
    SP.add_argument("-F3", "--Feature3", required=False,  help="Roatate ticks")
    SP.add_argument("-Rx", "--RotationX", required=False, default=0, help="Roatate ticks")   
    SP.add_argument("-Ry", "--RotationY", required=False, default=0, help="Roatate ticks")
    SP.add_argument("-H", "--FigHight", required=False,  default=6,  help="Figure Hight")
    SP.add_argument("-W", "--FigWidth", required=False, default=4, help="Figure Width")
    SP.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
    SP.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
    SP.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
    SP.add_argument("-T", "--PlotType", required=True,  help="")
    SP.add_argument("-L","--Label", required=False, default=False, help="Working Directory Path")
    
    args = parser.parse_args()

    if sys.argv[1] == "HeatMap":
        HeatMapPlot(args.InFile,  args.FigHight, args.FigWidth, args.Rotation, args.Workdirpath, args.htmlOutDir, args.htmlFname)

    elif sys.argv[1] == "ScatterPlot":
        ScatterPlot(args.InFile, args.Feature1, args.Feature2, args.Feature3, args.Label, args.PlotType, args.RotationX, args.RotationY, args.FigHight, args.FigWidth, args.Workdirpath,  args.htmlOutDir, args.htmlFname)

    elif sys.argv[1] == "BoxPlot":
        BoxPlot(args.InFile, args.Features, args.RotationX, args.RotationY, args.FigHight, args.FigWidth,  args.Workdirpath,  args.htmlOutDir, args.htmlFname)
   
    else:
        print("In Correct Option:")
