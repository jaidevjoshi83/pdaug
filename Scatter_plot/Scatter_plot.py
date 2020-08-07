import glob, os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt;plt.interactive(True)
import seaborn as sns, numpy as np, pandas as pd, random
import matplotlib.pyplot as plt
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
          <img src="out.png" alt="Smiley face" height="800" width="800">
        </div>

      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()

def BoxPlot(InFile, Feature1, Feature2, Feature3, Label, RotationX, RotationY, FigHight, FigWidth,  Workdirpath, htmlOutDir, htmlFname):

    Workdirpath = os.path.join(os.getcwd(),'report_dir')

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(InFile, sep="\t")
    plt.figure(figsize=(int(FigHight),int(FigWidth)))

    if Feature3 == False:
        sns.scatterplot(x=Feature1, y=Feature2, hue=Label, data=df)
        plt.xticks(rotation=int(RotationX))
        plt.yticks(rotation=int(RotationY))
        plt.savefig(os.path.join(Workdirpath, 'out.png'))

    else:

        sns.set_style("whitegrid", {'axes.grid' : False})

        fig = plt.figure(figsize=(6,6))
        ax = Axes3D(fig)
        g = ax.scatter(df[Feature1], df[Feature2], df[Feature3], c=df[Feature1], marker='o', depthshade=False, cmap='Paired')

        ax.set_xlabel(Feature1)
        ax.set_ylabel(Feature2)
        ax.set_zlabel(Feature3)

        plt.savefig(os.path.join(Workdirpath, htmlOutDir, 'out.png'))

    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))

if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    parser.add_argument("-F1", "--Feature1", required=True, default=True, help="Path to target tsv file")   
    parser.add_argument("-F2", "--Feature2", required=True, default=True, help="Roatate ticks")
    parser.add_argument("-F3", "--Feature3", required=False, default=False, help="Roatate ticks")
    parser.add_argument("-Rx", "--RotationX", required=False, default=0, help="Roatate ticks")   
    parser.add_argument("-Ry", "--RotationY", required=False, default=0, help="Roatate ticks")
    parser.add_argument("-H", "--FigHight", required=False,  default=6,  help="Figure Hight")
    parser.add_argument("-W", "--FigWidth", required=False, default=4, help="Figure Width")
    parser.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
    parser.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
    parser.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
    parser.add_argument("-L","--Label", required=False, default=False, help="Working Directory Path")
    args = parser.parse_args()

BoxPlot(args.InFile, args.Feature1, args.Feature2, args.Feature3, args.Label, args.RotationX, args.RotationY, args.FigHight, args.FigWidth, args.Workdirpath,  args.htmlOutDir, args.htmlFname)