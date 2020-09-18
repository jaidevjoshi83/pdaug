import matplotlib
matplotlib.use('Agg')
import glob, os
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt




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

if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile", required=True, default=None, help="Input file")
    parser.add_argument("-Rx", "--RotationX", required=False, default=0, help="Roatate xticks")
    parser.add_argument("-Ry", "--RotationY", required=False, default=0, help="Roatate yticks")
    parser.add_argument("-H", "--FigHight", required=False,  default=6,  help="Figure Hight")
    parser.add_argument("-W", "--FigWidth", required=False, default=4, help="Figure Width")
    parser.add_argument("-F", "--Features", required=True, default=None, help="Feature list")
    parser.add_argument("-O", "--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="Path to html dir")
    parser.add_argument("-Hf", "--htmlFname", required=False, help="HTML out file", default="report.html")
    parser.add_argument("-Wp", "--Workdirpath", required=False, default=os.getcwd(), help="Path to Working Directory")
    args = parser.parse_args()

BoxPlot(args.InFile, args.Features, args.RotationX, args.RotationY, args.FigHight, args.FigWidth,  args.Workdirpath,  args.htmlOutDir, args.htmlFname)






