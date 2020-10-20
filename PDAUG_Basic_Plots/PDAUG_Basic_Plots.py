from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import glob, os, sys
import pandas as pd 
import plotly.express as px
###################################
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


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


def FragReturn(Seq, d):

    tokens = []
    for seq in Seq: 

        for l in range(d):
            if l < d:
                for x in range(int(len(seq)/d)):
                    s = (x*d)+l
                    e = s+d
                    if len(seq[s:e]) == d:
                        tokens.append(seq[s:e])
          
                    else:
                        pass
            else:
                pass
    return tokens

def PlotWordCloud(TokenList, OutFile):

    comment_words = '' 
    stopwords = set(STOPWORDS) 
    comment_words += " ".join(TokenList)+" "
      
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
                       
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.savefig(OutFile,dpi=600)


def HeatMapPlot(Infile,  IndexColumn, x_label, y_label,  Workdirpath, htmlOutDir, htmlFname):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(Infile, sep="\t")
    y_ticks = list(df[IndexColumn])

    fig = px.imshow(df[df.columns.tolist()[1:]], labels=dict(x=x_label, y=y_label), y=y_ticks)
    fig.update_xaxes(side="top")

    fig.write_html(os.path.join(Workdirpath, htmlOutDir, htmlFname))


def BoxPlot(InFile, Feature, label,  Workdirpath, htmlOutDir, htmlFname):

    Workdirpath = os.path.join(os.getcwd(),'report_dir')

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(InFile, sep="\t")
    fig = px.box(df, y=Feature, color=label, notched=True, title="Box plot of "+Feature )
    fig.write_html(os.path.join(Workdirpath, htmlOutDir, htmlFname))


def ScatterPlot(InFile, Feature1, Feature2, Feature3, Label, PlotType, Workdirpath, htmlOutDir, htmlFname):

    Workdirpath = os.path.join(os.getcwd(),'report_dir')

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(InFile, sep="\t")


    if PlotType == "3D":
        fig = px.scatter_3d(df, x=Feature1, y=Feature2, z=Feature3, color=Label)
        fig.write_html(os.path.join(Workdirpath, htmlOutDir, htmlFname))

    elif PlotType == "2D":
        fig = px.scatter(df, x=Feature1, y=Feature2, color=Label)
        fig.write_html(os.path.join(Workdirpath, htmlOutDir, htmlFname))


def WordCloudPlot(InFile, d, Workdirpath, htmlOutDir, htmlFname):

    Workdirpath = os.path.join(os.getcwd(),'report_dir')
    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    Peps,_ = ReturnPeptide(InFile)
    Frags = FragReturn(Peps, int(d))

    PlotWordCloud(Frags, "out.png")
    #HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))


if __name__=="__main__":


    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    HM = subparsers.add_parser('HeatMap')
    HM.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    HM.add_argument("-C",  "--IndexColumn", required=True, help="")
    HM.add_argument("-x", "--x_label", required=True, help="")
    HM.add_argument("-y","--y_label", required=True, help="")
    HM.add_argument("--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'), help="Path to html directory")
    HM.add_argument("--htmlFname", required=False, help="HTML out file", default="report.html")
    HM.add_argument("--Workdirpath", required=False, default=os.getcwd(), help="Path to working directory")

    BP = subparsers.add_parser('BoxPlot')
    BP.add_argument("-I", "--InFile", required=True, default=None, help="Input file")
    BP.add_argument("-F", "--Feature", required=True, default=None, help="Feature list")
    BP.add_argument("-O", "--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="Path to html dir")
    BP.add_argument("-Hf", "--htmlFname", required=False, help="HTML out file", default="report.html")
    BP.add_argument("-Wp", "--Workdirpath", required=False, default=os.getcwd(), help="Path to Working Directory")
    BP.add_argument("-L", "--Label", required=False, default=False, help="Working Directory Path")
    
    SP = subparsers.add_parser('ScatterPlot')
    SP.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    SP.add_argument("-F1", "--Feature1", required=True, default=True, help="Path to target tsv file")   
    SP.add_argument("-F2", "--Feature2", required=True, default=True, help="Roatate ticks")
    SP.add_argument("-F3", "--Feature3", required=False,  help="Roatate ticks")
    SP.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
    SP.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="jai.html")
    SP.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
    SP.add_argument("-T", "--PlotType", required=True,  help="")
    SP.add_argument("-L","--Label", required=False, default=False, help="Working Directory Path")

    WC = subparsers.add_parser('WordCloud')
    WC.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    WC.add_argument("-D", "--FragSize", required=True, default=None, help="Path to target tsv file")
    WC.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
    WC.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="report.html")
    WC.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")


    args = parser.parse_args()

    if sys.argv[1] == "HeatMap":
        HeatMapPlot(args.InFile,  args.IndexColumn, args.x_label, args.y_label, args.Workdirpath, args.htmlOutDir, args.htmlFname)
                    
    elif sys.argv[1] == "ScatterPlot":
        ScatterPlot(args.InFile, args.Feature1, args.Feature2, args.Feature3, args.Label, args.PlotType, args.Workdirpath,  args.htmlOutDir, args.htmlFname)

    elif sys.argv[1] == "BoxPlot":
        BoxPlot(args.InFile, args.Feature, args.Label,  args.Workdirpath,  args.htmlOutDir, args.htmlFname)

    elif sys.argv[1] == "WordCloud":
        WordCloudPlot(args.InFile, args.FragSize, args.Workdirpath,  args.htmlOutDir, args.htmlFname)   

    else:
        print("In Correct Option:")
