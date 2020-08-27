import matplotlib.pyplot as plt
import Bio
from Bio import SeqIO
import os


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

def LegnthDestribution(InFile, Workdirpath, htmlOutDir, htmlFname):

    Workdirpath = os.path.join(os.getcwd(),'report_dir')

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    sizes = [len(rec.seq) for rec in SeqIO.parse(InFile, "fasta")]
    print (len(sizes), min(sizes), max(sizes))

    plt.hist(sizes, bins=20)
    plt.title("%i Negative bacteriocin sequences\nLengths %i to %i" \
                % (len(sizes),min(sizes),max(sizes)))
    plt.xlabel("Sequence length (bp)")
    plt.ylabel("Count")
    plt.savefig('out.png')

    plt.savefig(os.path.join(Workdirpath, htmlOutDir, 'out.png'))
    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))


if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile", required=True, default=None, help="Path to target tsv file")
    parser.add_argument("-O","--htmlOutDir", required=False, default=os.path.join(os.getcwd(),'report_dir'),  help="HTML Out Dir")
    parser.add_argument("-Hf","--htmlFname", required=False, help="HTML out file", default="report.html")
    parser.add_argument("-Wp","--Workdirpath", required=False, default=os.getcwd(), help="Working Directory Path")
    args = parser.parse_args()

    LegnthDestribution(args.InFile, args.Workdirpath, args.htmlOutDir, args.htmlFname)
