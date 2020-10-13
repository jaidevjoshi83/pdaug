import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.utils.sequence import compact
from quantiprot.metrics.aaindex import get_aa2charge, get_aa2hydropathy
from quantiprot.metrics.basic import average, average_absolute
from matplotlib import pyplot as plt




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
      <h1> Uversky plot </h1>
    </div>
    <div class="container">
      <div class="row">
        <div class="col-sm-4">
          <img src="1.png" alt="Smiley face" height="800" width="800">
        </div>

      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()

def Run_Uverskey(Fasta1, Fasta2, htmlOutDir, htmlFname, Workdirpath):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    amyload_seq = load_fasta_file(Fasta1)
    disprot_seq = load_fasta_file(Fasta2)

    # Non-standard letters in Disprot assigned neutral charge and hydropathy:
    net_abs_charge = Feature(get_aa2charge(default=0)).then(average_absolute)
    mean_hydropathy = Feature(get_aa2hydropathy(default=0)).then(average)

    uversky_fs = FeatureSet("uversky")
    uversky_fs.add(mean_hydropathy, name="mean_hydropathy")
    uversky_fs.add(net_abs_charge, name="net_abs_charge")

    amyload_uversky_seq = uversky_fs(amyload_seq)
    disprot_uversky_seq = uversky_fs(disprot_seq)

    # First approach to get hydrophobicity/charge pairs
    amyload_data_x = amyload_uversky_seq.columns(feature="mean_hydropathy")[0]
    amyload_data_y = amyload_uversky_seq.columns(feature="net_abs_charge")[0]
    plt.plot(amyload_data_x, amyload_data_y,'.', label="Amyload")

    # Second approach to get hydrophobicity/charge pairs
    disprot_data = compact(disprot_uversky_seq).columns()
    plt.plot(disprot_data[0], disprot_data[1],'.', label="Disprot")

    plt.plot([-0.78, 0.835], [0.0, 0.5],'k')
    plt.xlabel("mean hydrophobicity")
    plt.ylabel("net abs charge")
    plt.legend()

    plt.savefig(os.path.join(Workdirpath, htmlOutDir, "1.png"))
    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))


if __name__=="__main__":
    
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f1", "--Fasta1",
                       required=True,
                        default=None,
                        help="pep file")
                        
    parser.add_argument("-f2", "--Fasta2",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")   


    parser.add_argument("--htmlOutDir", 
                        required=False, 
                        default=os.path.join(os.getcwd(),'report_dir'), 
                        help="HTML Out Dir")

    parser.add_argument("--htmlFname", 
                        required=False, 
                        help="HTML out file", 
                        default="out.html")

    parser.add_argument("--Workdirpath", 
                        required=False, 
                        default=os.getcwd(), 
                        help="Working Directory Path")
                                       
    args = parser.parse_args()
    
    Run_Uverskey(args.Fasta1, args.Fasta2, args.htmlOutDir, args.htmlFname, args.Workdirpath)
