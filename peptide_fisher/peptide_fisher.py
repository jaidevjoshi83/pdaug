import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aa2volume, get_aa2hydropathy
from quantiprot.metrics.basic import average
from quantiprot.analysis.fisher import local_fisher_2d, _plot_local_fisher_2d
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
      <h1> Machine Learning Algorithm Assessment Report </h1>
    </div>
    <div class="container">
      <div class="row">
        <div class="col-sm-4">
          <img src="1.png" alt="Smiley face" height="500" width="400">
        </div>

      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()

def Run_fisher(Fasta1, Fasta2, windows_per_frame, overlap_factor, xlabel, ylabel, pop1_label, pop2_label, out_file_path, file_name):


    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(Infile, sep="\t")

    # Load sets of amyloidogenic and non-amyloidogenic peptides:
    amyload_pos_seq = load_fasta_file(Fasta1)
    amyload_neg_seq = load_fasta_file(Fasta2)

    #print Feature(get_aa2volume())
    # Calculate quantitive features: volume and hydropathy
    mean_volume = Feature(get_aa2volume()).then(average)
    mean_hydropathy = Feature(get_aa2hydropathy()).then(average)

    #print mean_volume
    fs = FeatureSet("volume'n'hydropathy")

    fs.add(mean_volume)
    fs.add(mean_hydropathy)

    amyload_pos_conv_seq = fs(amyload_pos_seq)
    amyload_neg_conv_seq = fs(amyload_neg_seq)

    # Do local Fisher:
    result = local_fisher_2d(amyload_pos_conv_seq, amyload_neg_conv_seq,
                             windows_per_frame=windows_per_frame, overlap_factor=overlap_factor)

    out_file_path = os.path.join(Workdirpath, htmlOutDir, "1.png")
    # Plot local Fisher:
    _plot_local_fisher_2d(result, xlabel=xlabel,
                                  ylabel=ylabel,
                                  pop1_label=pop1_label,
                                  pop2_label=pop2_label,
                                  out_file_path=out_file_path)
                            

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

    parser.add_argument("-o", "--overlap_factor",
                        required=False,
                        default=1,
                        help="Path to out file")  

    parser.add_argument("-w", "--windows_per_frame",
                        required=False,
                        default=10,
                        help="Path to out file")  

    parser.add_argument("-x", "--xlabel",
                        required=True,
                        default=None,
                        help="Path to out file")  

    parser.add_argument("-y", "--ylabel",
                        required=True,
                        default=None,
                        help="Path to out file")  

    parser.add_argument("-p1", "--pop1_label",
                        required=True,
                        default=None,
                        help="Path to out file")  

    parser.add_argument("-p2", "--pop2_label",
                        required=True,
                        default=None,
                        help="Path to out file")   

    parser.add_argument("-O", "--out_file_path",
                        required=True,
                        default=None,
                        help="Path to out file")

    parser.add_argument("--htmlOutDir", 
                        required=False, 
                        default=os.path.join(os.getcwd(),'report_dir'), 
                        help="HTML Out Dir")

    parser.add_argument("--htmlFname", 
                        required=False, 
                        help="HTML out file", 
                        default="jai.html")

    parser.add_argument("--Workdirpath", 
                        required=False, 
                        default=os.getcwd(), 
                        help="Working Directory Path")
      
         
                                               
    args = parser.parse_args()
    Run_fisher(args.Fasta1, args.Fasta2, args.windows_per_frame, args.overlap_factor, args.xlabel, args.ylabel, args.pop1_label, args.pop2_label, args.out_file_path, args.file_name)

