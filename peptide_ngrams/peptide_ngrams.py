import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import quantiprot 
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aa2hydropathy
from quantiprot.metrics.basic import identity
from quantiprot.metrics.ngram import pattern_match, pattern_count
from quantiprot.analysis.ngram import ngram_count
from quantiprot.analysis.ngram import zipf_law_fit
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
      <h1> ngrams analysis </h1>
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


def Run_ngrams(fasta1, fasta2, htmlOutDir, htmlFname, Workdirpath ):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    alphasyn_seq = load_fasta_file(fasta1)
    amyload_pos_seq = load_fasta_file(fasta2)

    fs_aa = FeatureSet("aa patterns")
    fs_aa.add(identity)
    fs_aa.add(pattern_match, pattern='VT', padded=True)
    fs_aa.add(pattern_count, pattern='VT')

    result_seq = fs_aa(alphasyn_seq)

    for seq in result_seq[:3]:
        print (seq)

    fs_hp = FeatureSet("hydropathy patterns")
    fs_hp.add(Feature(get_aa2hydropathy()))
    fs_hp.add(Feature(get_aa2hydropathy()).then(pattern_match, pattern=[0.0, 2.0],
                                                metric='taxi', radius=1.0))
    result_seq2 = fs_hp(alphasyn_seq)
    result_freq = ngram_count(alphasyn_seq, n=2)
    result_fit = zipf_law_fit(amyload_pos_seq, n=3, verbose=True)

    counts = sorted(result_fit["ngram_counts"], reverse=True)
    ranks = range(1, len(counts)+1)

    slope = result_fit["slope"]
    harmonic_num = sum([rank**-slope for rank in ranks])
    fitted_counts = [(rank**-slope) / harmonic_num * sum(counts) for rank in ranks]

    plt.plot(ranks, counts, 'k', label="empirical")
    plt.plot(ranks, fitted_counts, 'k--',
             label="Zipf's law\nslope: {:.2f}".format((slope)))
    plt.xlabel('rank')
    plt.ylabel('count')
    plt.xscale('log')
    plt.yscale('log')
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
                        default="jai.html")

    parser.add_argument("--Workdirpath", 
                        required=False, 
                        default=os.getcwd(), 
                        help="Working Directory Path")

    args = parser.parse_args()        
                                               
    Run_ngrams(args.Fasta1, args.Fasta2, args.htmlOutDir, args.htmlFname, args.Workdirpath)

