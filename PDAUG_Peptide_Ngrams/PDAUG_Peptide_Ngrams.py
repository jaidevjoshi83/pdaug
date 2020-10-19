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


def Run_ngrams(fasta1, fasta2, OutFile ):

    alphasyn_seq = load_fasta_file(fasta1)
    amyload_pos_seq = load_fasta_file(fasta2)

    fs_aa = FeatureSet("aa patterns")
    fs_aa.add(identity)
    fs_aa.add(pattern_match, pattern='VT', padded=True)
    fs_aa.add(pattern_count, pattern='VT')

    result_seq = fs_aa(alphasyn_seq)

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

    plt.savefig(OutFile)

if __name__=="__main__":
    
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f1", "--Fasta1",
                        required=True,
                        default=None,
                        help="First fasta file")
                        
    parser.add_argument("-f2", "--Fasta2",
                        required=True,
                        default=None,
                        help="Second fasta file")   


    parser.add_argument("--OutFile", 
                        required=True, 
                        help="HTML out file", 
                        default="report.html")


    args = parser.parse_args()        
                                               
    Run_ngrams(args.Fasta1, args.Fasta2, args.OutFile)

