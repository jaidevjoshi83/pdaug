import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.metrics.aaindex import get_aa2volume, get_aa2hydropathy
from quantiprot.metrics.basic import average
from quantiprot.analysis.fisher import local_fisher_2d, _plot_local_fisher_2d
from matplotlib import pyplot as plt

def Run_fisher(Fasta1, Fasta2, windows_per_frame, overlap_factor, xlabel, ylabel, pop1_label, pop2_label, out_file_path, file_name):
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

    # Plot local Fisher:
    _plot_local_fisher_2d(result, xlabel=xlabel,
                                  ylabel=ylabel,
                                  pop1_label=pop1_label,
                                  pop2_label=pop2_label,
                                  out_file_path=out_file_path)
                                  #'/Users/joshij/Desktop/quantiprot/quantiprot-master/examples/jai.png')

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
      
    parser.add_argument("-f", "--frame_range",
                        required=False,
                        default=None,
                        help="Path to out file")          
                                               
    args = parser.parse_args()
    Run_fisher(args.Fasta1, args.Fasta2, args.windows_per_frame, args.overlap_factor, args.xlabel, args.ylabel, args.pop1_label, args.pop2_label, args.out_file_path, args.file_name)

