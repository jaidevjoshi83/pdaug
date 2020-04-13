import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.utils.sequence import compact
from quantiprot.metrics.aaindex import get_aa2charge, get_aa2hydropathy
from quantiprot.metrics.basic import average, average_absolute
from matplotlib import pyplot as plt

def Run_Uverskey(Fasta1, Fasta2, out_file_path, Out_file_name):
    amyload_seq = load_fasta_file("data/Amyload_positive.fasta")
    disprot_seq = load_fasta_file("data/Disprot.fasta")

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
    plt.savefig()


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

    parser.add_argument("-p", "--out_file_path",
                        required=False,
                        default=1,
                        help="Path to out file")  

    parser.add_argument("-n", "--Out_file_name",
                        required=False,
                        default=10,
                        help="Path to out file")  
                                       
    args = parser.parse_args()
    
    Run_Uverskey(args.Fasta1, args.Fasta2, args.out_file_path, args.Out_file_name)
