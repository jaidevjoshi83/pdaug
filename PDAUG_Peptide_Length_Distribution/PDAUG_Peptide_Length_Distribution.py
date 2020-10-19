import matplotlib.pyplot as plt
import Bio
from Bio import SeqIO
import os


def LegnthDestribution(InFile, OutFile):


    sizes = [len(rec.seq) for rec in SeqIO.parse(InFile, "fasta")]

    plt.hist(sizes, bins=20)
    plt.title("%i Negative bacteriocin sequences\nLengths %i to %i" \
                % (len(sizes),min(sizes),max(sizes)))
    plt.xlabel("Sequence length (bp)")
    plt.ylabel("Count")

    plt.savefig(OutFile)



if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--InFile", required=True, default=None, help="Input file name")
    parser.add_argument("-O", "--OutFile", required=False, default="Out.png", help="Input file name")
    args = parser.parse_args()
    LegnthDestribution(args.InFile, args.OutFile)
