from modlamp.sequences import *
import argparse, sys
import pandas as pd 
import os

def Random_seq(seq_num, lenmin_s, lenmax_s, S_proba, OutFasta):

    b = Random(int(seq_num), int(lenmin_s),int(lenmax_s))
    b.generate_sequences(proba=float(S_proba))

    OutPep = b.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')

def Helices_seq(seq_num, lenmin_s, lenmax_s, OutFasta):

    h = Helices(int(seq_num), int(lenmin_s),int(lenmax_s))
    h.generate_sequences()

    OutPep = h.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')

 
def Kinked_seq(seq_num, lenmin_s, lenmax_s, OutFasta):

    k = Kinked(int(seq_num), int(lenmin_s),int(lenmax_s))
    k.generate_sequences()

    OutPep = k.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')


def Oblique_seq(seq_num, lenmin_s, lenmax_s, OutFasta):

    o = Oblique(int(seq_num), int(lenmin_s),int(lenmax_s))
    o.generate_sequences()
    o.sequences

    OutPep = o.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')


def Centrosymmetric_seq(seq_num, lenmin_s, lenmax_s, symmetry_s, OutFasta):

    s = Centrosymmetric(int(seq_num), int(lenmin_s),int(lenmax_s))
    s.generate_sequences(symmetry=symmetry_s)

    OutPep = s.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')


def HelicesACP_seq(seq_num, lenmin_s, lenmax_s, OutFasta):

    helACP = HelicesACP(int(seq_num), int(lenmin_s),int(lenmax_s))
    helACP.generate_sequences()

    OutPep = helACP.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')


def Hepahelices_seq(seq_num, lenmin_s, lenmax_s, OutFasta):

    h = Hepahelices(int(seq_num), int(lenmin_s),int(lenmax_s))  
    h.generate_sequences()

    OutPep = h.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')


def AMPngrams_seq(seq_num, lenmin_s, lenmax_s, OutFasta):

    s = AMPngrams(int(seq_num), int(lenmin_s),int(lenmax_s))
    s.generate_sequences()

    OutPep = s.sequences
    OutFasta = open(OutFasta, 'w')

    for i,O in enumerate(OutPep):
        OutFasta.write(">sequence_"+str(i)+'\n')
        OutFasta.write(O+'\n')

def AmphipathicArc_seq(seq_num, lenmin_s, lenmax_s, gen_seq, hyd_gra, OutFasta):

    amphi_hel = AmphipathicArc(int(seq_num), int(lenmin_s),int(lenmax_s))
    amphi_hel.generate_sequences(int(gen_seq))
    OutFasta = open(OutFasta, 'w')

    if hyd_gra == 'true':

        amphi_hel.make_H_gradient()
        OutPep = amphi_hel.sequences
                
        for i,O in enumerate(OutPep):
            OutFasta.write(">sequence_"+str(i)+'\n')
            OutFasta.write(O+'\n')

    elif hyd_gra == 'false':

        OutPep = amphi_hel.sequences
        
        for i,O in enumerate(OutPep):
            OutFasta.write(">sequence_"+str(i)+'\n')
            OutFasta.write(O+'\n')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Ran = subparsers.add_parser('Random')
    Ran.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Ran.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Ran.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Ran.add_argument("-p","--S_proba", required=False, default='AMP', help="AA probability to be used to generate sequences,'rand', 'AMP', 'AMPnoCM', 'randnoCM', 'ACP'")
    Ran.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    Hal = subparsers.add_parser('Helices')
    Hal.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Hal.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Hal.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Hal.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    Kin = subparsers.add_parser('Kinked')
    Kin.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Kin.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Kin.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Kin.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta") 

    Obl = subparsers.add_parser('Oblique')
    Obl.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Obl.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Obl.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Obl.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    Cen = subparsers.add_parser('Centrosymmetric')
    Cen.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Cen.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Cen.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Cen.add_argument("-S","--symmetry_s", required=False, default="asymmetric", help="symmetric,asymmetric")
    Cen.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    Hel = subparsers.add_parser('HelicesACP')
    Hel.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Hel.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Hel.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Hel.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    Hep = subparsers.add_parser('Hepahelices')
    Hep.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Hep.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Hep.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Hep.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    AMP = subparsers.add_parser('AMPngrams')
    AMP.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    AMP.add_argument("-m","--n_min", required=False, default=3, help="minimum number of ngrams to take for sequence assembly")
    AMP.add_argument("-M","--n_max", required=False, default=1, help="maximum number of ngrams to take for sequence assembly")
    AMP.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")
    
    Arc = subparsers.add_parser('AmphipathicArc')
    Arc.add_argument("-s","--seq_num", required=True, default=None, help="number of sequences to be generated")
    Arc.add_argument("-m","--lenmin_s", required=False, default=7, help="")
    Arc.add_argument("-M","--lenmax_s", required=False, default=20, help="Len max")
    Arc.add_argument("-a","--arcsize", help="Choose among 100, 140, 180, 220, 260, or choose mixed to generate a mixture")
    Arc.add_argument("-y","--hyd_gra", default='False', help="Method to mutate the generated sequences to have a hydrophobic gradient by substituting the last third of the sequence amino acids to hydrophobic.")
    Arc.add_argument("-O", "--OutFasta", required=True, default=None, help="Output Fasta")

    args = parser.parse_args()

    if sys.argv[1] == 'Random':
        Random_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.S_proba, args.OutFasta)
    elif sys.argv[1] == 'Helices':
        Helices_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.OutFasta)
    elif sys.argv[1] == 'Kinked':
        Kinked_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.OutFasta)
    elif sys.argv[1] == 'Oblique':
        Oblique_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.OutFasta)
    elif sys.argv[1] == 'Centrosymmetric':
        Centrosymmetric_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.symmetry_s, args.OutFasta)
    elif sys.argv[1] == 'HelicesACP':
        HelicesACP_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.OutFasta)
    elif sys.argv[1] == 'Hepahelices':
        Hepahelices_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.OutFasta)
    elif sys.argv[1] == 'AMPngrams':
        AMPngrams_seq(args.seq_num, args.n_min, args.n_max, args.OutFasta)
    elif sys.argv[1] == 'AmphipathicArc':
        AmphipathicArc_seq(int(args.seq_num), int(args.lenmin_s), int(args.lenmax_s), int(args.arcsize), args.hyd_gra, args.OutFasta)
    else:
        print("You entered Wrong Values: ")