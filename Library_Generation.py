from modlamp.sequences import *
import argparse, sys
import pandas as pd 

def Random_seq(seq_num, lenmin_s, lenmax_s, S_proba):
    b = Random(seq_num, lenmin_s,lenmax_s)
    b.generate_sequences(proba=S_proba)
    df = pd.DataFrame(b.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def Helices_seq(seq_num, lenmin_s, lenmax_s):
    h = Helices(seq_num, lenmin_s, lenmax_s)
    h.generate_sequences()
    df = pd.DataFrame(h.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')
 
def Kinked_seq(seq_num, lenmin_s, lenmax_s):
    k = Kinked(seq_num, lenmin_s, lenmax_s)
    k.generate_sequences()
    k.sequences
    df = pd.DataFrame(k.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def Oblique_seq(seq_num, lenmin_s, lenmax_s):
    o = Oblique(seq_num, lenmin_s, lenmax_s)
    o.generate_sequences()
    o.sequences
    df = pd.DataFrame(o.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def Centrosymmetric_seq(seq_num, lenmin_s, lenmax_s, symmetry_s):
    s = Centrosymmetric(seq_num, lenmin_s, lenmax_s)
    s.generate_sequences(symmetry=symmetry_s)
    df = pd.DataFrame(s.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def HelicesACP_seq(seq_num, lenmin_s, lenmax_s):
    helACP = HelicesACP(seq_num, lenmin_s, lenmax_s)
    helACP.generate_sequences()
    df = pd.DataFrame(helACP.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def Hepahelices_seq(seq_num, lenmin_s, lenmax_s):
    h = Hepahelices(seq_num, lenmin_s, lenmax_s)  
    h.generate_sequences()
    df = pd.DataFrame(h.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def AMPngrams_seq(seq_num, lenmin_s, lenmax_s):
    s = AMPngrams(seq_num, lenmin_s, lenmax_s)
    s.generate_sequences()
    df = pd.DataFrame(s.sequences, columns=['seq'])
    df.to_csv('Peptide_library.tsv', sep='\t')

def AmphipathicArc_seq(seq_num, lenmin_s, lenmax_s, gen_seq, hyd_gra):

    amphi_hel = AmphipathicArc(seq_num, lenmin_s, lenmax_s)
    amphi_hel.generate_sequences(gen_seq)

    if hyd_gra == 'True':
        df = pd.DataFrame(amphi_hel.sequences, columns=['seq'])
        df.to_csv('Peptide_library.tsv', sep='\t')
    elif hyd_gra == 'False':
        amphi_hel.make_H_gradient()
        df = pd.DataFrame(amphi_hel.sequences, columns=['seq'])
        df.to_csv('Peptide_library.tsv', sep='\t')

if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Ran = subparsers.add_parser('Random')
    Ran.add_argument("-s","--seq_num")
    Ran.add_argument("-m","--lenmin_s")
    Ran.add_argument("-M","--lenmax_s")
    Ran.add_argument("-p","--S_proba", help="'rand', 'AMP', 'AMPnoCM', 'randnoCM', 'ACP'")

    Hal = subparsers.add_parser('Helices')
    Hal.add_argument("-s","--seq_num")
    Hal.add_argument("-m","--lenmin_s")
    Hal.add_argument("-M","--lenmax_s")

    Kin = subparsers.add_parser('Kinked')
    Kin.add_argument("-s","--seq_num")
    Kin.add_argument("-m","--lenmin_s")
    Kin.add_argument("-M","--lenmax_s")

    Obl = subparsers.add_parser('Oblique')
    Obl.add_argument("-s","--seq_num")
    Obl.add_argument("-m","--lenmin_s")
    Obl.add_argument("-M","--lenmax_s")

    Cen = subparsers.add_parser('Centrosymmetric')
    Cen.add_argument("-s","--seq_num")
    Cen.add_argument("-m","--lenmin_s")
    Cen.add_argument("-M","--lenmax_s")
    Cen.add_argument("-S","--symmetry_s", help="symmetric,asymmetric")

    Hel = subparsers.add_parser('HelicesACP')
    Hel.add_argument("-s","--seq_num")
    Hel.add_argument("-m","--lenmin_s")
    Hel.add_argument("-M","--lenmax_s")

    Hep = subparsers.add_parser('Hepahelices')
    Hep.add_argument("-s","--seq_num")
    Hep.add_argument("-m","--lenmin_s")
    Hep.add_argument("-M","--lenmax_s")

    AMP = subparsers.add_parser('AMPngrams')
    AMP.add_argument("-s","--seq_num")
    AMP.add_argument("-m","--lenmin_s")
    AMP.add_argument("-M","--lenmax_s")

    Arc = subparsers.add_parser('AmphipathicArc')
    Arc.add_argument("-s","--seq_num")
    Arc.add_argument("-m","--lenmin_s")
    Arc.add_argument("-M","--lenmax_s")
    Arc.add_argument("-a","--arcsize", help="Choose among 100, 140, 180, 220, 260, or choose mixed to generate a mixture")
    Arc.add_argument("-y","--hyd_gra", default='False', help="Method to mutate the generated sequences to have a hydrophobic gradient by substituting the last third of the sequence amino acids to hydrophobic.")

    args = parser.parse_args()

    if sys.argv[1] == 'Random':

        Random_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.S_proba)

    elif sys.argv[1] == 'Helices':

        Helices_seq(args.seq_num, args.lenmin_s, args.lenmax_s)

    elif sys.argv[1] == 'Kinked':
        Kinked_seq(args.seq_num, args.lenmin_s, args.lenmax_s)

    elif sys.argv[1] == 'Oblique':
        Oblique_seq(args.seq_num, args.lenmin_s, args.lenmax_s)

    elif sys.argv[1] == 'Centrosymmetric':
        Centrosymmetric_seq(args.seq_num, args.lenmin_s, args.lenmax_s, args.symmetry_s)

    elif sys.argv[1] == 'HelicesACP':
        HelicesACP_seq(args.seq_num, args.lenmin_s, args.lenmax_s)

    elif sys.argv[1] == 'Hepahelices':
        Hepahelices_seq(args.seq_num, args.lenmin_s, args.lenmax_s)

    elif sys.argv[1] == 'AMPngrams':
        AMPngrams_seq(args.seq_num, args.lenmin_s, args.lenmax_s)

    elif sys.argv[1] == 'AmphipathicArc':

        AmphipathicArc_seq(int(args.seq_num), int(args.lenmin_s), int(args.lenmax_s), int(args.arcsize), args.hyd_gra)
    else:
        print"You entered Wrong Values: "

