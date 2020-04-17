from modlamp.descriptors import *
import pandas as pd
import os

def AutoCorrCal(InFile, window, ScaleName, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq = pd.read_csv(InFile)
    list_pep_name = seq[seq.columns.tolist()[0]].tolist()

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_autocorr(window)
    df = AMP.descriptor

    columns = ["CroAut_"+str(i) for i in range(len(df[0]))]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True,sep='\t')

def CrossCorrCal(InFile, window, ScaleName, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq = pd.read_csv(InFile)
    list_pep_name = seq[seq.columns.tolist()[0]].tolist()

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_crosscorr(window)
    df = AMP.descriptor

    columns = ["CroCor_"+str(i) for i in range(len(df[0]))]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True,sep='\t')

def CalculateMovementCal(InFile, window, angle, modality, ScaleName, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq = pd.read_csv(InFile)
    list_pep_name = seq[seq.columns.tolist()[0]].tolist()

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_moment(window, angle, modality)
    df = AMP.descriptor

    df = pd.DataFrame(df, columns=['Movement'])
    df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True,sep='\t')

def CalculateGlobalCal(InFile, window, modality, ScaleName, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq = pd.read_csv(InFile)
    list_pep_name = seq[seq.columns.tolist()[0]].tolist()

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_global(window, modality)
    df = AMP.descriptor

    df = pd.DataFrame(df, columns=['Global'])
    df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True, sep='\t')

def CalculateProfileCal(InFile, prof_type, window, ScaleName, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq = pd.read_csv(InFile)
    list_pep_name = seq[seq.columns.tolist()[0]].tolist()

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_profile(prof_type, window)
    df = AMP.descriptor

    df = pd.DataFrame(df, columns=['hyPhoPro','hyPhoMov'])
    df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True, sep='\t')

def CalculateArcCal(InFile, modality, out_dir): 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seq = pd.read_csv(InFile)
    list_pep_name = seq[seq.columns.tolist()[0]].tolist()

    AMP = PeptideDescriptor(list_pep_name, scalename="peparc")
    AMP.calculate_arc(modality)
    df =  AMP.descriptor

    columns = ["Arc_"+str(i) for i in range(len(df[0]))]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True, sep='\t')

#AutoCorrCal('jai.csv', 8, 'MSS', 'AutCor' )
#CrossCorrCal('jai.csv', 4, 'MSS', 'CroCor')
#CalculateMovementCal('jai.csv', 7, 100, 'max', 'kytedoolittle', 'Movement' )
#CalculateGlobalCal('jai.csv', 7, 'max', 'kytedoolittle', 'Global')
#CalculateProfileCal('jai.csv', 'H', 7, 'kytedoolittle', 'profile' )
#CalculateArcCal('jai.csv', 'max', "des")

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Aut = subparsers.add_parser('AutoCorrCal')
    Aut.add_argument("-i","--InFile", required="", default="", help="")
    Aut.add_argument("-w","--WindowSize", required="", default="", help="")
    Aut.add_argument("-s","--ScaleName", required="", default="", help="")
    Aut.add_argument("-o","--OutDir", required="", default="", help="")

    Cro = subparsers.add_parser('CrossCorrCal')
    Cro.add_argument("-i","--InFile", required="", default="", help="")
    Cro.add_argument("-w","--WindowSize", required="", default="", help="")
    Cro.add_argument("-s","--ScaleName", required="", default="", help="")
    Cro.add_argument("-o","--OutDir", required="", default="", help="")

    Mov = subparsers.add_parser('CalculateMovement')
    Mov.add_argument("-i","--InFile", required="", default="", help="")
    Mov.add_argument("-w","--WindowSize", required="", default="", help="")
    Mov.add_argument("-a","--Angle", required="", default="", help="")
    Mov.add_argument("-m","--Modality", required="", default="", help="")
    Mov.add_argument("-s","--ScaleName", required="", default="", help="")
    Mov.add_argument("-o","--OutDir", required="", default="", help="")

    Glo = subparsers.add_parser('GlobalCal')
    Glo.add_argument("-i","--InFile", required="", default="", help="")
    Glo.add_argument("-w","--WindowSize", required="", default="", help="")
    Glo.add_argument("-m","--Modality", required="", default="", help="")
    Glo.add_argument("-s","--ScaleName", required="", default="", help="")
    Glo.add_argument("-o","--OutDir", required="", default="", help="")

    Pro = subparsers.add_parser('ProfileCal')
    Pro.add_argument("-i","--InFile", required="", default="", help="")
    Pro.add_argument("-p","--ProfType", required="", default="", help="")
    Pro.add_argument("-w","--WindowSize", required="", default="", help="")
    Pro.add_argument("-s","--ScaleName", required="", default="", help="")
    Pro.add_argument("-o","--OutDir", required="", default="", help="")

    Arc = subparsers.add_parser('ArcCal')
    Arc.add_argument("-i","--InFile", required="", default="", help="")
    Arc.add_argument("-m","--Modality", required="", default="", help="")
    Arc.add_argument("-o","--OutDir", required="", default="", help="")

    args = parser.parse_args()

    if sys.argv[1] == 'AutoCorrCal':
        AutoCorrCal(args.InFile, args.window, args.ScaleName, args.out_dir)
    elif sys.argv[1] == 'CrossCorrCal':
        CrossCorrCal(args.InFile, args.window, args.ScaleName, args.out_dir)
    elif sys.argv[1] == 'CalculateMovement':
        CalculateMovementCal(args.InFile, args.window, args.angle, args.modality, args.ScaleName, args.out_dir)
    elif sys.argv[1] == 'GlobalCal':
        CalculateGlobalCal(args.InFile, args.window, args.modality, args.ScaleName, args.out_dir)
    elif sys.argv[1] == 'ProfileCal':
         CalculateProfileCal(args.InFile, args.prof_type, args.window, args.ScaleName, args.out_dir)
    elif sys.argv[1] == 'ArcCal':
        CalculateArcCal(args.InFile, args.modality, args.out_dir)
    else:
        print"You entered Wrong Values: "