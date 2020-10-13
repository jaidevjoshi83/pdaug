from modlamp.descriptors import *
import pandas as pd
import os

def AutoCorrCal(InFile, window, ScaleName, OutFile):


    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    list_pep_name = Peptides


    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_autocorr(int(window))
    df = AMP.descriptor

    columns = ["CroAut_"+str(i) for i in range(len(df[0]))]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(OutFile, index=False,sep='\t')

def CrossCorrCal(InFile, window, ScaleName, OutFile):

    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    list_pep_name = Peptides

    print (list_pep_name, ScaleName)

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_crosscorr(int(window))
    df = AMP.descriptor

    columns = ["CroCor_"+str(i) for i in range(len(df[0]))]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(OutFile, index=False,sep='\t')

def CalculateMovementCal(InFile, window, angle, modality, ScaleName, OutFile):


    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    list_pep_name = Peptides

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_moment(int(window), int(angle), modality)
    df = AMP.descriptor

    df = pd.DataFrame(df, columns=['Movement'])
    df.to_csv(OutFile, index=False,sep='\t')

def CalculateGlobalCal(InFile, WindowSize, modality, ScaleName, OutFile):


    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    list_pep_name = Peptides

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_global(int(WindowSize), modality)
    df = AMP.descriptor

    df = pd.DataFrame(df, columns=['Global'])
    df.to_csv(OutFile, index=False, sep='\t')

def CalculateProfileCal(InFile, prof_type, WindowSize, ScaleName, OutFile):


    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    list_pep_name = Peptides

    AMP = PeptideDescriptor(list_pep_name, ScaleName)
    AMP.calculate_profile(prof_type, int(WindowSize))
    df = AMP.descriptor

    df = pd.DataFrame(df, columns=['hyPhoPro','hyPhoMov'])
    df.to_csv(OutFile, index=False, sep='\t')

def CalculateArcCal(InFile, modality, OutFile): 

    file = open(args.InFile)
    lines = file.readlines()

    Index = []
    Peptides = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            Peptides.append(line.strip('\n'))
    
    list_pep_name = Peptides

    AMP = PeptideDescriptor(list_pep_name, scalename="peparc")
    AMP.calculate_arc(modality)
    df =  AMP.descriptor

    columns = ["Arc_"+str(i) for i in range(len(df[0]))]
    df = pd.DataFrame(df, columns=columns)
    df.to_csv(OutFile, index=False, sep='\t')

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    Aut = subparsers.add_parser('AutoCorrCal')
    Aut.add_argument("-i","--InFile", required=True, default=None, help="")
    Aut.add_argument("-w","--WindowSize", required=False, default=7, help="")
    Aut.add_argument("-s","--ScaleName", required=False, default="Eisenberg", help="")
    Aut.add_argument("-o","--OutFile", required=False, default="Descriptor.tsv", help="")

    Cro = subparsers.add_parser('CrossCorrCal')
    Cro.add_argument("-i","--InFile", required=True, default=None, help="")
    Cro.add_argument("-w","--WindowSize", required=False, default=7, help="")
    Cro.add_argument("-s","--ScaleName", required=False, default="Eisenberg", help="")
    Cro.add_argument("-o","--OutFile", required=False, default="Descriptor.tsv", help="")

    Mov = subparsers.add_parser('CalculateMovement')
    Mov.add_argument("-i","--InFile", required=True, default=None, help="")
    Mov.add_argument("-w","--WindowSize", required=False, default=1000, help="")
    Mov.add_argument("-a","--Angle", required=False, default=100, help="")
    Mov.add_argument("-m","--Modality", required=False, default="max", help="")
    Mov.add_argument("-s","--ScaleName", required=False, default="Eisenberg", help="")
    Mov.add_argument("-o","--OutFile", required=False, default="Descriptor.tsv", help="")

    Glo = subparsers.add_parser('GlobalCal')
    Glo.add_argument("-i","--InFile", required="", default="", help="")
    Glo.add_argument("-w","--WindowSize", required=False, default=1000, help="")
    Glo.add_argument("-m","--Modality", required=False, default="max", help="")
    Glo.add_argument("-s","--ScaleName", required=False, default="Eisenberg", help="")
    Glo.add_argument("-o","--OutFile", required=False, default="Descriptor.tsv", help="")

    Pro = subparsers.add_parser('ProfileCal')
    Pro.add_argument("-i","--InFile", required=True, default=None, help="")
    Pro.add_argument("-p","--ProfType", required=False, default="text", help="")
    Pro.add_argument("-w","--WindowSize", required=False, default=7, help="")
    Pro.add_argument("-s","--ScaleName", required=False, default="Eisenberg", help="")
    Pro.add_argument("-o","--OutFile", required=False, default="Descriptor.tsv", help="")

    Arc = subparsers.add_parser('ArcCal')
    Arc.add_argument("-i","--InFile", required=True, default=None, help="")
    Arc.add_argument("-m","--Modality", required=False, default="max", help="")
    Arc.add_argument("-o","--OutFile", required=False, default="Descriptor.tsv", help="")

    args = parser.parse_args()

    if sys.argv[1] == 'AutoCorrCal':
        AutoCorrCal(args.InFile, args.WindowSize, args.ScaleName, args.OutFile)
    elif sys.argv[1] == 'CrossCorrCal':
        CrossCorrCal(args.InFile, args.WindowSize, args.ScaleName, args.OutFile)
    elif sys.argv[1] == 'CalculateMovement':
        CalculateMovementCal(args.InFile, args.WindowSize, args.Angle, args.Modality, args.ScaleName, args.OutFile)
    elif sys.argv[1] == 'GlobalCal':
        CalculateGlobalCal(args.InFile, args.WindowSize, args.Modality, args.ScaleName, args.OutFile)
    elif sys.argv[1] == 'ProfileCal':
         CalculateProfileCal(args.InFile, args.ProfType, args.WindowSize, args.ScaleName, args.OutFile)
    elif sys.argv[1] == 'ArcCal':
        CalculateArcCal(args.InFile, args.Modality, args.OutFile)
    else:
        print ("You entered Wrong Values: ")