import glob
import pandas as pd 
import sys
import os
import argparse

from modlamp.wetlab import CD

parser = argparse.ArgumentParser(description='Deployment tool')
subparsers = parser.add_subparsers()

calc_ellipticity = subparsers.add_parser('calc_ellipticity')
calc_ellipticity.add_argument("-T","--Type", required=True, default=None, help="Input fasta sequence")
calc_ellipticity.add_argument("-H","--DirPath", required=False, default=os.getcwd(), help="Input fasta sequence")
calc_ellipticity.add_argument("-Wn","--WMin", required=True, default=None, help="Number of mutations to perform per sequence")
calc_ellipticity.add_argument("-Wx","--Wmax", required=True, default=None, help="Probability of mutating a sequence")
calc_ellipticity.add_argument("-A","--Amide", required=True, default=None, help="Mutated output fasta")
calc_ellipticity.add_argument("-P","--Pathlen", required=True, default=None, help="Mutated output fasta")
calc_ellipticity.add_argument("-O","--OutPut", required=False, default="OutFile.tsv", help="Mutated output fasta")

PlotData = subparsers.add_parser('PlotData')

PlotData.add_argument("-H","--DirPath", required=False, default=os.getcwd(), help="Input fasta sequence")
PlotData.add_argument("-T","--Type", required=True, default=None, help="Input fasta sequence")
PlotData.add_argument("-Wn","--WMin", required=True, default=None, help="Number of mutations to perform per sequence")
PlotData.add_argument("-Wx","--Wmax", required=True, default=None, help="Probability of mutating a sequence")
PlotData.add_argument("-A","--Amide", required=True, default=None, help="Mutated output fasta")
PlotData.add_argument("-P","--Pathlen", required=True, default=None, help="Mutated output fasta")

Dichroweb = subparsers.add_parser("Dichroweb")
Dichroweb.add_argument("-H","--DirPath", required=False, default=os.getcwd(), help="Input fasta sequence")
Dichroweb.add_argument("-T","--Type", required=True, default=None, help="Input fasta sequence")
Dichroweb.add_argument("-Wn","--WMin", required=True, default=None, help="Number of mutations to perform per sequence")
Dichroweb.add_argument("-Wx","--Wmax", required=True, default=None, help="Probability of mutating a sequence")
Dichroweb.add_argument("-A","--Amide", required=True, default=None, help="Mutated output fasta")
Dichroweb.add_argument("-P","--Pathlen", required=True, default=None, help="Mutated output fasta")

helicity = subparsers.add_parser('helicity')

helicity.add_argument("-H","--DirPath", required=False, default=os.getcwd(), help="Input fasta sequence")
helicity.add_argument("-Wn","--WMin", required=True, default=None, help="Number of mutations to perform per sequence")
helicity.add_argument("-Wx","--Wmax", required=True, default=None, help="Probability of mutating a sequence")
helicity.add_argument("-A","--Amide", required=True, default=None, help="Mutated output fasta")
helicity.add_argument("-P","--Pathlen", required=True, default=None, help="Mutated output fasta")
helicity.add_argument("-t","--temperature", required=False, default=24.0, help="")
helicity.add_argument("-k","--k", required=True, default=2.4, help="")
helicity.add_argument("-I","--Induction", required=False, default=True, help="")
helicity.add_argument("-O","--OutPut", required=False, default="result.tsv", help="")


args = parser.parse_args()

if sys.argv[1] == "calc_ellipticity":

    if args.Type == "calc_molar_ellipticity":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_molar_ellipticity()
        df = cd.molar_ellipticity
        df = pd.DataFrame(df[0])
        df.to_csv(args.OutPut, index=None, sep="\t")

    elif args.Type == "calc_meanres_ellipticity":
        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_meanres_ellipticity()
        df = cd.meanres_ellipticity
        df = pd.DataFrame(df[0])
        df.to_csv(args.OutPut, index=None, sep="\t")
    else:
        pass

if sys.argv[1] == "PlotData":

    if args.Type == "mean residue ellipticity":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_meanres_ellipticity()
        cd.plot(data="mean residue ellipticity", combine='solvent')

    elif args.Type == "molar ellipticity":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_molar_ellipticity()
        cd.plot(data="molar ellipticity", combine='solvent')

    elif args.Type == "circular dichroism":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_molar_ellipticity()
        cd.plot(data="circular dichroism", combine='solvent')

    else:
        pass

if sys.argv[1] == "Dichroweb":

    if args.Type == "mean residue ellipticity":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_meanres_ellipticity()
        cd.dichroweb(data="mean residue ellipticity")

    elif args.Type == "molar ellipticity":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_molar_ellipticity()
        cd.dichroweb(data='molar ellipticity')

    elif args.Type == "circular dichroism":

        cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))
        cd.calc_molar_ellipticity()
        cd.dichroweb(data='circular dichroism')


if sys.argv[1] == "helicity":
    cd = CD(args.DirPath, wmin=int(args.WMin), wmax=int(args.Wmax), amide=args.Amide, pathlen=float(args.Pathlen))    
    cd.calc_meanres_ellipticity()
    cd.helicity(temperature=float(args.temperature), k=float(args.k), induction=args.Induction, filename=args.OutPut )
