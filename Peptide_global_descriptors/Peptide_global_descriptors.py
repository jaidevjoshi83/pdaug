from modlamp.descriptors import *
import os
import pandas as pd
 
def Descriptor_calcultor(DesType, inputfile, ph, amide,OutFile ):

    list_pep_name = []
    f = open(inputfile)
    lines = f.readlines()
    
    for line in lines:
        if ">" in line:
            pass
        else:
            list_pep_name.append(line.strip('\n'))


    desc = GlobalDescriptor(list_pep_name)

    if DesType == "Length":
        desc.length()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "Weight":

        desc.calculate_MW()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "Charge":
        desc.calculate_charge(ph=int(ph), amide=amide)
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "ChargeDensity":
        desc.charge_density(ph=int(ph), amide=amide)
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "IsoelectricPoint":
        desc.isoelectric_point()
        df = desc.descriptor
        dfN = desc.featurenames
 
    elif DesType == "InstabilityIndex":
        desc.instability_index()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "Aromaticity":
        desc.aromaticity()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "AliphaticIndex":
        desc.aliphatic_index()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "BomanIndex":
        desc.boman_index()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "HydrophobicRatio":
        desc.hydrophobic_ratio()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "All":
        desc.calculate_all(amide=amide)
        df = desc.descriptor
        dfN = desc.featurenames

    df = desc.descriptor
    dfN = desc.featurenames
    dfOut = pd.DataFrame(df,columns=dfN)

    dfOut.to_csv(OutFile, index=True,sep='\t')
    print (dfOut)

if __name__=="__main__":



    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--DesType",
                        required=True,
                        default=None,
                        help="Descriptors Type")
                        
    parser.add_argument("-i", "--InFile",
                        required=True,
                        default=None,
                        help="Peptide File")   

    parser.add_argument("-p", "--Ph",
                        required=False,
                        default=7.0,
                        help="Ph 0-14") 

    parser.add_argument("-a", "--Amide",
                        required=False,
                        default="True",
                        help="True or False")   

    parser.add_argument("-O", "--OutFile",
                        required=False,
                        default="out.tsv",
                        help="True or False") 

                          
    args = parser.parse_args()

    Descriptor_calcultor(args.DesType, args.InFile, args.Ph, args.Amide, args.OutFile)
