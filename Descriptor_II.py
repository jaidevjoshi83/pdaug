from modlamp.descriptors import *
import os
import pandas as pd
 
def Descriptor_calcultor(DesType, inputfile, out_dir, ph, amide ):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(inputfile)
    list_pep_name = df[df.columns.tolist()[0]].tolist()
    desc = GlobalDescriptor(list_pep_name)

    if DesType == "Length":
        desc.length()
        df = desc.descriptor
        dfN = desc.featurenames

    elif DesType == "Weight":
        desc.calculate_MW(amide=amide)
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

    dfOut.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True,sep='\t')

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

    parser.add_argument("-o", "--OutDir",
                        required=None,
                        default=os.path.join(os.getcwd(),'OutDir'),
                        help="Path to out file")  
                          
    args = parser.parse_args()

    Descriptor_calcultor(args.DesType, args.InFile, args.OutDir, args.Ph, args.Amide)
