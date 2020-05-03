from modlamp.datasets import load_AMPvsTM
from modlamp.datasets import load_AMPvsUniProt
from modlamp.datasets import load_ACPvsTM
from modlamp.datasets import load_ACPvsRandom
import os
import pandas as pd

def DataGen(DataBaseType, OutFile):

    if DataBaseType == 'AMPvsTM':
        data = load_AMPvsTM()

    elif DataBaseType == 'AMPvsUniProt':
        data = load_AMPvsUniProt()

    elif DataBaseType == 'ACPvsTM':
        data = load_ACPvsTM()

    elif DataBaseType == 'ACPvsRandom':
        data = load_ACPvsRandom()
    else:
        print ("Enter Correct Values")
        exit()

    Target = data.target.tolist()
    Target_list = set(Target)
    df = data.sequences


    Target = pd.DataFrame(Target, columns=['Target'])
    df = pd.DataFrame(df, columns=['Peptide'])
    
    df = pd.DataFrame(df)
    df = pd.concat([df, Target], axis=1)

    df.to_csv(OutFile, index=False, sep='\t')


if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--DataBaseType",
                        required=True,
                        default=None,
                        help="pep file")
                        
    parser.add_argument("-o", "--OutFile",
                        required=False,
                        default='Out.tsv',
                        help="out put file name for str Descriptors")   

    args = parser.parse_args()
    DataGen(args.DataBaseType, args.OutFile)
