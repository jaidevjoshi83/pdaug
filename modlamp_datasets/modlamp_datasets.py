from modlamp.datasets import load_AMPvsTM
from modlamp.datasets import load_AMPvsUniProt
from modlamp.datasets import load_ACPvsTM
from modlamp.datasets import load_ACPvsRandom
import os
import pandas as pd

def DataGen(DataBaseType, OutDir):

    if not os.path.exists(str(OutDir)):
        os.makedirs(str(OutDir))

    if DataBaseType == 'AMPvsTM':
        data = load_AMPvsTM()

    elif DataBaseType == 'AMPvsUniProt':
        data = load_AMPvsUniProt()

    elif DataBaseType == 'ACPvsTM':
        data = load_ACPvsTM()

    elif DataBaseType == 'ACPvsRandom':
        data = load_ACPvsRandom()
    else:
        print "Enter Correct Values"
        exit()

    Target = data.target.tolist()
    df = data.sequences

    Target = pd.DataFrame(Target, columns=['Target'])
    df = pd.DataFrame(df)
    df = pd.concat([df, Target], axis=1)

    df.to_csv(os.path.join(OutDir,'pep_des.tsv'), index=True,sep='\t')

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--DataBaseType",
                        required=True,
                        default=None,
                        help="pep file")
                        
    parser.add_argument("-o", "--OutDir",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")   

    args = parser.parse_args()
    DataGen(args.DataBaseType, args.OutDir)
