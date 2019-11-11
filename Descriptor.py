import pandas as pd
from pydpi.pypro import PyPro
import os


def Decriptor_generator(inpfile, DesType, out_dir):


    if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    df = pd.read_csv(inpfile)

    list_pep_name = df[df.columns.tolist()[0]].tolist()

    #print list_pep_name

    out_df = pd.DataFrame()

    for p in list_pep_name:

        protein = PyPro()
        protein.ReadProteinSequence(p)

        if DesType == 'PAAC':
            DS = protein.GetPAAC(lamda=5,weight=0.5)
        elif DesType == 'APAAC':
            DS = protein.GetAPAAC(lamda=5,weight=0.5)
        elif DesType == 'CTD':
            DS = protein.GetCTD()
        elif DesType == 'DPComp':
            DS = protein.GetDPComp()
        elif DesType == 'GearyAuto':
            DS = protein.GetGearyAuto()
        elif DesType == 'MoranAuto':
            DS = protein.GetMoranAuto()
        elif DesType == 'MoreauBrotoAuto':
            DS = protein.GetMoreauBrotoAuto()
        elif DesType == 'QSO':
            DS = protein.GetQSO()
        elif DesType == 'SOCN':
            DS = protein.GetSOCN()
        elif DesType == 'TPComp':
            DS = protein.GetTPComp()
        elif DesType == 'All':
            DS_1 = protein.GetPAAC(lamda=5,weight=0.5)
            DS_2 = protein.GetAPAAC(lamda=5,weight=0.5)
            DS_3 = protein.GetCTD()
            DS_4 = protein.GetDPComp()
            DS_5 = protein.GetGearyAuto()
            DS_6 = protein.GetMoranAuto()
            DS_7 = protein.GetMoreauBrotoAuto()
            DS_8 = protein.GetQSO()
            DS_9 = protein.GetSOCN()
            DS_10 = protein.GetTPComp()

            DS = {}

            for D in (DS_1, DS_2, DS_3, DS_4, DS_5, DS_6, DS_7, DS_8, DS_9, DS_10):
                DS.update(D)
        else:

            print "Enter correct"
            pass

        df  = pd.DataFrame(DS, index=[0])
        out_df = pd.concat([out_df, df], axis=0)

    #print out_df
    out_df.to_csv(os.path.join(out_dir,'pep_des.tsv'), index=True,sep='\t')


if __name__=="__main__":
    
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--pep",
                        required=True,
                        default=None,
                        help="pep file")
                        
    parser.add_argument("-t", "--DesType",
                        required=True,
                        default=None,
                        help="out put file name for str Descriptors")   

    parser.add_argument("-o", "--OutDir",
                        required=None,
                        default=os.path.join(os.getcwd(),'OutDir'),
                        help="Path to out file")  

                                               
    args = parser.parse_args()
  

    Decriptor_generator(args.pep, args.DesType, args.OutDir)

    #Decriptor_generator('jai.csv', 'PAAC')