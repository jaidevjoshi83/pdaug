import glob
import pandas as pd 
import sys


def MergeData(infiles, add_class_label, class_label, OutPut):

    data_frame = pd.DataFrame()

    if add_class_label == 'True' or add_class_label == 'true':
        for i, file in enumerate(infiles.split(',')): 
            df1 = pd.read_csv(file,sep='\t')
            df2 = pd.DataFrame(df1.shape[0]*[i], columns=[class_label])
            df3 =  pd.concat([df1,df2], axis=1)
            data_frame =  pd.concat([data_frame,df3])
        final_DF = data_frame.fillna(0)

    else:

        for file in infiles.split(','): 
            df1 = pd.read_csv(file,sep='\t')
            data_frame =  pd.concat([data_frame,df1])
        final_DF = data_frame.fillna(0)

    final_DF.to_csv(OutPut, sep="\t", index=False)


if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--infiles", required=True, default=None, help=".tsv")
    parser.add_argument("-L", "--add_class_label", required=False, default=False, help="Path to target tsv file")
    parser.add_argument("-C", "--class_label", required=False, default='class_label', help="Path to target tsv file")
    parser.add_argument("-O", "--OutPut", required=False, default='Out.tsv', help="Path to target tsv file")

    args = parser.parse_args()

    MergeData(args.infiles, args.add_class_label, args.class_label, args.OutPut)
