import tempfile
import pandas as pd
import shutil
import sys
import glob
import fragbuilder
import openbabel
from fragbuilder import peptide

import os


def read_pep_file(pep_infile):
    
    file = open(pep_infile)
    lines = file.readlines()

    Index = []
    list_pep_name = []

    for line in lines:
        if '>' in line:
            Index.append(line.strip('\n'))
        else:
            line = line.strip('\n')
            line = line.strip('\r')
            list_pep_name.append(line.strip('\n'))

    return list_pep_name

def structure_gen(pep_seq, out_dir):

    if not os.path.exists(os.path.join(out_dir, 'DataFile')):
        os.makedirs(os.path.join(out_dir, 'DataFile'))

    b = len(pep_seq)

    for seq in pep_seq:

        pep = peptide.Peptide(seq, nterm = "charged", cterm = "neutral")
        pep.regularize()
        pep.write_pdb(os.path.join(out_dir, 'DataFile', seq+".pdb"))

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "sdf")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol,  os.path.join(out_dir, 'DataFile', seq+".pdb")) 

        mol.AddHydrogens()       

    
def main_process(str_pep_file, out_dir):
      
    my_pep = read_pep_file(str_pep_file) 
    structure_gen(my_pep, out_dir)
    
    
if __name__=="__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--pep", required=True, default=None, help="pep file")                    
    parser.add_argument("-o", "--OutDir", required=None, default=os.getcwd(), help="Path to out file")  
                                               
    args = parser.parse_args()
    main_process(args.pep, args.OutDir)
   
    
    
   
    
