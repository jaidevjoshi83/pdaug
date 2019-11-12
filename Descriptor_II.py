from modlamp.descriptors import *
from modlamp.sequences import *
 



desc = GlobalDescriptor('AFDGHLKI')

"""
def Descriptor_calcultor(DesType, inputfile, out_dir):



	if DesType == "Length":

	elif DesType == "Formula":

	elif DesType == "Weight":

	elif DesType == "Charge":

	elif DesType == "ChargeDensity":

	elif DesType == "IsoelectricPoint":

	elif DesType == "InstabilityIndex":

	elif DesType == "Aromaticity":

	elif DesType == "AliphaticIndex":
    
    elif DesType == "BomanIndex":

    elif DesType == "HydrophobicRatio":

    elif DesType == "All":
"""




desc.length()

for a in desc.descriptor:
	print (a[0])


import pandas as pd 

desc = GlobalDescriptor(['AFDGHLKI','AFDGHLKI','AFDGHLKI','AFDGHLKI'])
desc.length()

df = pd.DataFrame(desc.descriptor)

print (df)
print (desc.descriptor)

desc = GlobalDescriptor('IAESFKGHIPL')
desc.calculate_MW(amide=True)
#print (desc.descriptor[0])
desc.calculate_charge(ph=7.0, amide=True)
#print ('CHARAGE',desc.descriptor[0])
desc.charge_density(ph=6, amide=True)
#print (desc.descriptor[0])
desc.isoelectric_point()
#print (desc.descriptor[0])
desc.instability_index()
#print (desc.descriptor[0])
desc.aromaticity()
#print (desc.descriptor[0])
desc.aliphatic_index()
#print (desc.descriptor[0])
desc.boman_index()
#print (desc.descriptor[0])
desc.hydrophobic_ratio()
#print (desc.descriptor[0])


desc.calculate_all(amide=True)
desc.featurenames
['Length', 'MW', 'ChargeDensity', 'pI', 'InstabilityInd', 'Aromaticity', 'AliphaticInd', 'BomanInd', 'HydRatio']
#print (desc.descriptor)

#desc.save_descriptor('/path/to/outputfile.csv')  # save the descriptor data (with feature names header)