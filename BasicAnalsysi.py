import  modlamp
from modlamp import  analysis



from modlamp.sequences import HelicesACP
helACP = HelicesACP(100, 7, 18)
helACP.generate_sequences()

print()

g = analysis.GlobalAnalysis(helACP.sequences, names=['Library1']) 
g.plot_summary()

