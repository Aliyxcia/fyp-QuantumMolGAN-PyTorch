import numpy as np
from rdkit import Chem
from sparse_molecular_dataset import SparseMolecularDataset

# filename = 'gdb11_size11.smi'
# length = 22796628
# lines = np.random.choice(length,50,replace=False)
# data = []
# for i,line in enumerate(open(filename, 'r').readlines()):
#     if i in lines:
#         data.append(line.split()[0])
# print(data)

mol_11 = ['CC(=O)C1(F)CC(O)(CF)C1', 'CC(F)=CC(=O)NNC(N)=N', 'NC1CC(C1)CC(F)(F)CF', 'CC1COC(C)(C=C1)C(F)=C', 'CN(C=N)N=C(NN)NC=O', 'CC(C)ON=C(CO)CC=O', 'CC(=O)C1=COC=CC(=N)N1', 'CC(C)C1(O)C2NN=C(C)C12', 'CC1NC2(C)C3CNC1C23N', 'CCCCC(C)ON=CCC', 'OCC(F)CCCOC=NO', 'NN1C(=N)C(=O)N=C1CC#C', 'CC1CC(CC=C)C(N)CN1', 'NC1(CN2CCNCC12)C#C', 'CC1(F)C2CC2C(CF)C1=C', 'CCN(C)c1nc(N)c(C)[nH]1', 'CC12CC3(CO1)OCCC3=C2', 'NN=C(CCO)C1CC(=O)O1', 'Nn1nc(O)cc1OC2CC2', 'CC1(N)CC12NC3CC3C2N', 'NC12C=COCC1=COC2=C', 'CC1=C(OC=CCC1)C(O)=O', 'N=CNC12CC=NNCC1C2', 'Oc1cc2c(O)cnn2nn1', 'FC12C3CC(C=CC13)C2C=O', 'CC1Cc2nonc2CC=C1', 'O=C1CC2CCOC2C(=O)C1', 'FC1CC2OC1C=COC2=O', 'COC(CC=CCC=C)=CC', 'CC1C(C)C(=C)C(C#N)N1C', 'CCN(N)C1=NN=C(C)CC1', 'NCC(F)(C1CO1)C2(O)CC2', 'NC1C2CC(C1F)C3(CO3)C2', 'CC1C2ON(C)C(=O)C1C2C', 'O=C1NC2CNC23CCCC13', 'Oc1noc2c(C=C)[nH]nc12', 'CC1C=CC2=C1C=CC2C#C', 'FC1C=C(OCC2NC12)C#N', 'CC12NC3C4CC(C3=C1)C24N', 'CCC1OC2CON=C2C1F', 'Nc1ocnc1C2OC2CO', 'ON=c1o[nH]c2CCN=Cc12', 'NCC#Cc1[nH]nc2CNc12', 'C1OCC2OC23CC=COC13', 'CC1CCC2N1c3cocc23', 'FCC1ON=C2OC3CC3C12', 'FCC1CCC2=C1C3OC3C2', 'O=C1Oc2[nH]ncc2N3CC13', 'C1OC1C2(CC2)n3cncn3', 'FC1CCC2C3CN3C1C2=O']

dataset = SparseMolecularDataset()
dataset.generate_from_list(mol_11)



