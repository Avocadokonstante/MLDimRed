import os
import sys
import numpy as np
import dataprep_utils as dpu
#import pyanitools as pya
#import scipy as sc
#import matplotlib.pyplot as plt
#import turbomole_functions as tm
#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem import Draw
import time

# ---- Constants
HToEV = 27.21138505
AToBohr = 1.889725989
hbar_eVs = 6.582e-16  # eV
e = 1.6e-19  # C
m_to_A = 1e-10
c = 2.998e8  # m/s
NA = 6.022e23  # g/mol
elementmasses = {"C": 12.01115,
                 "H": 1.00797,
                 "O": 15.99940,
                 "N": 14.007
                 }

# ---- Table from ANI paper
# 1-8 is Number of atoms
# Ts is Max Temperature to randomly perturb the molecule along the normal modes
# Ss Number of data points generated per degree of freedom

Ts = {1: 2000.0, 2: 1500.0, 3: 1000.0, 4: 600.0, 5: 600.0, 6: 600.0, 7: 600.0, 8: 450.0}
Ss = {1: 500, 2: 450, 3: 425, 4: 400, 5: 200, 6: 30, 7: 20, 8: 5}
Ss_own = {1: 200, 2: 100, 3: 100, 4: 50, 5: 7, 6: 1.5, 7: 0.7, 8: 0.14}

# ---- Methods to write out results
def write_xyz(coords, elements, smiles, filename):
    outfile = open(filename, "w")

    for molidx, mol in enumerate(coords):
        el = elements[molidx]
        smi = smiles[molidx]
        outfile.write("%i\n%s\n" % (len(mol), smi))

        for atomidx, atom in enumerate(mol):
            outfile.write("%s %f %f %f\n" % (el[atomidx], atom[0], atom[1], atom[2]))

    outfile.close()

def write_xyz_single_mol(coords, elements, smiles, filename):
    outfile = open(filename, "w")
    outfile.write("%i\n%s\n" % (len(coords), smiles))

    for atomidx, atom in enumerate(coords):
        outfile.write("%s %f %f %f\n" % (elements[atomidx], atom[0], atom[1], atom[2]))

    outfile.close()

def get_cs(N):
    # sequential generation of random numbers
    cs = np.zeros((N))
    s = 0.0
    order = np.array(range(N))
    np.random.shuffle(order)
    # c_max=1.2
    cs_sum = np.random.random() * (float(int(num))) ** 0.5  # *2.0#**0.5

    for idx in order:
        c_new = 100.0
        while c_new > cs_sum:
            c_new = np.abs(np.random.normal(scale=1.0)) / float(N)
        cs[idx] = c_new * (cs_sum - s)  # np.exp(-1.0/(1.0-s)))#0.5*(1.0-s))
        s = np.sum(cs)
    cs = np.abs(cs)

    return (cs)


# ---- Method to generate Conformers
def generate_confs(coords_new, elements_new, wavenumbers, vectors):
    # energies_eV=np.array(wavenumbers)/8065.54429 # 1/cm to eV
    wavenumbers_np = np.array(wavenumbers)  ## 1/cm
    #forceconstants = 4.0 * np.pi ** 2 * c ** 2 * wavenumbers_np ** 2 * 100 ** 2.0 * reducedmasses_kg  ## N/m or J/m^2
    #forceconstants = forceconstants / e * m_to_A ** 2.0  ## eV / A^2

    mass_per_atom_vector = []

    for element in elements_new:
        for i in range(0, 3):
            mass_per_atom_vector.append(elementmasses[element])

    mass_per_atom_vector = np.array(mass_per_atom_vector)
    vectors_massweighted = np.zeros((len(vectors), len(coords_new), 3))

    for vec_idx, vec in enumerate(vectors):
        vectors_massweighted[vec_idx] = (vec.flatten() * mass_per_atom_vector ** 0.5).reshape((len(coords_new), 3))

    scalar_products_massweighted = np.zeros((len(vectors), len(vectors)))

    for idx1 in range(len(vectors)):
        for idx2 in range(len(vectors)):
            scalar_products_massweighted[idx1][idx2] = np.sum(
                vectors_massweighted[idx1].flatten() * vectors_massweighted[idx2].flatten())

    # print([scalar_products_massweighted[i][i] for i in range(len(scalar_products_massweighted))])
    # generate orthonormal vectors: they were used by ANI authors

    vectors_orthonormal = np.zeros((len(vectors), len(coords_new), 3))

    for vec_idx, vec in enumerate(vectors_massweighted):
        vectors_orthonormal[vec_idx] = np.copy(vec) / np.linalg.norm(vec.flatten())
    # scalar_products_orthonormal=np.zeros((len(vectors),len(vectors)))
    # for idx1 in range(len(vectors)):
    #    for idx2 in range(len(vectors)):
    #        scalar_products_orthonormal[idx1][idx2]=np.sum(vectors_orthonormal[idx1].flatten()*vectors_orthonormal[idx2].flatten())
    # print([scalar_products_orthonormal[i][i] for i in range(len(scalar_products_orthonormal))])
    # some parameters from the paper

    Nf = len(wavenumbers)
    Na = float(len(coords_new))
    kBT_eV = 0.025 / 300.0 * Ts[int(num)]
    # the non-stochastic part of the coefficients
    Rs0 = np.sqrt((3.0 * Na * kBT_eV) / (forceconstants))
    # generate some random conformers, get the DFT energies and make a histogram
    num_conformers = int(round(float(Ss_own[int(num)]) * float(Nf)))
    # E_tm=[]
    conformers_own = []
    conformers_own.append(coords_new)

    for confidx in range(num_conformers):
        # get random numbers with sum 1
        cs = get_cs(Nf)
        # get random signs
        signs = (np.random.randint(0, 2, size=Nf) * 2 - 1)
        # get the coeficcients
        Rs = signs * Rs0 * np.sqrt(cs)  # /2.0**0.5
        # calculate the coordinates of the new conformer
        coords_conformer = np.copy(coords_new)

        for R_idx, R in enumerate(Rs):
            coords_conformer += R * np.copy(vectors_orthonormal[R_idx])

        conformers_own.append(coords_conformer)

        # continue
        # print("DFT calculation %i of %i"%(confidx+1,num_conformers))
        # tm_done,e_tm=tm.get_energy(coords_conformer,S,0)
        # E_tm.append(e_tm)

    return (conformers_own)

def get_conformeres(coords, elements):
    pass



coords, elements = dpu.read_coords_elements("/home/klara/Bachelorarbeit/mldimred/input/methanole.xyz")
