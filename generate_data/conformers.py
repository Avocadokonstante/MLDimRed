import numpy as np
import dftbplus_utils as dftb

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

def generate_conformers(settings):
    modes = dftb.get_modes()
    hess = dftb.get_hess(settings)
    print(modes)

    r = np.random.randn() * 0.1
    c_here += r * hess[v_idx]
    f = settings["vibspectrum"][v_idx]
    r = np.random.randn() * 0.1 * (f / 100.0) ** 2.0
    c_here += r * hess[v_idx]

generate_conformers()













