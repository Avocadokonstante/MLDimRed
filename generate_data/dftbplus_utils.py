from __future__ import print_function
from __future__ import absolute_import
from tqdm import tqdm
import subprocess
import os
import time
import sys
import numpy as np
from ase import Atoms
from ase.calculators.dftb import Dftb

def readXYZ(filename):
    infile = open(filename, "r")
    coords = []
    elements = []
    lines = infile.readlines()
    if len(lines) < 3:
        exit("ERROR: no coordinates found in %s/%s" % (os.getcwd(), filename))
    for line in lines[2:]:
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    infile.close()
    coords = np.array(coords)
    return coords, elements

def readXYZs(filename):
    infile = open(filename,"r")
    coords = [[]]
    elements = [[]]
    for line in infile.readlines():
        if len(line.split()) == 1 and len(coords[-1]) != 0:
            coords.append([])
            elements.append([])
        elif len(line.split()) == 4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    infile.close()
    return coords,elements


def get_hess(settings):
    coords = settings["coords"]
    elements = settings["elements"]
    hess = []

    atoms = Atoms(elements, coords)
    calc = (Dftb(atoms=atoms,
                label='dftb_hessian',
                Driver="SecondDerivatives{}",
                Hamiltonian_MaxAngularMomentum_='',
                Hamiltonian_MaxAngularMomentum_O='p',
                Hamiltonian_MaxAngularMomentum_H='s',
                Hamiltonian_MaxAngularMomentum_C='p',
                ))
    atoms.clac = calc
    calc.calculate(atoms)

    f = open("hessian.out", "r")
    lines = f.readlines()
    for line in lines[0:]:
        for number in line.split():
            hess.append(float(number))

    n = len(elements)
    hess = np.array(hess).reshape(3 * n, n, 3)

    print("Hessian shape: ", hess.shape)
    print("   ---   Hessian is calculated")
    settings["hess"] = hess
    return hess

def calculate_modes():
    os.system('cd /home/klara/Bachelorarbeit/mldimred/generate_data/modes ; modes modes_in.hsd')

def get_modes():
    '''
    reads in modes in 1/cm from modes.xyz
    '''
    modes = []
    if not (os.path.isfile("modes/modes.xyz")):
        calculate_modes()
    else :
        m_file = open("modes/modes.xyz", "r")


        modes =  []

        for line in m_file.readlines():
            if len(line.split()) == 4:
                modes.append((float(line.split()[2])))
        m_file.close()
    return modes

def do_dftbplus_runs(settings, name, coords_todo, elements_todo):
    if "test" in name:
        outdir = settings["outdir_test"]
    else:
        outdir = settings["outdir"]
        # initial xtb runs
    if os.path.exists("%s/es_%s.txt" % (outdir, name)) and not settings["overwrite"]:
        print("   ---   load %s labels" % (name))
        es = np.loadtxt("%s/es_%s.txt" % (outdir, name))
    else:
        print("   ---   load t%s labels" % (name))
        es, broken_list = run_dftb(coords_todo, elements_todo)

    if "test" in name:
        np.savetxt("{}/es_test.txt".format (outdir), es)
        np.savetxt("{}/broken_test.txt".format(outdir), broken_list)
    else:
        np.savetxt("{}/es_train.txt".format(outdir), es)
        np.savetxt("{}/broken_train.txt".format(outdir), broken_list)

    return (es, broken_list)

def run_dftb(coords, elements):
    es = []
    broken_list = []
    print("   ---   Energies are calculated")
    broken = 0

    for molidx in tqdm(range(len(coords))):

        c_here = coords[molidx]
        el_here = elements[molidx]

        atoms = Atoms(el_here, c_here)
        atoms.set_calculator(Dftb(label='dftb', atoms=atoms,
                                  Driver="{}",
                                  Hamiltonian_MaxAngularMomentum_='',
                                  Hamiltonian_MaxAngularMomentum_O='"p"',
                                  Hamiltonian_MaxAngularMomentum_H='"s"',
                                  Hamiltonian_MaxAngularMomentum_C='"p"',
                                  Hamiltonian_MaxAngularMomentum_N='"p"'
                                  ))

        try:
            energy = atoms.get_total_energy()
            es.append(energy)
        except:
            print("DFTB+ Error")
            broken += 1
            broken_list.append(molidx)
            #lösche coord aus coord.file
            continue

    print(broken)
    print(broken_list)
    es = np.array(es)
    return es, broken_list



def exportXYZs(coords,elements,filename):
    outfile = open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements[idx])))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[idx][atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()

