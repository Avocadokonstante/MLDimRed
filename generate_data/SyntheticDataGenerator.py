import os
import sys
#import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as scsp
import dftbplus_utils as dftbpl
import utils as u
import ml_utils as ml

# read input
infilename = "../input/methanole.xyz"
coords, elements = dftbpl.readXYZ(infilename)
print(coords)
print(elements)
n = len(coords)
overwrite = False
settings = {"overwrite": overwrite,
            "infilename": infilename,
            "coords": coords,
            "elements": elements,
            "n": n,
            "amplitude" : 0.3
            }

# output
u.prep_dirs(settings)


# optimize and get hessian
hess = dftbpl.get_hess(settings)


# test the vibrations
modes = dftbpl.get_modes()
settings["vibspectrum"] = modes
u.vibrations(settings)

# initial sampling
num_train = 2000
coords_train, elements_train = u.do_sampling(settings, "train", num_train)
num_test = 500
coords_test, elements_test = u.do_sampling(settings, "test", num_test)


print("Begin energy calculation")
es_train , broken_train= dftbpl.do_dftbplus_runs(settings, "train", coords_train, elements_train)
es_test, broken_test = dftbpl.do_dftbplus_runs(settings, "test", coords_test, elements_test)


# training and test data
X_train, X_train_scaled, y_train, y_train_scaled, x_scaler, y_scaler = ml.convert_to_ML_data(coords_train, elements_train, es_train)
X_test, X_test_scaled, y_test, y_test_scaled, x_scaler, y_scaler = ml.convert_to_ML_data(coords_test, elements_test, es_test, x_scaler, y_scaler)


y_min = np.min(y_test)-0.1*(np.max(y_test)-np.min(y_test))
y_max = np.max(y_test)+0.1*(np.max(y_test)-np.min(y_test))