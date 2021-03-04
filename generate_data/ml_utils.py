from __future__ import print_function
from __future__ import absolute_import
import shutil
import uuid
import os
import sys
#import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as scsp
import time
import subprocess
import shlex
import joblib
import sklearn
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def convert_to_ML_data(coords_todo, elements_todo, es_todo, x_scaler=None, y_scaler=None):
    X_todo = get_representation(coords_todo, elements_todo)
    if x_scaler is None:
        x_scaler = StandardScaler()
        X_todo_scaled = x_scaler.fit_transform(X_todo)
    else:
        X_todo_scaled = x_scaler.transform(X_todo)
    y_todo = es_todo.reshape(-1,1)
    if y_scaler is None:
        y_scaler = StandardScaler()
        y_todo_scaled = y_scaler.fit_transform(y_todo)
    else:
        y_todo_scaled = y_scaler.transform(y_todo)
    return(X_todo, X_todo_scaled, y_todo, y_todo_scaled, x_scaler, y_scaler)


def get_representation(coords, elements):
    X = []
    for c_here in coords:
        ds = scsp.distance.pdist(c_here)
        X.append(ds)
    #X = 1.0/np.array(X)
    X = np.array(X)
    return(X)


def do_model_training(settings, X_train_scaled, y_train_scaled, name, n_models):
    outdir = settings["outdir"]
    found_all=True
    for i in range(n_models):
        if not os.path.exists("%s/models/model_%s_%i.joblib"%(outdir, name, i)):
            found_all = False
    if found_all:
        print("   ---   load models")
        models = []
        for i in range(n_models):
            model = joblib.load("%s/models/model_%s_%i.joblib"%(outdir, name, i))
            models.append(model)
        n_models = len(models)
    else:
        print("   ---   train models")
        models = train(X_train_scaled, y_train_scaled, n_models)
        for i, model in enumerate(models):
            joblib.dump(model, "%s/models/model_%s_%i.joblib"%(outdir, name, i))
    return(models, n_models)


def train(X_train, y_train, n_models = 3):
    models = []
    for i in range(n_models):
        regr = MLPRegressor(hidden_layer_sizes=(200, 100, 50, ), activation='tanh', random_state=i, max_iter=5000).fit(X_train, y_train.ravel())
        models.append(regr)
    return(models)




def do_predictions(models, X_scaled, y_scaler):
    preds = []
    for i, model in enumerate(models):
        pred_scaled = model.predict(X_scaled)
        preds_unscaled = y_scaler.inverse_transform(pred_scaled)
        preds.append(preds_unscaled)
    preds = np.array(preds).T
    return(preds)


def reg_stats(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #if scaler:
    #  y_true_unscaled = scaler.inverse_transform(y_true)
    #  y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return(r2, mae)






def NN_tensorflow():

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    #test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))


    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(128,activation='tanh', activity_regularizer=tf.keras.regularizers.L2(0.01)),
      tf.keras.layers.Dense(10, activation='tanh', activity_regularizer=tf.keras.regularizers.L2(0.01))
    ])
    model.compile(
        loss='MSE',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['MSE'],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )





