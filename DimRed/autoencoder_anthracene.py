import dataprep_utils as dpu
import analyse_utils as pltu
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
tf.compat.v1.disable_eager_execution()
import keras as ks
import numpy as np
import csv


tf.disable_v2_behavior()
tf.enable_eager_execution()

def get_shuffled_indices(num_examples: int):
    indices: any = np.arange(num_examples)
    np.random.shuffle(indices)
    return indices

directory ="new_directory"
title = "anthracene"

x_train = np.load('anthracene/anthracene_train.npy')
x_test = np.load('anthracene/anthracene_test.npy')
elements = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'C', 'H', 'C', 'H', 'H', 'H']

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#shuffle data
shuffle_train = get_shuffled_indices(x_train.shape[0])
x_train = x_train[shuffle_train]


#save shuffled inverse distances as coordinates in xyz file
dpu.save_inverse_distances_as_coordinates(x_train, elements, '{}_inverse_distances_train_shuffled.xyz'.format(title), directory)
dpu.save_inverse_distances_as_coordinates(x_test, elements, '{}_inverse_distances_test_shuffled.xyz'.format(title), directory)

#Scale data
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform((x_train))
x_test_scaled = scaler.transform(x_test)


class Autoencoder(Model):
  def __init__(self, latent_dim, startLayer, activation):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
        layers.Dense(startLayer, activation=activation, kernel_regularizer='l2'),
        layers.Dropout(0.25),
        layers.Dense(latent_dim, activation=activation, kernel_regularizer='l2'),

    ])
    self.decoder = tf.keras.Sequential([
        layers.Dense(latent_dim, activation=activation, kernel_regularizer='l2'),
        layers.Dropout(0.25),
        layers.Dense(startLayer, activation='linear', kernel_regularizer='l2'),

    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def r2_metric(y_true, y_pred):
    """
    Compute r2 metric.    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.    Returns:
        tf.tensor: r2 metric.    """
    SS_res =  ks.backend.sum(ks.backend.square(y_true - y_pred))
    SS_tot = ks.backend.sum(ks.backend.square(y_true-ks.backend.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + ks.backend.epsilon()) )

train = x_train[:8000]
print(train.shape)
val = x_train[8000:]
print(val.shape)

autoencoder = Autoencoder(8, x_train.shape[1], 'tanh')
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
autoencoder.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=['mse', 'mae', r2_metric])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

history = autoencoder.fit(train, train,
                    epochs=10000,
                    callbacks=[callback],
                    shuffle=False,
                    validation_data=(val, val))

print(history.history.keys())
print(history)


pltu.autoencoder_history_lines(history, 'all', title, directory)
pltu.write_epoche_results(history, title, directory)


#On Testset
encoded_distances = autoencoder.encoder(x_test).numpy()
decoded_distances = autoencoder.decoder(encoded_distances).numpy()

test_rescaled = scaler.inverse_transform(x_test)
x_decoded_test = scaler.inverse_transform(decoded_distances)

r2_test = (r2_score(test_rescaled, x_decoded_test))
mse_test = (mse(test_rescaled, x_decoded_test))
mae_test = (mae(test_rescaled, x_decoded_test))


with open('{}/{}_results.csv'.format(directory, title), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(('mse_train',history.history['mse'][-1]))
    csvwriter.writerow(('val_mse', history.history['val_mse'][-1]))
    csvwriter.writerow(('mse_test', mse_test))

    csvwriter.writerow(('mae_train',history.history['mae'][-1]))
    csvwriter.writerow(('val_mae', history.history['val_mae'][-1]))
    csvwriter.writerow(('mae_test',mae_test))

    csvwriter.writerow(('r2_train',history.history['r2_metric'][-1]))
    csvwriter.writerow(('r2_val', history.history['val_r2_metric'][-1]))
    csvwriter.writerow(('r2_test', r2_test))


dpu.save_inverse_distances_as_coordinates(decoded_distances, elements, '{}_decoded_coords.xyz'.format(title), directory)









