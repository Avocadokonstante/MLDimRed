import dataprep_utils as dpu
import analyse_utils as pltu
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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

#read in training and test as inverse Distances
x_train = dpu.get_inverse_distances("../generate_data/output/train.xyz")
x_test = dpu.get_inverse_distances("../generate_data/output/test.xyz")
elements = dpu.get_elements('../generate_data/input/methanole.xyz')
print(elements)
directory ="new_directory"
title = "methanole"
print(len(x_train))
print(len(x_test))

broken_train = dpu.read_broken_coords("../generate_data/output/broken_train.txt")
broken_test = dpu.read_broken_coords("../generate_data/output_test/broken_test.txt")

#remove broken coordinates
broken_train = np.array(broken_train)
broken_train = [int(float(i)) for i in broken_train]

broken_test = np.array(broken_test)
broken_test = [int(float(i)) for i in broken_test]

x_train = np.delete(x_train, broken_train, 0)
x_test = np.delete(x_test, broken_test, 0)

print(len(x_train))
print(len(x_test))

#shuffle data
shuffle_train = get_shuffled_indices(x_train.shape[0])
x_train = x_train[shuffle_train]

#save shuffled inverse distances as coordinates in xyz file
dpu.save_inverse_distances_as_coordinates(x_train, elements, '{}_inverse_distances_train_shuffled.xyz'.format(title), directory)
dpu.save_inverse_distances_as_coordinates(x_test, elements, '{}_inverse_distances_test_shuffled.xyz'.format(title), directory)


class Autoencoder(Model):
  def __init__(self, latent_dim, startLayer, activation):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
        layers.Dense(startLayer, activation=activation),
        layers.Dense(latent_dim, activation=activation),

    ])
    self.decoder = tf.keras.Sequential([
        layers.Dense(latent_dim, activation=activation),
        layers.Dense(startLayer, activation='linear'),

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

train = x_train[:1600]
print(train.shape)
val = x_train[1600:]
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

#pltu.auto_all_error_plot(history, 'all', title, directory)
pltu.autoencoder_history_lines(history, 'all', title, directory)
pltu.write_epoche_results(history, title, directory)

#On Testset
encoded_distances = autoencoder.encoder(x_test).numpy()
decoded_distances = autoencoder.decoder(encoded_distances).numpy()

mae_test = (mean_absolute_error(x_test, decoded_distances))
mse_test = (mean_squared_error(x_test, decoded_distances))
r2_test = (r2_score(x_test, decoded_distances))


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


def latentspace_analyse_with_original_geometry(encoded_distances):
    dimension = 0
    test = np.linspace(-2, 2, 1000)

    methanole, element = dpu.read_coords_elements("../generate_data/input/methanole.xyz")
    inv_dist_methanole = dpu.calculate_inv_dist_vector(methanole[0])
    inv_dist_methanole = [inv_dist_methanole]
    input = np.array(inv_dist_methanole)
    encoded_methanole = autoencoder.encoder(input).numpy()
    decoded_methanole = autoencoder.decoder(encoded_methanole)
    m = dpu.distance_vector_to_distance_matrix(decoded_methanole[0])
    m = dpu.coordinates_from_distancematrix(m)

    encoded_methanole = encoded_methanole[0]

    #new_molecules.append(molecule)
    for j in range(len(encoded_methanole)):
        new_molecules = []
        new_molecules.append(encoded_methanole)
        for i in test:
            temp_mol = np.copy(encoded_methanole)
            temp_mol[dimension] = i
            new_molecules.append(temp_mol)

        new_molecules = np.array(new_molecules)
        decoded_new_molecules = autoencoder.decoder(new_molecules).numpy()
        filename = 'dim{}.xyz'.format(j)
        dpu.save_inverse_distances_as_coordinates(decoded_new_molecules, elements, filename, directory)

    return decoded_new_molecules

#latentspace_analyse_with_original_geometry(encoded_distances)



