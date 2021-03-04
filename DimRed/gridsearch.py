import dataprep_utils as dpu
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from itertools import product

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
tf.compat.v1.disable_eager_execution()
import keras as ks
import numpy as np
import pandas as pd

tf.disable_v2_behavior()
tf.enable_eager_execution()


def get_shuffled_indices(num_examples: int):
    indices: any = np.arange(num_examples)
    np.random.shuffle(indices)
    return indices

def r2_metric(y_true, y_pred):
    """
    Compute r2 metric.    Args:
        y_true (tf.tensor): True y-values.
        y_pred (tf.tensor): Predicted y-values.    Returns:
        tf.tensor: r2 metric.    """
    SS_res =  ks.backend.sum(ks.backend.square(y_true - y_pred))
    SS_tot = ks.backend.sum(ks.backend.square(y_true-ks.backend.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + ks.backend.epsilon()) )


#read in training and test as inverse Distances
x_train = dpu.get_inverse_distances("../generate_data/output/train.xyz")
x_test = dpu.get_inverse_distances("../generate_data/output/test.xyz")
elements = dpu.get_elements('../generate_data/input/methanole.xyz')
startLayer = len(x_test[0])

broken_train = dpu.read_broken_coords("../generate_data/output/broken_train.txt")
broken_test = dpu.read_broken_coords("../generate_data/output_test/broken_test.txt")

#remove broken coordinates
broken_train = np.array(broken_train)
broken_train = [int(float(i)) for i in broken_train]

broken_test = np.array(broken_test)
broken_test = [int(float(i)) for i in broken_test]

x_train = np.delete(x_train, broken_train, 0)
x_test = np.delete(x_test, broken_test, 0)

#shuffle data
shuffle_train = get_shuffled_indices(x_train.shape[0])
x_train = x_train[shuffle_train]

class Autoencoder(Model):
  def __init__(self, activation, num_neurons_encoder, num_neurons_decoder):
    super(Autoencoder, self).__init__()
    layer_en = tf.keras.Sequential([layers.Dense(num_neurons_encoder[n], activation=activation) for n in range(len(num_neurons_encoder))])
    layer_de = tf.keras.Sequential([layers.Dense(num_neurons_decoder[n], activation=activation) for n in range(len(num_neurons_decoder) - 1)])
    self.encoder = tf.keras.Sequential([layer_en])
    self.decoder = tf.keras.Sequential([layer_de, layers.Dense(num_neurons_decoder[-1], activation='linear')])

  def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded


def get_architectures(hiddenLayer, latent_dim):
    tube = [latent_dim, startLayer]
    decoder_list = []
    #generate tube
    for i in range(hiddenLayer):
        tube.append(startLayer)
        add = tube.copy()
        decoder_list.append(add)

    #generate interpolation
    for i in range(3, hiddenLayer+3):
        new = np.linspace(startLayer,  latent_dim, i, dtype=int)
        new = new.tolist()
        if new != [startLayer ,latent_dim]:
            decoder_list.append(new[::-1])

    return decoder_list

def grid_search(latent_dim, param_grid, optimizer_list, x_train, x_val, x_test):
    round = 1
    hiddenLayers = 3


    data = {'round': [],
            'epochs': [],
            'val_mae': [],
            'val_mse': [],
            'val_r2': [],
            'test_mae': [],
            'test_mse': [],
            'test_r2': [],
            'encoder_structure': [],
            'decoder_structure': [],
            'activation': [],
            'optimizer': [],
            'learning_rate': []}
    df = pd.DataFrame(data)

    architectures = get_architectures(hiddenLayers, latent_dim)
    print(architectures)

    for activation, opt, lr, architecture in product(param_grid["activation"], param_grid['optimizer'], param_grid['learning_rates'], architectures):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
        num_neurons_encoder = architecture[::-1]
        num_neurons_decoder = architecture
        autoencoder = Autoencoder(activation, num_neurons_encoder, num_neurons_decoder)
        optimizer = optimizer_list.get(opt)
        optimizer.learning_rate = lr

        try:
            autoencoder.compile(optimizer=optimizer, loss=losses.MeanSquaredError(),
                                metrics=['mse', 'mae', r2_metric])
            history = autoencoder.fit(x_train, x_train,
                                      epochs=10000,
                                      callbacks=[callback],
                                      shuffle=True,
                                      validation_data=(x_val, x_val))

            print(history.history.keys())
            print(autoencoder.summary())

            encoded_distances = autoencoder.encoder(x_test).numpy()
            decoded_distances = autoencoder.decoder(encoded_distances).numpy()

            mae_test = (mean_absolute_error(x_test, decoded_distances))
            mse_test = (mean_squared_error(x_test, decoded_distances))
            r2_test = (r2_score(x_test, decoded_distances))


            new_row = {'round': round,
                       'epochs': len(history.history['val_mae']),
                       'val_mae': history.history['val_mae'][-1],
                       'val_mse': history.history['val_mse'][-1],
                       'val_r2': history.history['val_r2_metric'][-1],
                       'test_mae': mae_test,
                       'test_mse': mse_test,
                       'test_r2': r2_test,
                       'encoder_structure': architecture[::-1],
                       'decoder_structure': architecture,
                       'activation': activation,
                       'optimizer': tf.keras.optimizers.serialize(optimizer).get('class_name'),
                       'learning_rate': (tf.keras.optimizers.serialize(optimizer).get('config')).get('learning_rate')}

            print('NEW ROW')
            print(new_row)

            df_new = df.append(new_row, ignore_index=True)
            df = df_new
            df_new.to_csv('gridsearch_{}_3.csv'.format(latent_dim))

            autoencoder.save_weights('saved_model3/model_m12_{}'.format(round))

            round += 1
        except:
            pass


param_grid = {
        'activation': ['elu','relu', 'linear', 'tanh', 'sigmoid'],
        'learning_rates': [1e-3, 1e-2, 1e-4],
        'optimizer': ['adadelta', 'adamax', 'adam', 'SGD', 'nadam']
    }

optimizer_list = {
        'SDG': tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False, name='SGD'),
        'adam': tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'),
        'nadam': tf.keras.optimizers.Nadam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
        'adamax': tf.keras.optimizers.Adamax(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'),
        'adadelta': tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.95, epsilon=1e-07, name='Adadelta')

}

grid_search(12, param_grid, optimizer_list, x_train[:1600], x_train[1600:], x_test)
