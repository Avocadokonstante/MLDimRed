import dataprep_utils as dpu
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import analyse_utils as au
from tqdm import tqdm
import keras as ks
import os
import matplotlib.pyplot as plt
import pandas as pd
import rmsd
import seaborn as sns

x_train = np.load('anthracene/anthracene_train.npy')
x_test = np.load('anthracene/anthracene_test.npy')
elements = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'C', 'C', 'H', 'C', 'H', 'H', 'H']

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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

#shuffle data
shuffle_train = get_shuffled_indices(x_train.shape[0])
x_train = x_train[shuffle_train]

print('x_train: ',x_train.shape)
print('x_test: ' ,x_test.shape)

directory = "new_directory"
title='pca'
if not os.path.isdir(directory):
    os.mkdir(directory)

#PCA for one dimension
pca = PCA(n_components=8)
X_encoded = pca.fit_transform(x_train)
X_decoded = pca.inverse_transform(X_encoded)

x_encoded_test = pca.transform(x_test)
x_decoded_test = pca.inverse_transform(x_encoded_test)

#Calculate Error for every reduced Dimension
def latent_space_analysis(x_train, x_test):
    max_dim=len(x_train[1])

    data = {'latent space dimension': [],
            'train_mae': [],
            'train_mse': [],
            'train_r2': [],
            'test_mae': [],
            'test_mse': [],
            'test_r2': [], }
    df = pd.DataFrame(data)

    for dim in tqdm(range(1, max_dim+1)):
        pca = PCA(n_components=dim)
        X_encoded = pca.fit_transform(x_train)
        X_decoded = pca.inverse_transform(X_encoded)

        r2_train = (r2_score(x_train, X_decoded))
        mse_train = mse(x_train, X_decoded)
        mae_train = (mae(x_train, X_decoded))

        x_encoded_test = pca.transform(x_test)
        x_decoded_test = pca.inverse_transform(x_encoded_test)

        test_rescaled = scaler.inverse_transform(x_test)
        x_decoded_test = scaler.inverse_transform(x_decoded_test)

        r2_test = (r2_score(test_rescaled, x_decoded_test))
        mse_test = (mse(test_rescaled, x_decoded_test))
        mae_test = (mae(test_rescaled, x_decoded_test))

        new_row = {'latent space dimension': dim,
                   'train_mae': mae_train,
                   'train_mse': mse_train,
                   'train_r2' : r2_train,
                   'test_mae': mae_test,
                   'test_mse': mse_test,
                   'test_r2': r2_test,
                   }

        df_new = df.append(new_row, ignore_index=True)
        df = df_new
        df_new.to_csv('{}/pca_results.csv'.format(directory))

latent_space_analysis(x_train, x_test)



#plot Metrics in Diagram
def latent_space_error_plot(directory, n=1):
    results = pd.read_csv('{}/pca_results.csv'.format(directory))

    latent = results['latent space dimension']
    latent = latent[::n]
    r2_test = results['test_r2']
    r2_test = r2_test[::n]
    test_mse = results['test_mse']
    test_mse = test_mse[::n]
    test_mae = results['test_mae']
    test_mae = test_mae[::n]

    paper_rc = {'lines.linewidth': 2.5}
    sns.set_theme(font_scale=1.25, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
    sns.set_style('white')

    ax = plt.gca()

    plt.plot(latent, r2_test, 'gv', markersize=6)
    plt.legend(['r2_test'], loc='lower right')
    plt.grid(True)
    plt.title('PCA mit verschiedenen Dimensionen')
    plt.ylabel('r2 Wert')
    plt.xlabel('Dimension')
    plt.tight_layout()
    plt.savefig('{}/pca.pdf'.format(directory))
    plt.savefig('{}/pca.png'.format(directory))
    plt.close()


    plt.plot(latent, test_mae, 'g^', markersize=6)
    plt.plot(latent, test_mse, 'm*', markersize=7)
    plt.legend(['mae_test in nm', 'mse_test in nm^2'], loc='upper right')
    plt.title('PCA mit verschiedenen Dimensionen')
    plt.ylabel('Fehler')
    plt.xlabel('Dimension')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('{}/pca_mse.pdf'.format(directory))
    plt.savefig('{}/pca_mse.png'.format(directory))
    plt.close()

latent_space_error_plot(directory, 10)


def rmsd_test(dim, x_test):
    pca = PCA(n_components=dim)
    pca.fit_transform(x_train)

    x_encoded_test = pca.transform(x_test)
    x_decoded_test = pca.inverse_transform(x_encoded_test)

    x_decoded = np.asarray(x_decoded_test)
    dpu.save_inverse_distances_as_coordinates(x_decoded, elements, 'pca_x_test_{}.xyz'.format(dim), 'test')
    x_decoded = scaler.inverse_transform(x_decoded)
    m = dpu.distance_vectors_to_distance_matrixs(x_decoded)
    m = np.array(m)
    m_list = []
    for i in m:
        m_list.append(dpu.coordinates_from_distancematrix(i))

    x_test = scaler.inverse_transform(x_test)
    x_test_coord = dpu.distance_vectors_to_distance_matrixs(x_test)
    x_test_coord = np.array(x_test_coord)
    x_test_coord_list = []
    for i in x_test_coord:
        x_test_coord_list.append(dpu.coordinates_from_distancematrix(i))

    print(type(m_list))
    print(m_list[1])
    print(x_test_coord_list[1])
    rmsd_list = []
    for i in range(len(m_list)):
        Bout, R, t = dpu.rigid_transform(m_list[i], x_test_coord_list[i])
        res = rmsd.rmsd(x_test_coord_list[i], Bout)
        rmsd_list.append(res)


    print(rmsd_list)
    au.plot_histogram(directory, 'rmsd_{}'.format(dim), rmsd_list, 50)
    print(rmsd)

rmsd_test(66, x_test)
