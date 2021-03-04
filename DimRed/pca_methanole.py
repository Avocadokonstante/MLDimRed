import dataprep_utils as dpu
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import analyse_utils as au
import keras as ks
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import rmsd
import seaborn as sns

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

x_train = dpu.get_inverse_distances("../generate_data/output/train.xyz")
x_test = dpu.get_inverse_distances("../generate_data/output/test.xyz")
elements = dpu.get_elements('../generate_data/input/methanole.xyz')
print(elements)

#remove broken coordinates
broken_train = dpu.read_broken_coords("../generate_data/output/broken_train.txt")
broken_test = dpu.read_broken_coords("../generate_data/output_test/broken_test.txt")

broken_train = np.array(broken_train)
broken_train = [int(float(i)) for i in broken_train]

broken_test = np.array(broken_test)
broken_test = [int(float(i)) for i in broken_test]

x_train = np.delete(x_train, broken_train, 0)
x_test = np.delete(x_test, broken_test, 0)

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

        r2_test = (r2_score(x_test, x_decoded_test))
        mse_test = (mse(x_test, x_decoded_test))
        mae_test = (mae(x_test, x_decoded_test))

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

latent_space_error_plot(directory)

#Analyse one dimension of reduced inverse distance vector
def latentspace_analyse_with_original_geometry(dim):
    dimension = 0
    test = np.linspace(-1, 1, 1000)

    pca = PCA(n_components=dim)

    methanole, element = dpu.read_coords_elements("../input/methanole.xyz")
    inv_dist_methanole = dpu.calculate_inv_dist_vector(methanole[0])
    inv_dist_methanole = [inv_dist_methanole]
    input = np.array(inv_dist_methanole)
    print(input)
    stuff = pca.fit_transform(x_train)
    print(stuff)
    input = input.reshape(1, -1)
    encoded_methanole = pca.transform(input)
    decoded_methanole = pca.inverse_transform(encoded_methanole)
    m = dpu.distance_vector_to_distance_matrix(decoded_methanole[0])
    m = dpu.coordinates_from_distancematrix(m)

    encoded_methanole = encoded_methanole[0]

    for j in range(len(encoded_methanole)):
        new_molecules = []
        new_molecules.append(encoded_methanole)
        for i in test:
            temp_mol = np.copy(encoded_methanole)
            temp_mol[dimension] = i
            new_molecules.append(temp_mol)

        new_molecules = np.array(new_molecules)
        decoded_new_molecules = pca.inverse_transform(new_molecules)
        filename = 'dim{}.xyz'.format(j)
        dpu.save_inverse_distances_as_coordinates(decoded_new_molecules, elements, filename, directory)

    return decoded_new_molecules

#latentspace_analyse_with_original_geometry()

#histograms comparing original and reconstructed geometries
def rmsd_test(dim, x_test):
    pca = PCA(n_components=dim)
    pca.fit_transform(x_train)

    x_encoded_test = pca.transform(x_test)
    x_decoded_test = pca.inverse_transform(x_encoded_test)

    x_decoded = np.asarray(x_decoded_test)
    dpu.save_inverse_distances_as_coordinates(x_decoded, ['C', 'O', 'H', 'H', 'H', 'H'], 'pca_x_test_{}.xyz'.format(dim), 'test')
    m = dpu.distance_vectors_to_distance_matrixs(x_decoded)
    m = np.array(m)
    m_list = []
    for i in m:
        m_list.append(dpu.coordinates_from_distancematrix(i))

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

rmsd_test(8, x_test)
