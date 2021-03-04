import numpy as np
import rmsd
import os

def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def calculate_inv_dist_vector(coord):
    '''
    Input: np.array of shape (atoms length, 3)
    output: Inverse distance vector as np.array
    '''
    distmatrix = np.asarray(calculate_inverse_distance(coord))
    inv_dist_vector = np.asarray(calculate_upper_triangular_matrix(distmatrix))
    return (inv_dist_vector)

def calculate_inverse_distance(coord):
    '''
    Input: np.array of shape (atoms length, 3)
    Output: inverse distance matrix as np.array of shape (atoms length, atoms length)
    '''
    atom_num = len(coord)

    distmatrix = np.zeros([atom_num, atom_num])
    for i in range(atom_num):
        for j in range(atom_num):
            distmatrix[i,j] = np.linalg.norm(coord[i] - coord[j])

    return distmatrix

def calculate_upper_triangular_matrix(matrix):
    '''
    Input: np.array of shape (N,N)
    Output: list of upper triangular matrix elements excluding diagonal elements
    '''
    dist_vec = []

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i < j:
                dist_vec.append(matrix[i][j])

    return dist_vec

def coordinates_from_distancematrix(DistIn,use_center = None,dim=3):
    """
    Computes list of coordinates from a distance matrix of shape (N,N)    Args:
        DistIn (numpy array): Distance matrix of shape (N,N) with D_ij = |r_i-r_j|
        use_center (int): which atom should be the center, dafault = None means center of mass
        dim (int): the dimension of embedding, 3 is default    Return:
        List of Atom coordinates [[x_1,x_2,x_3],[x_1,x_2,x_3],...]     Uses vectorized Alogrithm:
        http://scripts.iucr.org/cgi-bin/paper?S0567739478000522
        https://www.researchgate.net/publication/252396528_Stable_calculation_of_coordinates_from_distance_information
    no check of positive semi-definite or possible k-dim >= 3 is done here
    performs svd from numpy
    may even wok for (...,N,N) but not tested
    """
    DistIn = np.array(DistIn)
    dimIn = DistIn.shape[-1]
    if use_center is None:
        #Take Center of mass (slightly changed for vectorization assuming d_ii = 0)
        di2 = np.square(DistIn)
        di02 = 1/2/dimIn/dimIn*(2*dimIn*np.sum(di2,axis=-1)-np.sum(np.sum(di2,axis=-1),axis=-1))
        MatM = (np.expand_dims(di02,axis=-2) + np.expand_dims(di02,axis=-1) - di2)/2 #broadcasting
    else:
        di2 = np.square(DistIn)
        MatM = (np.expand_dims(di2[...,use_center],axis=-2) + np.expand_dims(di2[...,use_center],axis=-1) - di2 )/2
    u,s,v = np.linalg.svd(MatM)
    vecs = np.matmul(u,np.sqrt(np.diag(s))) # EV are sorted by default
    distout = vecs[...,0:dim]
    return distout

def exportXYZs(coords,elements,filename):
    outfile = open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements)))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()

def get_inverse_distances(c_path):
    '''
    Input: Path to the coordinates that the ML input should be calculated from as xyz
    Output: inverse distance vectors as np.array of shape (samples, inverse_distance_vector)
    '''
    # ---- evtls erweitern zu Energies einlesen

    # ---- Read in Coordinates
    coords, elements = read_coords_elements(c_path)

    #print(coords.shape)
    #print(elements.shape)
    #print(len(energies))

    # ---- Calculate inverse distance vektor
    inv_vector = get_inverse_distance_vectors(coords)
    rmsd_test(coords)

    return (inv_vector)

def get_inverse_distance_vectors(coords):
    '''
    Input: np.array of shape (samples, atom length, 3)
    Output: inverse distance vectors (list) as np.array
    '''
    inverse_distance_vectors = []
    for mol in coords:
        inverse_distance_vectors.append(calculate_inv_dist_vector(mol))
    inverse_distance_vectors = np.array(inverse_distance_vectors)

    return inverse_distance_vectors

def get_distmatrix_list(coords):
    '''
    Input: np.array "list" of all coordinates shape (samples, atoms length, 3)
    Output: list of calculated distance matrices as np.array
    '''
    distmatrix_list = []
    for mol in coords:
        distmatrix_list.append(np.asarray(calculate_inverse_distance(mol)))

    test_inv_dist(distmatrix_list)
    distmatrix_list = np.array(distmatrix_list)
    test_inv_dist(distmatrix_list)
    return distmatrix_list

def read_broken_coords(b_path):
    b_file = open(b_path, "r")
    broken = []

    for line in b_file.readlines():
        broken.append([line.split()[0].capitalize()])
    b_file.close()

    return broken

def read_energies(e_path):
    '''
    Input: Path to enegry file as txt
    Output: Parsed energies as list
    '''
    e_file = open(e_path, "r")
    energies = []

    for line in e_file:
        energies.append(float(line[:-1]))
    e_file.close()

    return energies

def read_coords_elements(c_path):
    '''
    Input: Path to coordinate file as xyz
    Output: Parsed coordinates, elements as np.array
    '''
    c_file = open(c_path, "r")
    elements = []
    coords = []
    atom_number = int(c_file.read(1))

    for line in c_file.readlines():
        if len(line.split()) == 4:
            elements.append([line.split()[0].capitalize()])
            coords.append(np.array([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]))
    c_file.close()

    num_mol = int(len(coords)/(atom_number))
    coords = np.reshape(coords, [num_mol, atom_number, 3])
    elements = np.asarray(elements)

    rmsd_test(coords)

    return coords, elements

def get_elements(path):
    elements = []
    file = open(path, "r")

    for line in file.readlines():
        if len(line.split()) == 4:
            elements.append([line.split()[0].capitalize()])
    file.close()

    elements = (np.array(elements)).flatten()
    return(elements)

def rigid_transform(A, B,correct_reflection=False):
    """
    Rotate and shift pointcloud A to pointcloud B. This should implement Kabsch algorithm.    Important: the numbering of points of A and B must match, no shuffled pointcloud.
    This works for 3 dimensions only. Uses SVD.    Args:
        A (numpy array): list of points (N,3) to rotate (and translate)
        B (numpy array): list of points (N,3) to rotate towards: A to B, where the coordinates (3) are (x,y,z)    Returns:
        A_rot (numpy array): Rotated and shifted version of A to match B
        R (numpy array): Rotation matrix
        t (numpy array): translation from A to B    Note:
        Explanation of Kabsch Algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
        For further literature
        https://link.springer.com/article/10.1007/s10015-016-0265-x
        https://link.springer.com/article/10.1007%2Fs001380050048
        maybe work for (...,N,3), not tested
    """
    A = np.transpose(np.array(A))
    B = np.transpose(np.array(B))
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    Am = A - np.expand_dims(centroid_A,axis=1)
    Bm = B - np.expand_dims(centroid_B,axis=1)
    H = np.dot(Am ,np.transpose(Bm))
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,U.T)
    d = np.linalg.det(R)
    if(d<0):
        #print("Warning: det(R)<0, det(R)=",d)
        if(correct_reflection==True):
            #print("Correcting R...")
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
    Bout = np.dot(R,Am) + np.expand_dims(centroid_B,axis=1)
    Bout = np.transpose(Bout)
    t = np.expand_dims(centroid_B-np.dot(R,centroid_A),axis=0)
    t = t.T
    return Bout,R, t

def save_inverse_distances_as_coordinates(inv_dist_vectors, elements, title, directory):
    make_directory(directory)
    matrix_list = distance_vectors_to_distance_matrixs(inv_dist_vectors)
    coords = []
    for coord in matrix_list:
        c = coordinates_from_distancematrix(coord)
        coords.append(c)

    exportXYZs(coords, elements, '{}/{}'.format(directory,title))

def inverse_distances_to_coordinates(inv_dist_vectors):
    matrix_list = distance_vectors_to_distance_matrixs(inv_dist_vectors)
    coords = []
    for coord in matrix_list:
        c = coordinates_from_distancematrix(coord)
        coords.append(c)

    return(np.asarray(coords))

def distance_vector_to_distance_matrix(vector):
    x = 0
    i = 1
    while x < len(vector):
        x += i
        i += 1
    temp = np.zeros([i, i])
    l = 0
    for n in range(i):
        for m in range(i):
            if n == m:
                temp[n][m] = 0
            elif n < m:
                temp[n][m] = vector[l]
                temp[m][n] = vector[l]
                l += 1
    return temp

def distance_vectors_to_distance_matrixs(distance_vectors):
    matrix_list = []
    for vector in distance_vectors:
        matrix_list.append(distance_vector_to_distance_matrix(vector))

    return(np.array(matrix_list))

def write_xyz(coords, elements, smiles, filename):
    outfile = open(filename, "w")

    for molidx, mol in enumerate(coords):
        el = elements[molidx]
        smi = smiles[molidx]
        outfile.write("%i\n%s\n" % (len(mol), smi))

        for atomidx, atom in enumerate(mol):
            outfile.write("%s %f %f %f\n" % (el[atomidx], atom[0], atom[1], atom[2]))

    outfile.close()

def write_xyz_single(coord, elements, filename):
    outfile = open(filename, "w")

    outfile.write("%i\n\n" % (len(elements)))

    for atomidx, atom in enumerate(coord):
        outfile.write("%s %f %f %f\n" % (elements[atomidx], atom[0], atom[1], atom[2]))

    outfile.close()

#---Tests--------------------------------------------------------------

def test_inv_dist(distmatrix_list):
    for matrix in distmatrix_list:
        recovered_coord = coordinates_from_distancematrix(matrix)
        inverse_distances_original = np.array(calculate_upper_triangular_matrix(matrix))

        inverse_distances = np.array(calculate_inv_dist_vector(recovered_coord))
        recover_distancematrix = distance_vector_to_distance_matrix(inverse_distances)


        assert((np.linalg.norm(matrix - recover_distancematrix)) <= 1e-9)
        #print(inverse_distances_original, inverse_distances)
        #print(inverse_distances - inverse_distances_original)
        #print(np.linalg.norm(inverse_distances - inverse_distances_original))
        assert((np.linalg.norm(inverse_distances - inverse_distances_original)) <= 1e-10)

def rmsd_test(coords):
    distmatrix_list = get_distmatrix_list(coords)
    methanole = ["C", "O", "H", "H", "H", "H"]
    recovered_coords = []
    for matrix in distmatrix_list:
        recovered = coordinates_from_distancematrix(matrix)
        recovered_coords.append(recovered)

    recovered_coords = np.array(recovered_coords)

    rmsd_list = []
    A_list = []
    B_list = []
    B_rot_list = []
    for i in range(len(coords)):
        A = coords[i]
        B = recovered_coords[i]

        # Manipulate
        #A -= rmsd.centroid(A)
        #B -= rmsd.centroid(B)

        #print("Translated RMSD", rmsd.rmsd(A, B))
        #save_plot(A, B, "plot_translated")

        #U = rmsd.kabsch(A, B)
        #A = np.dot(A, U)

        #print("Rotated RMSD", rmsd.rmsd(A, B))
        #save_plot(A, B, "plot_rotated")

        Bout, R, t = rigid_transform(A,B)
        #print(Bout)
        #print(R)

        rmsd_val = rmsd.rmsd(B, Bout)
        assert rmsd_val < 1e-14
        rmsd_list.append(rmsd_val)


        #print(Bout)
        #print(A)

        A_list.append(A)
        B_rot_list.append(Bout)
        B_list.append(B)
    #print(A.shape, B.shape)

    #write_xyz_single(B_list[-1], methanole, "recovered.xyz")
    #write_xyz_single(B_rot_list[-1], methanole, "recovered_and_rotated.xyz")
    #write_xyz_single(A_list[-1], methanole, "original.xyz")

    print(rmsd_list)

