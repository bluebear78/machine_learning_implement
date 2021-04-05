import numpy as np


def n_size_ndarray_creation(n, dtype=np.int):
    X = np.array(range(n**2)).reshape(n,-1)
    return X


def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    if type==0:
        make_matrix = np.zeros(shape,dtype=dtype)
    elif type==1:
        make_matrix = np.ones(shape,dtype=dtype)
    elif type==99:
        make_matrix = np.random.random_sample(shape)
    else:
        make_matrix = np.empty(shape)
    return make_matrix


def change_shape_of_ndarray(X, n_row):
    if n_row == 1:
        return X.flatten()
    return X.reshape(n_row,-1)


def concat_ndarray(X_1, X_2, axis_para):
    if X_1.ndim == 1:
        X_1 = X_1.reshape(1, -1)
    if X_2.ndim == 1:
        X_2 = X_2.reshape(1, -1)
    try:
        return np.concatenate((X_1,X_2), axis=axis_para)
    except ValueError as e:
        return False


def normalize_ndarray(X, axis=99, dtype=np.float32):
    if axis==0:
        X_mean = X.mean(axis=axis)
        std = X.std(axis=axis)
        return (X-X_mean) / std
    elif axis==1:
        means = X.mean(axis=axis).reshape(-1,1)
        std = X.std(axis=axis).reshape(-1,1)
        return (X-means) / std
    elif axis==99:
        return (X-X.mean()) / X.std()
    else:
        return False;


def save_ndarray(X, filename="test.npy"):
    np.save(filename,X)


def boolean_index(X, condition):
    return np.where(eval(str("X")+condition))



def find_nearest_value(X, target_value):
    return X.T[np.argmin(abs(X.reshape(1,-1) - target_value))]


def get_n_largest_values(X, n):
    return X[X.argsort()[::-1]][:n]


if __name__ == '__main__':
    print(type(n_size_ndarray_creation(4,np.int)))
