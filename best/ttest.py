import numpy as np
from numpy import linalg as LA

def generate_group1(n):
    # image from distribution 1
    a = np.zeros((n, n))
    n_2 = int(n/2)
    n_4 = int(n/4)

    s1 = np.random.normal(0.25, 0.1, (n_2, n_2))
    for i in range(n_2):
        for j in range(n_2):
            a[i + n_4][j + n_4] = s1[i][j]
    return a

def generate_group2(n):
    # image from distribution 1
    a = generate_group1(n)
    n_4 = int(n/4)
    n_8 = int(n/8)
    s2 = np.random.normal(0.75, 0.15, (n_4, n_4))
    for i in range(n_4):
        for j in range(n_4):
            a[i + n_4 + n_8][j + n_8 + n_4] = s2[i][j]
    return a

def generate_test_data(option, n, batch_size):
    if option == 1:
        data = [np.matrix.flatten(generate_group1(n)) for i in range(batch_size)]
    else:
        data = [np.matrix.flatten(generate_group2(n)) for i in range(batch_size)]
    return np.asarray(data)

def get_t_statistic(set1, set2, n, nrows):
    s1 = []
    s2 = []
    for i in range(len(set1)):
        s1.append(extract_region(set1[i], n))
        s2.append(extract_region(set2[i], n))
    return get_p_value(np.asarray(s1), np.asarray(s2), nrows)

def extract_region(a, n):
    n_4 = int(n/2)
    n_8 = int(n/4)
    s = np.zeros(n_4**2)
    a = np.reshape(a, (n, n))
    for i in range(n_4):
        for j in range(n_4):
            # s[i * n_4 + j] = a[i + n_4 + n_8][j + n_8 + n_4]
            s[i * n_4 + j] = a[i + n_8][j + n_8]
    return s


########### T-test ##################################
def get_p_value_old(s1, s2, nrows):
    # extract s region from images
    # s1: 1000 x n
    n_pixels = np.shape(s1)[1]
    p_values = np.zeros(n_pixels)

    for j in range(n_pixels):
        a1 = [s1[i][j] for i in range(nrows)]
        a2 = [s2[i][j] for i in range(nrows)]
        s, p = scipy.stats.ttest_ind(a1, a2)
        p_values[j] = p
        # if j == 0:
            # from IPython import embed; embed()
    return p_values
