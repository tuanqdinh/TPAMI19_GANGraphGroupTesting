
import numpy as np
import scipy
from scipy import stats

def eval_basic_tester(X, Y):
    n = np.shape(X)[0]
    np.random.shuffle(X)

########### T-test ##################################
def get_p_value(s1, s2):
    # extract s region from images
    # s1: 1000 x n
    n_pixels = np.shape(s1)[1]
    p_values = np.zeros(n_pixels)

    for j in range(n_pixels):
        a1 = [s1[i][j] for i in np.arange(np.shape(s1)[0])]
        a2 = [s2[i][j] for i in np.arange(np.shape(s2)[0])]
        s, p = scipy.stats.ttest_ind(a1, a2)
        p_values[j] = p
        # if j == 0:
            # from IPython import embed; embed()
    return p_values

def test_group():
    nrows = 1000
    real_1 = generate_test_data(1, im_size, nrows)
    real_2 = generate_test_data(2, im_size, nrows)
    p_values_real = get_t_statistic(real_1, real_2, im_size, nrows)
    syn_1 = np.load('test_syn_group_1.npy')
    syn_2 = np.load('test_syn_group_2.npy')
    p_values_syn = get_t_statistic(syn_1, syn_2, im_size, nrows)
    plt.figure(0)
    plt.scatter(np.arange(len(p_values_syn)), p_values_real, c='b')
    plt.scatter(np.arange(len(p_values_syn)), p_values_syn, c='g')
    plt.show()
