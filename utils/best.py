# %matplotlib inline
import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

'''
    @input: 2 groups
    @output: dist of difference and hpd
'''
def best(y1, y2, label1='group1', label2='group2'):
    y = pd.DataFrame(dict(value=np.r_[y1, y2], group=np.r_[[label1]*len(drug), [label2]*len(placebo)]))
    y.hist('value', by='group');

    ##-------PRIORS- specified our probabilistic model
    # Means
    mu_m = y.value.mean()
    mu_s = y.value.std() * 2 # only here is different
    with pm.Model() as model:
        group1_mean = pm.Normal('group1_mean', mu_m, sd=mu_s)
        group2_mean = pm.Normal('group2_mean', mu_m, sd=mu_s)
    # Stds
    sigma_low = 1
    sigma_high = 10
    with model:
        group1_std = pm.Uniform('group1_std', lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform('group2_std', lower=sigma_low, upper=sigma_high)
    # Nu
    with model:
        nu = pm.Exponential('nu_minus_one', 1/29.) + 1
    # pm.kdeplot(np.random.exponential(30, size=10000), shade=0.5);

    with model:
        lam1 = group1_std**-2
        lam2 = group2_std**-2
        group1 = pm.StudentT(label1, nu=nu, mu=group1_mean, lam=lam1, observed=y1)
        group2 = pm.StudentT(label2, nu=nu, mu=group2_mean, lam=lam2, observed=y2)

    ####---- calculating the comparisons of interest
    with model:
        diff_of_means = pm.Deterministic('difference of means', group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic('difference of stds', group1_std - group2_std)
        effect_size = pm.Deterministic('effect size',
                                       diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))
   ####---- Fit Model
    with model:
        #M.sample(iter=110000, burn=10000)
       trace = pm.sample(2000, cores=2)

    #### Summary
    def trace_sd(x):
        return pd.Series(np.std(x, 0), name='sd')
def trace_quantiles(x):
    return pd.DataFrame(pm.quantiles(x, [5, 50, 95]))
t = pm.summary(trace, varnames=['mu'], stat_funcs=[trace_quantiles])
    return pm.summary(trace,varnames=['difference of means'])
    # pm.summary(trace,varnames=['difference of means', 'difference of stds', 'effect size'])

def plot(pm, trace):
    #### Plot the stochastic parameters of the model
     pm.plot_posterior(trace, varnames=['group1_mean','group2_mean', 'group1_std', 'group2_std', 'nu_minus_one'], color='#87ceeb');
     pm.plot_posterior(trace, varnames=['difference of means','difference of stds', 'effect size'], ref_val=0, color='#87ceeb');

     ####  plots the potential scale reduction parameter
     pm.forestplot(trace, varnames=['group1_mean', 'group2_mean']);
     pm.forestplot(trace, varnames=['group1_std', 'group2_std', 'nu_minus_one']);
     plt.show()
     # save figure

if __name__ == '__main__':
    # data preparation
    drug = (101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
            109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
            96,103,124,101,101,100,101,101,104,100,101)
    placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
               104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
               101,100,99,101,100,102,99,100,99)

    y1 = np.array(drug)
    y2 = np.array(placebo)
    t = best(y1, y2) #are (mean, sd, mc_error, hpd_2.5, hpd_97.5, n_eff and Rhat.)
    var = 'hpd_2.5'; v = t[var].values[0]
