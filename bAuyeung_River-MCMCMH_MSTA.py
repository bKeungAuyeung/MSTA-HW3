# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(5) # set random seed
walkers = np.random.uniform(0, 2000, 50) # from 0 to 2000, pick 50 values
burn_in = 1000 # burn in of 1000 iterations
total_iterations = 2000 # 2000 iterations total. usable iterations = total iterations - burn_in

# we know that we intially have a flat prior and we assume
# the likelihood is a normal distribution. using the log-likelihood
# we can obtain the posterior distribution using the following

def log_likelihood(F_meas, F_infer):
    std = np.sqrt(F_meas) # assume std (i.e. errors) are sqrt(F_meas)
    return -0.5 * np.sum(np.log(2*np.pi*std**2) + ((F_meas - F_infer)/std)**2)

def log_posterior(F_meas, F_infer, prior):
    return log_likelihood(F_meas, F_infer) + prior