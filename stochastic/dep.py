import pyomo.environ as pyo
import numpy as np
import pandas as pd
# import pyro
# import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import bernoulli
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import random
from scipy.stats import gaussian_kde, multivariate_normal
import time
import re
from sklearn.neighbors import KernelDensity
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import preproc

preproc.generate_scenarios()


