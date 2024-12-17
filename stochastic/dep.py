import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 부모의 부모 디렉토리 추가
import scenred.scenario_reduction as scen_red
import pyomo.environ as pyo
import numpy as np
import pandas as pd
# import pyro
# import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
# from pyro.optim import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import bernoulli
from scipy.stats import gaussian_kde
import seaborn as sns
# from sklearn.linear_model import LinearRegression
import scipy.stats as stats
# import random
from scipy.stats import gaussian_kde, multivariate_normal
import time
import re
from sklearn.neighbors import KernelDensity
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import preproc

# preproc.generate_scenarios()
# 최적화 모델 생성



def build_model(scenarios, E_0):
    model = pyo.ConcreteModel()

    # Parameters
    model.E_0 = pyo.Param(initialize=E_0, mutable=True)

    num_scenarios = len(scenarios)

    # First-stage decision variables
    model.b_da = pyo.Var(bounds=(-P_r, 0), domain=pyo.Reals, initialize = -P_r / 2)
    model.q_da = pyo.Var(domain=pyo.Reals, bounds=(0.9*E_0, 1.1*E_0), initialize = E_0)

    # Scenario-specific components
    model.scenarios = pyo.RangeSet(0, num_scenarios - 1)
    model.P_da = pyo.Param(model.scenarios, initialize={i: scenarios[i][0] for i in range(num_scenarios)})
    model.P_rt = pyo.Param(model.scenarios, initialize={i: scenarios[i][1] for i in range(num_scenarios)})
    model.E_1 = pyo.Param(model.scenarios, initialize={i: scenarios[i][2] for i in range(num_scenarios)})
    model.random_factor = pyo.Param(model.scenarios, initialize={i: scenarios[i][3] for i in range(num_scenarios)})

    # Second-stage decision variables
    def q_rt_bounds(model, s):
        return model.E_1[s]
    model.b_rt = pyo.Var(model.scenarios, bounds=(-P_r, 0), domain=pyo.Reals, initialize = - P_r / 2)
    model.q_rt = pyo.Var(model.scenarios, bounds=q_rt_bounds, domain=pyo.Reals, initialize=lambda model, s: model.E_1[s])

    # Scenario-specific binary variables
    model.y_da = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.y_rt = pyo.Var(model.scenarios, domain=pyo.Binary)

    # Scenario-specific analysis variables
    model.Q_da = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals, initialize= model.q_da / 2)
    model.Q_rt = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals, initialize=lambda model, s: model.q_rt[s] / 2)
    model.Q_c = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals, initialize=lambda model, s: model.random_factor[s])

    # Third-stage decision variables
    def z_bounds(model, s):
        return (0, model.E_1[s])
    model.z = pyo.Var(model.scenarios, bounds=z_bounds, domain=pyo.Reals, initialize=lambda model, s: model.E_1[s])

    # Linearization Real Variables
    model.m1_V = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m2_V = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m1_E = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m2_E = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m3_E = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m1_Im = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m2_Im = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m3_Im = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.m4_Im = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.S1_V = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.S1_E = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.S1_Im = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.S2_Im = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.f_E = pyo.Var(model.scenarios, domain=pyo.Reals)
    model.f_max = pyo.Var(model.scenarios, domain=pyo.Reals)

    # Linearization Binary Variables (Scenario-specific)
    model.n1_V = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n2_V = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n1_E = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n2_E = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n3_E = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n1_Im = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n2_Im = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n3_Im = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n4_Im = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n1_F = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n2_F = pyo.Var(model.scenarios, domain=pyo.Binary)
    model.n3_F = pyo.Var(model.scenarios, domain=pyo.Binary)

    # Constraints
    model.constrs = pyo.ConstraintList() ## SW: constrs list로 만들어서 넣으면 엄청 느린데..
    def scenario_constraints(model, s):

        model.constrs.add(model.b_da - model.P_da[s] <= M * (1 - model.y_da[s]))
        model.constrs.add(model.q_da <= M * model.y_da[s])
        model.constrs.add(model.Q_da[s] <= model.q_da)
        model.constrs.add(model.q_da - M * (1 - model.y_da[s]) <= model.Q_da[s])

        model.constrs.add(model.b_rt[s] - model.P_rt[s] <= M * (1 - model.y_rt[s]))
        model.constrs.add(model.q_rt[s] <= M * model.y_rt[s])
        model.constrs.add(model.q_rt[s] - M * (1 - model.y_rt[s]) <= model.Q_rt[s])
        model.constrs.add(model.Q_rt[s] <= model.q_rt[s])
 
        # Rule constraint
        model.constrs.add(model.Q_c[s] == model.random_factor[s] * model.Q_rt[s])
        model.constrs.add(model.b_rt[s] <= model.b_da)

        # f_V Linearization constraints
        model.constrs.add(model.S1_V[s] == model.b_rt[s] * model.m1_V[s] - model.Q_da[s] * model.P_da[s] - model.m1_V[s] * model.P_rt[s] + model.P_rt[s] * model.Q_da[s])
    
        model.constrs.add(model.m1_V[s] <= model.z[s])
        model.constrs.add(model.m1_V[s] <= model.Q_c[s])
        model.constrs.add(model.m1_V[s] >= model.z[s] - M * (1 - model.n1_V[s]))
        model.constrs.add(model.m1_V[s] >= model.Q_c[s] - M * model.n1_V[s])

        model.constrs.add(model.m2_V[s] >= model.S1_V[s])
        model.constrs.add(model.m2_V[s] >= 0)
        model.constrs.add(model.m2_V[s] <= model.S1_V[s] + M * (1 - model.n2_V[s]))
        model.constrs.add(model.m2_V[s] <= M * model.n2_V[s])

        # f_E Linearization constraints
        model.constrs.add(model.m1_E[s] <= model.Q_da[s])
        model.constrs.add(model.m1_E[s] <= model.q_rt[s])
        model.constrs.add(model.m1_E[s] >= model.Q_da[s] - M * (1 - model.n1_E[s]))
        model.constrs.add(model.m1_E[s] >= model.q_rt[s] - M * model.n1_E[s])
  
        model.constrs.add(model.m2_E[s] <= model.z[s])
        model.constrs.add(model.m2_E[s] <= model.q_rt[s])
        model.constrs.add(model.m2_E[s] >= model.z[s] - M * (1 - model.n2_E[s]))
        model.constrs.add(model.m2_E[s] >= model.q_rt[s] - M * model.n2_E[s])
  
        model.constrs.add(model.f_E[s] >= (model.P_rt[s] - model.b_da) * model.m1_E[s] - M * (1 - model.n3_E[s]))
        model.constrs.add(model.f_E[s] <= (model.P_rt[s] - model.b_da) * model.m2_E[s] + M * model.n3_E[s])
        model.constrs.add(model.f_E[s] == (model.P_rt[s] - model.b_da) * (model.m1_E[s] - model.m2_E[s]))
        
        # f_Im Linearization constraints
        model.constrs.add(model.S1_Im[s] == (model.z[s] - model.Q_c[s]) - 0.12 * C)

        model.constrs.add(model.S2_Im[s] == model.P_rt[s] - model.b_rt[s])

        model.constrs.add(model.m1_Im[s] >= model.S1_Im[s])
        model.constrs.add(model.m1_Im[s] >= 0)
        model.constrs.add(model.m1_Im[s] <= model.S1_Im[s] + M * (1 - model.n1_Im[s]))
        model.constrs.add(model.m1_Im[s] <= M * model.n1_Im[s])

        model.constrs.add(model.m2_Im[s] >= model.S2_Im[s])
        model.constrs.add(model.m2_Im[s] >= 0)
        model.constrs.add(model.m2_Im[s] <= model.S2_Im[s] + M * (1 - model.n2_Im[s]))
        model.constrs.add(model.m2_Im[s] <= M * model.n2_Im[s])
        model.constrs.add(model.m3_Im[s] >= -model.b_rt[s])
        model.constrs.add(model.m3_Im[s] >= 0)
        model.constrs.add(model.m3_Im[s] <= -model.b_rt[s] + M * (1 - model.n3_Im[s]))
        model.constrs.add(model.m3_Im[s] <= M * model.n3_Im[s])
        model.constrs.add(model.m4_Im[s] >= model.m3_Im[s])
        model.constrs.add(model.m4_Im[s] >= model.m2_Im[s])
        model.constrs.add(model.m4_Im[s] <= model.m3_Im[s] + M * (1 - model.n4_Im[s]))
        model.constrs.add(model.m4_Im[s] <= model.m2_Im[s] + M * model.n4_Im[s])

        # f_max linearization constraints
        model.constrs.add(model.f_max[s] >= model.m2_V[s])
        model.constrs.add(model.f_max[s] >= model.f_E[s])
        model.constrs.add(model.f_max[s] >= 0)
        model.constrs.add(model.f_max[s] <= model.m2_V[s] + M * (1 - model.n1_F[s]))
        model.constrs.add(model.f_max[s] <= model.f_E[s] + M * (1 - model.n2_F[s]))
        model.constrs.add(model.f_max[s] <= M * (1 - model.n3_F[s]))
        model.constrs.add(model.n1_F[s] + model.n2_F[s] + model.n3_F[s] == 1)
        
    for s in model.scenarios:
        scenario_constraints(model, s)

    nonanti_scen_list = []
    # Non-anticipativity constraints
    for i in range(num_scenarios):
        for j in range(i + 1, num_scenarios):
            if scenarios[i][0] == scenarios[j][0] and scenarios[i][2] == scenarios[j][2]:
                model.constrs.add(model.b_rt[i] == model.b_rt[j])
                model.constrs.add(model.q_rt[i] == model.q_rt[j])  
                nonanti_scen_list.append(i)
    model.nonanti_scen = pyo.Set(initialize=nonanti_scen_list) #model에 저장
    # Objective Function
    def objective_rule(model):
        return sum((model.P_da[s] * model.Q_da[s] + model.P_rt[s] * (model.z[s] - model.Q_da[s])) + model.f_max[s] + (-model.m1_Im[s] * model.m4_Im[s]) + (model.z[s] * P_r) for s in model.scenarios)/len(model.scenarios)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    return model
def calculate_profit(model, idx):
    return (
        pyo.value(model.P_da[idx]) * pyo.value(model.Q_da[idx]) +
        pyo.value(model.P_rt[idx]) * (pyo.value(model.z[idx]) - pyo.value(model.Q_da[idx])) +
        pyo.value(model.f_max[idx]) - (pyo.value(model.m1_Im[idx]) * pyo.value(model.m4_Im[idx])) +
        pyo.value(model.z[idx]) * P_r
    )
def solve_sp_model(scenarios, new_E_0):
    model = build_model(scenarios, new_E_0)
    model.E_0 = new_E_0
    solver = pyo.SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
    solver.solve(model, tee=True)
    print('insample profit: ', pyo.value(model.objective))
    print('b_da: ', pyo.value(model.b_da))
    print('q_da: ', pyo.value(model.q_da))
    results = []
    for s in range(len(scenarios)):
        profit = calculate_profit(model, s)
        results.append({
            # "b_da": pyo.value(model.b_da),
            # "q_da": pyo.value(model.q_da),
            "b_rt": pyo.value(model.b_rt[s]),
            "q_rt": pyo.value(model.q_rt[s]),
            "z": pyo.value(model.z[s]),
            "Profit": profit
        })

    return model, pd.DataFrame(results)

P_r = 80
C = 18330.6
M = 10**6

new_E_0_values = [10000, 8000, 12000]

scenarios = preproc.regenerate_scenarios(new_E_0_values[1], sample_size=100)
for scen in scenarios:
    print(scen)


arrays = list(zip(*scenarios))

arr1 = list(np.array(arrays[0]))
arr3 = list(np.array(arrays[2]))

W2 = [arr1, arr3]
probabilities = np.ones(len(scenarios))/len(scenarios)
S = scen_red.ScenarioReduction(W2, probabilities=probabilities, cost_func='2norm', r = 2, scen0 = np.zeros(2))


'''
# scenarios2 = np.random.rand(10,30)  # Create 30 random scenarios of length 10. 
### Seokwoo: scenarios의 길이는 10개고 각 array에 30개 scenario 존재. 
### 즉, 시나리오 1은 scenarios2[:,0], 시나리오 2는 scenarios2[:,1]을 의미
probabilities = np.random.rand(30)
probabilities = probabilities/np.sum(probabilities)  # Create random probabilities of each scenario and normalize 

S = scen_red.ScenarioReduction(scenarios2, probabilities=probabilities, cost_func='2norm', r = 2, scen0 = np.zeros(10))
S.fast_forward_sel(n_sc_red=5, num_threads = 4)  # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads 
scenarios_reduced = S.scenarios_reduced  # get reduced scenarios
probabilities_reduced = S.probabilities_reduced  # get reduced probabilities
'''





model, res = solve_sp_model(scenarios, new_E_0_values[1])

scenarios[1]
print(scenarios)

