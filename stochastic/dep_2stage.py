import sys
import os
import networkx as nx
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

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.mixture import GaussianMixture
from scipy.stats import bernoulli
from scipy.stats import gaussian_kde
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
import scipy.stats as stats
# import random
from scipy.stats import gaussian_kde, multivariate_normal
import time
# import re
# from sklearn.neighbors import KernelDensity
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import preproc

# preproc.generate_scenarios()
# 최적화 모델 생성



def build_model(scenarios, probabilities, E_0):
    P_r = 80
    C = 18330.6
    M = 10**6

    model = pyo.ConcreteModel()

    # Parameters
    model.E_0 = pyo.Param(initialize=E_0, mutable=True)

    num_scenarios = len(scenarios)

    # First-stage decision variables
    model.b_da = pyo.Var(bounds=(-P_r, 0), domain=pyo.Reals, initialize = -P_r / 2)
    model.q_da = pyo.Var(domain=pyo.Reals, bounds=(0.9*E_0, 1.1*E_0), initialize = E_0)
    model.v_1 = pyo.Var(domain=pyo.Reals, bounds = (-10**2, 10**2 ))
    model.v_0 = pyo.Var(domain=pyo.Reals, bounds = (-10**2, 10**2))
    # Scenario-specific components
    model.scenarios = pyo.RangeSet(0, num_scenarios - 1)
    # model.probabilities = pyo.Set(initialize = probabilities)
    model.P_da = pyo.Param(model.scenarios, initialize={i: scenarios[i][0] for i in range(num_scenarios)})
    model.P_rt = pyo.Param(model.scenarios, initialize={i: scenarios[i][1] for i in range(num_scenarios)})
    model.E_1 = pyo.Param(model.scenarios, initialize={i: scenarios[i][2] for i in range(num_scenarios)})
    model.random_factor = pyo.Param(model.scenarios, initialize={i: scenarios[i][3] for i in range(num_scenarios)})
    model.prob = pyo.Param(model.scenarios, initialize={i: probabilities[i] for i in range(num_scenarios)})
    # Second-stage decision variables
    def q_rt_bounds(model, s):
        return model.E_1[s] #q_rt는 항상 E_1
    model.b_rt = pyo.Var(model.scenarios, bounds=(-P_r, 0), domain=pyo.Reals, initialize = - P_r / 2) #TODO: 여기 bounds 오른쪽에 b_da 넣어야하는거 아님?
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
    model.n1_V = pyo.Var(model.scenarios, domain=pyo.Binary) #TODO: 이렇게 하는게 아니라 그냥 벡터로 선언해서 indexing하면 되잖아.
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
    model.constrs = pyo.ConstraintList()
    def scenario_constraints(model, s):

        model.constrs.add(model.b_da - model.P_da[s] <= M * (1 - model.y_da[s]))
        model.constrs.add(model.q_da <= M * model.y_da[s])
        model.constrs.add(model.Q_da[s] <= model.q_da)
        model.constrs.add(model.q_da - M * (1 - model.y_da[s]) <= model.Q_da[s])
                                                          ## TODO: 이러면 만약 내가 q_da를 양수로 입찰하면 무조건 y_da = 1
                                                          ## 그러면 b_da <= P_da[s] for all s -> b_da <= minimum(P_da[s])= -77
        model.constrs.add(model.b_rt[s] - model.P_rt[s] <= M * (1 - model.y_rt[s])) 
        model.constrs.add(model.q_rt[s] <= M * model.y_rt[s])
        model.constrs.add(model.q_rt[s] - M * (1 - model.y_rt[s]) <= model.Q_rt[s])
        model.constrs.add(model.Q_rt[s] <= model.q_rt[s]) ## TODO: ???? 이러면 y_rt =0 -> E_1 = Q_rt <= q_rt = 0 돼서 절대 y_rt가 0이 안되는데
                                                          ## TODO: 애초에 y_rt가 제대로 된 logical implication 기능을 함?
        # Rule constraint
        model.constrs.add(model.Q_c[s] == model.random_factor[s] * model.Q_rt[s])
        model.constrs.add(model.b_rt[s] <= model.b_da)

        # f_V Linearization constraints
        model.constrs.add(model.S1_V[s] == model.b_rt[s] * model.m1_V[s] - model.Q_da[s] * model.P_da[s] - model.m1_V[s] * model.P_rt[s] + model.P_rt[s] * model.Q_da[s])
    
        model.constrs.add(model.m1_V[s] <= model.z[s])
        model.constrs.add(model.m1_V[s] <= model.Q_c[s]) #TODO: m1_V[s] <= model.q_rt[s]는? 
        model.constrs.add(model.m1_V[s] >= model.z[s] - M * (1 - model.n1_V[s])) 
        model.constrs.add(model.m1_V[s] >= model.Q_c[s] - M * model.n1_V[s]) #TODO: Q_c가 아니라 q_rt아니야? 그리고 binary 총 3개 아니야? 

        model.constrs.add(model.m2_V[s] >= model.S1_V[s]) #이거 이름을 꼭 m2_V로 해야 함? f_V로 하면 안됨?
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
  
        model.constrs.add(model.f_E[s] >= (model.P_rt[s] - model.b_da) * model.m1_E[s] - M * (1 - model.n3_E[s])) ##TODO: 이건 무슨 의미?
        model.constrs.add(model.f_E[s] <= (model.P_rt[s] - model.b_da) * model.m2_E[s] + M * model.n3_E[s]) ##TODO: 이건 무슨 의미?
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

    # Non-anticipativity constraints
    for i in range(num_scenarios):
        for j in range(i + 1, num_scenarios):
            if scenarios[i][0] == scenarios[j][0] and scenarios[i][2] == scenarios[j][2]:
                model.constrs.add(model.b_rt[i] == model.b_rt[j])
                model.constrs.add(model.q_rt[i] == model.q_rt[j])  #SW: 의미없음. 어차피 q_rt는 E_1 고정이니까 있으나마나한 제약식. 어차피 presolve단계에서 사라지긴할듯
    # Objective Function
    def objective_rule(model):
        return sum(model.prob[s]*((model.P_da[s] * model.Q_da[s] + model.P_rt[s] * (model.z[s] - model.Q_da[s])) + model.f_max[s] + (-model.m1_Im[s] * model.m4_Im[s]) + (model.z[s] * P_r)) for s in model.scenarios)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    return model
def calculate_profit(model, idx):
    return (
        pyo.value(model.P_da[idx]) * pyo.value(model.Q_da[idx]) +
        pyo.value(model.P_rt[idx]) * (pyo.value(model.z[idx]) - pyo.value(model.Q_da[idx])) +
        pyo.value(model.f_max[idx]) - (pyo.value(model.m1_Im[idx]) * pyo.value(model.m4_Im[idx])) +
        pyo.value(model.z[idx]) * P_r
    )
def solve_sp_model(scenarios, probabilities, new_E_0):
    model = build_model(scenarios, probabilities, new_E_0)
    model.E_0 = new_E_0
    solver = pyo.SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
    solver_result = solver.solve(model, tee=True)
    print('insample profit: ', pyo.value(model.objective))
    print('b_da: ', pyo.value(model.b_da))
    print('q_da: ', pyo.value(model.q_da))
    results = []
    for s in range(len(scenarios)):
        profit = calculate_profit(model, s)
        results.append({
            "b_da": pyo.value(model.b_da),
            "q_da": pyo.value(model.q_da),
            "b_rt": pyo.value(model.b_rt[s]),
            "q_rt": pyo.value(model.q_rt[s]),
            "z": pyo.value(model.z[s]),
            "Profit": profit,
            "prob": pyo.value(model.prob[s])
        })

    return model, pd.DataFrame(results), solver_result
def create_weighted_tree(W_1_red, W_2_red):
    G = nx.DiGraph()
    
    # 루트 노드 추가
    G.add_node('root')
    
    # 노드 라벨 딕셔너리 생성
    labels = {'root': 'root'}
    
    # 첫 번째 레벨 노드 생성 (20개)
    for i in range(len(W_1_red[0])):
        node_name = f'L1_{i}'
        node_values = [W_1_red[0][i], W_1_red[1][i]]
        G.add_node(node_name, values=node_values)
        G.add_edge('root', node_name)
        # 라벨에 값 추가 (소수점 1자리까지 표시)
        labels[node_name] = f'[{node_values[0]:.1f},\n{node_values[1]:.1f}]'
        
        # 각 첫 번째 레벨 노드에 대해 5개의 자식 노드 생성
        for j in range(len(W_2_red[0])):
            child_name = f'L2_{i}_{j}'
            child_values = [W_2_red[0][j], W_2_red[1][j]]
            G.add_node(child_name, values=child_values)
            G.add_edge(node_name, child_name)
            # 라벨에 값 추가
            labels[child_name] = f'{child_name}\n[{child_values[0]:.1f},\n{child_values[1]:.1f}]'
    
    plt.figure(figsize=(25, 15))
    
    # 수평 레이아웃을 위한 포지션 계산
    pos = {}
    
    # 루트 노드 위치 설정
    pos['root'] = np.array([0, 0])
    
    # 첫 번째 레벨 노드들의 위치 설정
    level1_y = np.linspace(-8, 8, len(W_1_red[0]))  # 20개 노드를 위한 y 좌표
    for i, y in enumerate(level1_y):
        pos[f'L1_{i}'] = np.array([2, y])
    
    # 두 번째 레벨 노드들의 위치 설정
    for i in range(len(W_1_red[0])):
        level2_y = np.linspace(level1_y[i] - 0.4, level1_y[i] + 0.4, 5)
        for j, y in enumerate(level2_y):
            pos[f'L2_{i}_{j}'] = np.array([4, y])
    
    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=2000)  # 노드 크기 증가
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, arrows=True)
    
    # 노드 라벨 그리기 (값 포함)
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    
    plt.title("Horizontal Weighted Tree Visualization with Values")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



new_E_0_values = [10000, 8000, 12000]

seed = 15

scenarios = preproc.regenerate_scenarios(new_E_0_values[1], sample_size=1000, seed =seed)



arrays = list(zip(*scenarios))
arrays = np.asarray(arrays)
arr1 = np.array(arrays[0]) #P da
arr2 = np.array(arrays[1]) #P rt
arr3 = np.array(arrays[2]) #E1 (Q da)
arr4 = np.array(arrays[3]) #Q3

prob = np.ones(len(scenarios))/len(scenarios)

# price_path = np.array([arr1, arr2])
# gen_path = np.array([arr3, arr4])
# S_price = scen_red.ScenarioReduction(price_path, probabilities=prob, cost_func='2norm', r = 2, scen0 = np.zeros(10))
# S_price.fast_forward_sel(n_sc_red=100, num_threads = 4)
# scen_price_red = S_price.scenarios_reduced  # get reduced scenarios
# prob_price_red = S_price.probabilities_reduced  # get reduced probabilities

optim_list = {}
for num in [20,30,40,50,60,70,90]:
    W_1 = np.array([arr1,arr3]) #idea: 일단 P_da, Q_da를 묶고 나머지 P_rt, Q_c는 각 분기마다 5개정도로?
    num_W_1 = num #최소 >=40 부터 덜 sensitive해지는듯
    S_1 = scen_red.ScenarioReduction(W_1, probabilities=prob, cost_func='2norm', r = 2, scen0 = np.zeros(10))
    S_1.fast_forward_sel(n_sc_red=num_W_1, num_threads = 4)
    W_1_red = S_1.scenarios_reduced  # get reduced scenarios
    prob_1_red = S_1.probabilities_reduced 


    W_2 = np.array([arr2,arr4]) #idea: 일단 P_da, Q_da를 묶고 나머지 P_rt, Q_c는 각 분기마다 5개정도로?
    num_W_2 = 5
    S_2 = scen_red.ScenarioReduction(W_2, probabilities=prob, cost_func='2norm', r = 2, scen0 = np.zeros(10))
    S_2.fast_forward_sel(n_sc_red=num_W_2, num_threads = 4)
    W_2_red = S_2.scenarios_reduced  # get reduced scenarios
    prob_2_red = S_2.probabilities_reduced 

    scenarios_red = []
    probabilities_red = []
    for i in range(num_W_1):
        for j in range(num_W_2):
            scenario = (W_1_red[0][i], W_2_red[0][j], W_1_red[1][i], W_2_red[1][j])
            scenarios_red.append(scenario)
            probability = prob_1_red[i]*prob_2_red[j]
            probabilities_red.append(probability)

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





    model, res, solver_result = solve_sp_model(scenarios_red, probabilities_red, new_E_0_values[1])
    optim_list[num]=(pyo.value(model.objective), np.round(float(solver_result.Solver[0]["Wall time"]),2))


tree = create_weighted_tree(W_1_red, W_2_red)

scenarios[1]
print(scenarios)

