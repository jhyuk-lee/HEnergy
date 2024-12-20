﻿import pyomo.environ as pyo
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

# Day-ahead price
def process_da_file(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[3:27, 2].tolist()
    return data

# Real-time price
def process_rt_file(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[3:99, 2].values.reshape(-1, 4).mean(axis=1)
    return data.tolist()


# 가격 시나리오 생성 함수
def generate_price_scenarios(filtered_df, N_price, seed=None):
    np.random.RandomState(seed)
    np.random.RandomState(seed)
    P_da_distribution = filtered_df['P(da)'].tolist()
    P_rt_distribution = filtered_df['P(rt)'].tolist()

    P_da_samples = np.random.choice(P_da_distribution, N_price)
    P_rt_samples = np.random.choice(P_rt_distribution, N_price)

    P_rt_final = []
    P_da_final = []

    for P_da, P_rt in zip(P_da_samples, P_rt_samples):
        if P_rt <= P_da:
            P_rt_final.append(P_rt)
            P_da_final.append(P_da)
        else:
            P_rt_final.append(P_da)
            P_da_final.append(P_da)

    return P_da_final, P_rt_final

# 발전량 시나리오 생성 함수
def generate_E_1_scenarios(N_E1, kde, E_0, E_0_values, E_1_values,seed=None):
    np.random.RandomState(seed) # seed 고정
    E_0_min, E_0_max = E_0_values.min(), E_0_values.max()
    E_0_range = E_0_max - E_0_min 

    correlation = np.corrcoef(E_0_values.ravel(), E_1_values.ravel())[0, 1]
    alpha = 0.1 * E_0_range 
    tolerance = alpha * (1 - abs(correlation))
    
    mask = (E_0_values.ravel() >= E_0 - tolerance) & (E_0_values.ravel() <= E_0 + tolerance)
    E_1_subset = E_1_values[mask].reshape(-1, 1)
    
    kde_conditional = KernelDensity(kernel='gaussian', bandwidth=5)
    kde_conditional.fit(E_1_subset)
    
    E_1_scenarios = kde_conditional.sample(N_E1).ravel()
    return E_1_scenarios


# 급전지시량 시나리오 생성 함수
def generate_random_factor_scenarios(N_Qc=10, mean=1.0, std_dev=0.2, seed=None):
    np.random.RandomState(seed)
    num_ones = int(0.9 * N_Qc)
    ones = [1.0] * num_ones

    num_rest = N_Qc - num_ones
    discrete_values = [0.2 + 0.2 * i for i in range(10)] 
    if num_rest > 0:
        weights = np.exp(-((np.array(discrete_values) - mean) ** 2) / (2 * std_dev ** 2))
        weights /= weights.sum()
        rest = np.random.choice(discrete_values, size=num_rest, p=weights, replace=True)
    else:
        rest = []

    scenarios = ones + rest.tolist()
    np.random.shuffle(scenarios) 
    return scenarios

# 시나리오 생성 함수

def generate_scenarios_with_streaming(filtered_df, N_E1, N_price, kde, E_0, E_0_values, E_1_values, sample_size=10000):
    """
    스트리밍 방식으로 시나리오를 생성.

    Args:
        N_E1: E_1 시나리오의 개수
        N_price: 가격 시나리오의 개수
        kde: KDE 객체
        E_0: 고정된 E_0 값
        sample_size: 샘플링할 시나리오 개수

    Returns:
        샘플링된 시나리오 리스트 [(P_da, P_rt, E_1, random_factor), ...]
    """

    P_da_samples, P_rt_samples = generate_price_scenarios(filtered_df, N_price, seed)

    E_1_scenarios = generate_E_1_scenarios(N_E1, kde, E_0, E_0_values, E_1_values, seed)
    
    random_factors = generate_random_factor_scenarios(seed)

    all_scenarios = itertools.product(P_da_samples, P_rt_samples, E_1_scenarios, random_factors)

    # 샘플링
    reservoir = []
    for idx, scenario in enumerate(all_scenarios):
        if len(reservoir) < sample_size:
            reservoir.append(scenario)
        # else:
        #     replace_idx = random.randint(0, idx)
        #     if replace_idx < sample_size:
        #         reservoir[replace_idx] = scenario

    return reservoir

def regenerate_scenarios_with_streaming(filtered_df, kde, new_E_0, E_0_values, E_1_values, N_price, N_E1, sample_size, seed):
    
    P_da_samples, P_rt_samples = generate_price_scenarios(filtered_df, N_price, seed)

    E_1_scenarios = generate_E_1_scenarios(N_E1, kde, new_E_0, E_0_values, E_1_values, seed)

    random_factors = generate_random_factor_scenarios(seed=seed)
    random_factors = np.random.choice(random_factors, sample_size) #sample size에 맞추려고 무작위로 계속 뽑음

    # da_indices = np.random.randint(0, N_price, sample_size)
    # rt_indices = np.random.randint(0, N_price, sample_size)
    # e1_indices = np.random.randint(0, N_E1, sample_size)
    # all_scenarios = itertools.product(P_da_samples, P_rt_samples, E_1_scenarios, random_factors)

    reservoir = [(P_da_samples[i], P_rt_samples[i],
                  E_1_scenarios[i], random_factors[i]) for i in range(sample_size)]
    
    # for idx, scenario in enumerate(all_scenarios):
    #     if len(reservoir) < sample_size:
    #         reservoir.append(scenario)
    #     else:
    #         # replace_idx = random.randint(0, idx)
    #         # if replace_idx < sample_size:
    #         #     reservoir[replace_idx] = scenario
    #         pass

    return reservoir




def regenerate_scenarios(new_E_0, sample_size, save=False, seed=None):
    curr_dir = Path(__file__).parent
    project_dir = curr_dir.parent
    target_dir = project_dir / '모의 실시간시장 가격'
    directory_path_da = str(target_dir / '하루전')
    directory_path_rt = str(target_dir / '실시간 확정')


    # print(f"Current directory: {curr_dir}")
    # print(f"Project directory: {project_dir}")
    # print(f"Target directory: {target_dir}")



    csv_files_da = [f for f in os.listdir(directory_path_da) if f.endswith('.csv')]
    csv_files_rt = [f for f in os.listdir(directory_path_rt) if f.endswith('.csv')]

    ### 일부 컴퓨터에선 csv_files 리스트가 날짜순으로 자동 정렬되지 않음
    ### 따라서 아래 match를 da_file 기준으로 할 때 rt_file이 동일한 날짜 페어가 선택되지 않음.
    ### da_file은 3/1일인데 rt_file은 3/29일이 선택되는 버그
    csv_files_da = sorted(csv_files_da, key=lambda x: datetime.strptime(x.split('_')[1], '%Y%m%d'))
    csv_files_rt = sorted(csv_files_rt, key=lambda x: datetime.strptime(x.split('_')[1], '%Y%m%d'))


    data = []

    for da_file, rt_file in zip(csv_files_da, csv_files_rt):
        match = re.search(r'(\d{8})', da_file)
        if match:
            date_str = match.group(1)
            date = datetime.strptime(date_str, '%Y%m%d')

            day_ahead_data = process_da_file(os.path.join(directory_path_da, da_file))
            real_time_data = process_rt_file(os.path.join(directory_path_rt, rt_file))

            for hour in range(24):
                timestamp = date + timedelta(hours=hour)
                data.append({'P(da)': day_ahead_data[hour], 'P(rt)': real_time_data[hour], 'timestamp': timestamp})

    original_price_df = pd.DataFrame(data)
    original_price_df = original_price_df.sort_values('timestamp',ascending=True).reset_index(drop=True)
    # global original_price_df
    # print(original_price_df)


    E_1_df = pd.read_csv(str(curr_dir) + '/jeju_estim.csv')
    E_1_df['timestamp'] = pd.to_datetime(E_1_df['timestamp'])
    E_1_df['timestamp'] = E_1_df['timestamp'].dt.tz_localize(None)
    E_1_df.rename(columns={'forecast_da': 'E_0', 'forecast_rt': 'E_1', 'timestamp': 'date'}, inplace=True)

    E_1_df = (E_1_df.set_index('date').resample('H').sum().reset_index())

    # E_1_df


    # 시간대 6시 - 18시, 날짜 3월 ~ 현재

    E_1_filtered = E_1_df.loc[(E_1_df['date'].dt.hour >= 6) & (E_1_df['date'].dt.hour <= 18)].copy()
    P_da_filtered = original_price_df.loc[(original_price_df['timestamp'].dt.hour >= 6) & 
                                        (original_price_df['timestamp'].dt.hour <= 18) & 
                                        (original_price_df['timestamp'].dt.minute == 0)].copy()
    P_rt_filtered = original_price_df.loc[(original_price_df['timestamp'].dt.hour >= 6) & 
                                        (original_price_df['timestamp'].dt.hour <= 18)].copy()

    E_1_filtered['date'] = pd.to_datetime(E_1_filtered['date'])
    E_1_filtered['hour'] = E_1_filtered['date'].dt.hour

    P_da_filtered['date'] = pd.to_datetime(P_da_filtered['timestamp'])
    P_da_filtered['hour'] = P_da_filtered['timestamp'].dt.hour

    P_rt_filtered['date'] = pd.to_datetime(P_rt_filtered['timestamp'])
    P_rt_filtered['hour'] = P_rt_filtered['timestamp'].dt.hour

    start_date = '2024-03-01'
    end_date = '2024-11-25'

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    E_1_filtered = E_1_filtered[(E_1_filtered['date'] >= start_date) & (E_1_filtered['date'] <= end_date)]
    P_da_filtered = P_da_filtered[(P_da_filtered['date'] >= start_date) & (P_da_filtered['date'] <= end_date)]
    P_rt_filtered = P_rt_filtered[(P_rt_filtered['date'] >= start_date) & (P_rt_filtered['date'] <= end_date)]

    merged_df = pd.merge(E_1_filtered, P_da_filtered[['date', 'hour', 'P(da)']], on=['date', 'hour'], how='left')
    merged_df = pd.merge(merged_df, P_rt_filtered[['date', 'hour', 'P(rt)']], on=['date', 'hour'], how='left')
    merged_df.fillna(0, inplace=True)

    merged_df['date'] = merged_df.apply(lambda row: row['date'].replace(hour=row['hour']), axis=1)
    merged_df.drop(columns=['hour'], inplace=True)

    # merged_df
 
    filtered_df = merged_df[(merged_df['P(da)'] < 0) & 
                            (merged_df['P(rt)'] < 0) & 
                            (merged_df['P(da)'] > merged_df['P(rt)'])]

    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.index = filtered_df.index + 1

    # filtered_df




    # E_0 기반 E_1 conditional KDE
    E_0_values = merged_df['E_0'].values.reshape(-1, 1)
    E_1_values = merged_df['E_1'].values

    kde = KernelDensity(kernel='gaussian', bandwidth=10)
    kde.fit(np.hstack([E_0_values, E_1_values.reshape(-1, 1)]))

    E_0_range = np.linspace(E_0_values.min(), E_0_values.max(), 100)
    E_1_range = np.linspace(E_1_values.min(), E_1_values.max(), 100)
    E_0_grid, E_1_grid = np.meshgrid(E_0_range, E_1_range)
    grid_points = np.vstack([E_0_grid.ravel(), E_1_grid.ravel()]).T

    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(E_0_grid.shape)


    print("sampling 시작")
    start_time = time.time()

    # 전체 시나리오 생성
    
    if seed == None:
        seed = 42
    np.random.seed(seed)
    N_E1 = sample_size
    N_price = sample_size
    
    total_sample_size = sample_size
    E_0 = new_E_0

    if save== False:
        regen_scenarios = regenerate_scenarios_with_streaming(filtered_df, kde, new_E_0=E_0, E_0_values=E_0_values, E_1_values=E_1_values,
                                                            N_price=N_price, N_E1=N_E1, sample_size=total_sample_size, seed=seed)
        end_time = time.time()
        execution_time = end_time - start_time

        # 실행 시간 출력
        # print(f"Total Execution Time: {execution_time:.2f} seconds")
        # print()   
    else:
        all_scenarios = generate_scenarios_with_streaming(filtered_df, N_E1, N_price, kde, E_0, E_0_values, E_1_values, sample_size=total_sample_size)
        end_time = time.time()
        execution_time = end_time - start_time

        # 실행 시간 출력
        # print(f"Total Execution Time: {execution_time:.2f} seconds")
        # print()  
        target_dir = curr_dir / 'mc_scenarios'
        mc_path = os.path.join(target_dir, 'mc_scenarios')
        all_path = os.path.join(target_dir, 'all_scenarios')
        with open('{}.npy'.format(mc_path), 'wb') as f:
            np.save(f, mc_scenarios)
        with open('{}.npy'.format(all_path), 'wb') as f:
            np.save(f, all_scenarios)

    return regen_scenarios
    # print("Sampled MC Scenarios (Top 10):")
    # for i, scenario in enumerate(mc_scenarios[:10], 1):
    #     print(f"MC Scenario {i}:  P_da = {scenario[0]:.2f}, P_rt = {scenario[1]:.2f}, E_1 = {scenario[2]:.2f}, Random Factor = {scenario[3]:.2f}")
if __name__ == '__main__':
    # 이 파일을 직접 실행할 때만 실행되는 코드
    regenerate_scenarios(10000, sample_size=40000, seed = 42)