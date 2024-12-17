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
    if seed:
        np.random.seed(seed)
    
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
def generate_E_1_scenarios(N_E1, kde, E_0, E_0_values, E_1_values):
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
def generate_random_factor_scenarios(N_Qc=10, mean=1.0, std_dev=0.2):

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

    P_da_samples, P_rt_samples = generate_price_scenarios(filtered_df, N_price)

    E_1_scenarios = generate_E_1_scenarios(N_E1, kde, E_0, E_0_values, E_1_values)
    
    random_factors = generate_random_factor_scenarios()

    all_scenarios = itertools.product(P_da_samples, P_rt_samples, E_1_scenarios, random_factors)

    # 샘플링
    reservoir = []
    for idx, scenario in enumerate(all_scenarios):
        if len(reservoir) < sample_size:
            reservoir.append(scenario)
        else:
            replace_idx = random.randint(0, idx)
            if replace_idx < sample_size:
                reservoir[replace_idx] = scenario

    return reservoir

def generate_scenarios():
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
                # print(data)

    original_price_df = pd.DataFrame(data)
    # global original_price_df
    print(original_price_df)


    E_1_df = pd.read_csv(str(curr_dir) + '\\jeju_estim.csv')
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
    """
    # Plot histograms for P(da) and P(rt) in merged_df
    columns_to_plot = ['P(da)', 'P(rt)']
    colors = ['blue', 'orange']  # Assigning specific colors

    plt.figure(figsize=(12, 6))

    for i, (col, color) in enumerate(zip(columns_to_plot, colors)):
        plt.subplot(1, 2, i + 1)
        plt.hist(merged_df[col], bins=30, alpha=0.7, color=color, edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    """

    """
    #상관관계 확인
    correlation_matrix = merged_df[['P(da)', 'P(rt)', 'E_1']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()
    """

    filtered_df = merged_df[(merged_df['P(da)'] < 0) & 
                            (merged_df['P(rt)'] < 0) & 
                            (merged_df['P(da)'] > merged_df['P(rt)'])]

    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.index = filtered_df.index + 1

    # filtered_df


    """
    # 산점도 그리기
    plt.figure(figsize=(8, 6))
    plt.scatter(E_0_values, E_1_values, alpha=0.7, c='blue', edgecolor='k')
    plt.title("Scatter Plot of E_0 and E_1")
    plt.xlabel("E_0")
    plt.ylabel("E_1")
    plt.grid(True)
    plt.show()
    """

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

    # plt.figure(figsize=(8, 6))
    # plt.contourf(E_0_grid, E_1_grid, density, levels=50, cmap='viridis')
    # plt.colorbar(label="Density")
    # plt.scatter(E_0_values, E_1_values, c='red', s=10, label="Original Data")
    # plt.title("KDE Multivariate Density")
    # plt.xlabel("E_0")
    # plt.ylabel("E_1")
    # plt.legend()
    # plt.show()
    #상관관계 고려한 샘플링. 사용 X

    """
    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde, multivariate_normal

    # 시나리오 생성 함수 정의
    def generate_scenarios(seed, N, proposal_std=0.5):
        np.random.seed(seed)
        
        scenarios = []

        kde = gaussian_kde(merged_df[['E_1', 'P(da)', 'P(rt)']].T)
        cov_matrix = np.cov(merged_df[['E_0', 'E_1']].T)
        mean_E0 = merged_df['E_0'].mean()
        mean_E1 = merged_df['E_1'].mean()
        cov_E0E1 = cov_matrix[0, 1] / cov_matrix[0, 0]
        cond_mean_E1 = mean_E1 + cov_E0E1 * (E_0 - mean_E0)
        cond_var_E1 = cov_matrix[1, 1] - cov_E0E1 * cov_matrix[0, 1]

        current_E_1 = np.random.normal(cond_mean_E1, np.sqrt(cond_var_E1) + proposal_std) 
        current_P_da, current_P_rt = kde.resample(1).flatten()[1:]
        current_sample = np.array([current_E_1, current_P_da, current_P_rt])

        while len(scenarios) < N:
            proposed_P_da, proposed_P_rt = kde.resample(1).flatten()[1:]
            proposed_E_1 = np.random.normal(cond_mean_E1, np.sqrt(cond_var_E1) + proposal_std)
            proposed_sample = np.array([proposed_E_1, proposed_P_da, proposed_P_rt])
            
            # Metropolis-Hastings 알고리즘
            current_density = kde.pdf(current_sample) * multivariate_normal.pdf(current_sample[0], mean=cond_mean_E1, cov=cond_var_E1)
            proposed_density = kde.pdf(proposed_sample) * multivariate_normal.pdf(proposed_sample[0], mean=cond_mean_E1, cov=cond_var_E1)
            acceptance_ratio = min(1, proposed_density / current_density)

            if np.random.rand() < acceptance_ratio:
                current_sample = proposed_sample
                current_E_1, current_P_da, current_P_rt = proposed_E_1, proposed_P_da, proposed_P_rt

            if np.random.rand() < 0.05: 
                random_factor = np.random.uniform(0, 2)
            else: 
                random_factor = 1.0

            scenarios.append((current_P_da, current_P_rt, current_E_1, random_factor))

            scenarios_df = pd.DataFrame(scenarios, columns=['P_da', 'P_rt', 'E_1', 'random_factor'])
            scenarios_df.drop_duplicates(subset=['P_da', 'P_rt', 'E_1'], keep='first', inplace=True)
            scenarios = scenarios_df.values.tolist()

        return scenarios[:N]

    E_0 = merged_df['E_0'].mean() 
    scenarios = generate_scenarios(seed=42, N=100, proposal_std=0.5) 
    scenarios_df = pd.DataFrame(scenarios, columns=['P_da', 'P_rt', 'E_1', 'random_factor'])

    print(scenarios_df.head())
    """

    """
    # 가격 figure 정리
    N_price = 1000
    seed = 42
    P_da_samples, P_rt_samples = generate_price_scenarios(filtered_df, N_price, seed=seed)

    sampled_prices = pd.DataFrame({
        'P(da)': P_da_samples,
        'P(rt)': P_rt_samples
    })

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(P_da_samples, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Sampled P(da) Distribution")
    plt.xlabel("P(da) (Day-Ahead Price)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(P_rt_samples, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title("Sampled P(rt) Distribution")
    plt.xlabel("P(rt) (Real-Time Price)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    """

    """
    # 급전지시량 figure 정리
    random_factors = generate_random_factor_scenarios(N_Qc=1000)  

    discrete_values = [0.2 + 0.2 * i for i in range(10)] 
    counts = [random_factors.count(val) for val in discrete_values]
    probabilities = [count / len(random_factors) for count in counts] 

    plt.figure(figsize=(12, 6))
    plt.bar(discrete_values, probabilities, width=0.1, color="skyblue", edgecolor="black", align='center')

    plt.title("Random factor Scenarios Distribution", fontsize=16)
    plt.xlabel("Random factor", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.xticks(discrete_values, [f"{val:.1f}" for val in discrete_values], fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    """



    start_time = time.time()

    # 전체 시나리오 생성
    seed = 42
    random.seed(seed)
    N_E1 = 200
    N_price = 200
    total_sample_size = 400000  # 전체 시나리오 개수

    E_0 = 10000  # E_0 값

    all_scenarios = generate_scenarios_with_streaming(filtered_df, N_E1, N_price, kde, E_0, E_0_values, E_1_values, sample_size=total_sample_size)

    # 샘플링 (MC Simulation용)
    num_mc_scenarios = 10000
    mc_scenarios = random.sample(all_scenarios, num_mc_scenarios)

    end_time = time.time()
    execution_time = end_time - start_time

    # 실행 시간 출력
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    # print()

    target_dir = curr_dir / 'mc_scenarios'
    mc_path = os.path.join(target_dir, 'mc_scenarios')
    all_path = os.path.join(target_dir, 'all_scenarios')

    with open('{}.npy'.format(mc_path), 'wb') as f:
        np.save(f, mc_scenarios)
    with open('{}.npy'.format(all_path), 'wb') as f:
        np.save(f, all_scenarios)


    # print("Sampled MC Scenarios (Top 10):")
    # for i, scenario in enumerate(mc_scenarios[:10], 1):
    #     print(f"MC Scenario {i}:  P_da = {scenario[0]:.2f}, P_rt = {scenario[1]:.2f}, E_1 = {scenario[2]:.2f}, Random Factor = {scenario[3]:.2f}")
if __name__ == '__main__':
    # 이 파일을 직접 실행할 때만 실행되는 코드
    generate_scenarios()