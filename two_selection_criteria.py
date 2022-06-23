import numpy as np

def criteria_kolmogorov_smirnov(predicted_data, truth_data):
    from scipy.stats import ks_2samp
    print(predicted_data)
    test = ks_2samp(predicted_data, truth_data)
    alpha = 0.05
    n_x = len(predicted_data)
    n_y = len(truth_data)
    print(test)
    print(f'n_x: {n_x}')
    print(f'n_y: {n_y}')
    critical_value = 1.36 * np.sqrt((n_x + n_y) / (n_x * n_y))
    print(f'sup|M-T|: {test.statistic}')
    print(f'critical_value: {critical_value}')
    if test.statistic < critical_value:
        print('from one distribution')
    else:
        print('different distributions')


def pearson_criterion(distribution_predicted_data, distribution_truth_data, sum_predict, sum_truth):
    chi_square = 0
    n_x = sum(distribution_predicted_data)*sum_predict
    n_y = sum(distribution_truth_data)*sum_truth
    print('Pirson')
    print(distribution_predicted_data)
    print(distribution_truth_data)
    for i in range(0, len(distribution_predicted_data)):
        n_xi = distribution_predicted_data[i] * sum_predict
        n_yi = distribution_truth_data[i] * sum_truth
        chi_square += (1/(n_xi+n_yi))*(distribution_predicted_data[i] - distribution_truth_data[i])**2
    chi_square = n_x*n_y * chi_square
    print(f'chi_square {chi_square}')
    if (chi_square < 0):
        print(f'X^2<0 {chi_square}')
    # Для уровня значимости 0.05
    # 36.4 для r-1=24
    critical_value = 36.4
    if chi_square < critical_value:
        print('from one distribution')
    else:
        print('different distributions')
