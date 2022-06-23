import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import datetime

# Подбор ядра и параметра ширины окна для оценки плотности методом Парсена-Розенблатта
def density_estimation():
    data = pd.read_csv(r"data/sim_p_2.txt", sep=', ')

    x_train = data['Rig_ист'][:, np.newaxis]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(np.arange(len(x_train)), x_train, color='red')
    plt.xlabel('Sample no.')
    plt.ylabel('Value')
    plt.title('Scatter plot')
    plt.subplot(122)
    plt.hist(x_train, bins=50)
    plt.title('Histogram')
    fig.subplots_adjust(wspace=.3)

    x_test = np.linspace(0, 100, 10000)[:, np.newaxis]


    model = KernelDensity()
    model.fit(x_train)
    log_dens = model.score_samples(x_test)

    plt.fill(x_test, np.exp(log_dens), color='cyan')
    plt.savefig(f'result/densityEstimation/default.png', format='png')
    plt.close()


    bandwidths = [0.01, 0.05, 0.1, 0.5, 1, 4]
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    plt_ind = np.arange(6) + 231

    for b, ind in zip(bandwidths, plt_ind):
        kde_model = KernelDensity(kernel='gaussian', bandwidth=b)
        kde_model.fit(x_train)
        score = kde_model.score_samples(x_test)
        plt.subplot(ind)
        plt.fill(x_test, np.exp(score), c='seagreen')
        plt.title("h="+str(b))

    fig.subplots_adjust(hspace=0.5, wspace=.3)
    plt.savefig(f'result/densityEstimation/bandwidths.png', format='png')
    plt.close()

    # Подбор ядра
    kernels = ['cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat']
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
    plt_ind = np.arange(6) + 231


    for k, ind in zip(kernels, plt_ind):
        kde_model = KernelDensity(kernel=k)
        kde_model.fit([[0]])
        score = kde_model.score_samples(np.arange(-2, 2, 0.1)[:, None])
        plt.subplot(ind)
        plt.fill(np.arange(-2, 2, 0.1)[:, None], np.exp(score), color='seagreen')
        plt.title(k)

    fig.subplots_adjust(hspace=0.5, wspace=.3)
    plt.savefig(f'result/densityEstimation/kernels.png', format='png')
    plt.close()


    # Различные ядра оценки плотности
    def my_scores(estimator, X):
        scores = estimator.score_samples(X)
        # Remove -inf
        scores = scores[scores != float('-inf')]
        # Return the mean values
        return np.mean(scores)


    kernels = ['cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat']
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,10))
    plt_ind = np.arange(6) + 231
    h_vals = np.arange(0.01, 0.1, .02)

    for k, ind in zip(kernels, plt_ind):
        grid = GridSearchCV(KernelDensity(kernel=k),
                            {'bandwidth': h_vals},
                            scoring=my_scores)
        grid_result = grid.fit(x_train)
        print(f'current kernel: {k}')
        # Запись в файл результатов перекрестной проверки
        cv_results = pd.DataFrame(grid_result.cv_results_)
        cv_results.to_csv(f'result/kernels/cv_results_best_kernel_{k}')
        kde = grid.best_estimator_
        log_dens = kde.score_samples(x_test)
        plt.subplot(ind)
        plt.fill(x_test, np.exp(log_dens), c='seagreen')
        plt.title(k + " h=" + "{:.2f}".format(kde.bandwidth))

    fig.subplots_adjust(hspace=.5, wspace=.3)
    plt.savefig(f'result/best_kernels.png', format='png')
    plt.close()


    print('Start best bandwidth and kernel')
    now = datetime.datetime.now()
    print(f'Start time: {now}.')
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': h_vals, 'kernel': kernels},
                        scoring=my_scores)
    print('Start fit')
    grid_result = grid.fit(x_train)
    print(f'End fit: {now}.')
    best_kde = grid.best_estimator_

    # Запись в файл результатов перекрестной проверки
    cv_results = pd.DataFrame(grid_result.cv_results_)
    cv_results.to_csv('result/densityEstimation/cv_results_best_kernel_and_bandwidths.csv')
    log_dens = best_kde.score_samples(x_test)
    plt.fill(x_test, np.exp(log_dens), color='seagreen')
    plt.title("Best Kernel: " + best_kde.kernel+" h="+"{:.2f}".format(best_kde.bandwidth))
    plt.savefig(f'result/densityEstimation/best_scopes.png', format='png')
    plt.close()
    print('End best bandwidth and kernel')
    now = datetime.datetime.now()
    print(f'End time: {now}.')