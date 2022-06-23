import numpy as np
from sklearn.model_selection import train_test_split

'''
Использована реализация из примера авторов OmniFold
https://github.com/hep-lbdl/OmniFold/blob/main/GaussianExample_minimal.ipynb
@article{Andreassen:2019cjw,
    author = "Andreassen, Anders and Komiske, Patrick T. and Metodiev, Eric M. and Nachman, Benjamin and Thaler, Jesse",
    title = "{OmniFold: A Method to Simultaneously Unfold All Observables}",
    eprint = "1911.09107",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1103/PhysRevLett.124.182001",
    journal = "Phys. Rev. Lett.",
    volume = "124",
    pages = "182001",
    year = "2020"
}'''


# f - коэффициент правдоподобия
def reweight(events, model, batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    # nan_to_num заменяет NaN на ноль,
    # +infinity на большие конечные числа, -infinity на очень маленькое или отрицательное число
    # squeeze удаляет оси с одним элементом массива
    return np.squeeze(np.nan_to_num(weights))


# theta0 - массив из пар - [истинное Gen, измеренное Sim]
# theta_unknown_S - измеренные значения Data
def omnifold(theta0, theta_unknown_S, iterations, model, verbose=0):
    # Создает пустой массив формы: длины iterations, где каждый состоит из 2 массивов по len(theta0) элементов
    weights = np.empty(shape=(iterations, 2, len(theta0)))
    # shape = (iteration, step, event)

    # истинное Gen
    theta0_G = theta0[:, 0]
    # измеренное Sim
    theta0_S = theta0[:, 1]
    # Массив из 0 и 1
    labels0 = np.zeros(len(theta0))
    labels_unknown = np.ones(len(theta_unknown_S))
    # Объединяем данные на уровне детектора
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))
    # На уровне частиц дважды значения Gen
    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown))

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))

    for i in range(iterations):

        if (verbose > 0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose > 0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

        # sample sample_weight - это массив чисел который определяет какой вес придать каждому элементу в выборке
        # при вычислении значения потерь.
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'],
                      weighted_metrics=['binary_crossentropy'])
        model.fit(X_train_1,
                  Y_train_1,
                  sample_weight=w_train_1,
                  epochs=20,
                  batch_size=1000,
                  validation_data=(X_test_1, Y_test_1, w_test_1),
                  verbose=verbose)

        weights_pull = weights_push * reweight(theta0_S, model)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose > 0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'],
                      weighted_metrics=['binary_crossentropy'])
        model.fit(X_train_2,
                  Y_train_2,
                  sample_weight=w_train_2,
                  epochs=20,
                  batch_size=1000,
                  validation_data=(X_test_2, Y_test_2, w_test_2),
                  verbose=verbose)

        weights_push = reweight(theta0_G, model)
        weights[i, 1:2, :] = weights_push
        pass

    return weights
