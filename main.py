import os
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
import datetime
import mdn
import omnifold as of
import density_estimation as de
import pylandau
from sklearn.neighbors import KernelDensity

from save_helper import create_fit_logs, save_model_nn, save_plt_figure_to_file, save_dictionary_to_file, \
    save_text_to_file
from figure_helper import create_heatmap
from two_selection_criteria import criteria_kolmogorov_smirnov, pearson_criterion


def _main():
    #df = pd.read_csv('result/cv_results.csv', delimiter=',')
    #create_heatmap(df['std_test_r2'], name_xlabel='Nodes', name_ylabel='Layers', xticklabels=[3, 6, 9, 12, 15], yticklabels=[1,2,4,6], cmap='viridis', format="%0.4f")
    #save_plt_figure_to_file(f'result/heatmap_layers_nodes_stds_{datetime.date.today()}.png', 'png')
    now = datetime.datetime.now()
    print(f'Start time: {now}.')
    data_test = pd.read_csv(r"data/exp_p_1.txt", engine='python')
    data = pd.read_csv(r"data/sim_p_2.txt", sep=', ', engine='python')
    # Отсортировать данные по возрастанию
    data = data.sort_values(by='Rig_изм')
    data['Delta'] = data['Rig_изм'] - data['Rig_ист']
    # Вывод спектра жесткости частиц
    """fig, ax = plt.subplots()
    plt.axis([-5,105,-100,100])
    plt.xlabel('Измеренная жёсткость', fontsize=12)
    plt.ylabel('Delta', fontsize=12)
    plt.title('Зависимость измеренных значений от разности (Rig_изм - Rig_ист)')
    plt.plot(data['Rig_изм'],data['Delta'],'ro', ms=1)
    plt.show()
    data['Delta'].to_csv(f'data/delta_sim_p_2.txt', sep=' ', index=False, header=True)"""

    new_row = {'Rig_изм':0, 'Rig_ист':0}
    data = data.append(new_row, ignore_index=True)
    data = data.append(new_row, ignore_index=True)
    # Сохранить сортированные данные
    #sort_data.to_csv(f'data/sorted_sim_p_2.txt', sep=' ', index=False, header=True)
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Измеренный и истинный спектр
    _ = plt.hist(data["Rig_ист"].to_numpy(), bins=np.arange(0, 20 + .1, .1), label='true')
    _ = plt.hist(data["Rig_изм"].to_numpy(), bins=np.arange(0, 20 + .1, .1), label='measured', alpha=0.5)

    plt.xlabel(r'$E[Rig]$')
    plt.ylabel("Counts / bin")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()
    save_plt_figure_to_file(f'result/rigidity_distribution.png', 'png')
    x_train = data['Rig_изм'].astype(np.float32)
    y_train = data['Rig_ист'].astype(np.float32)
    ncols = 1
    x_train.head()

    # Коэффициент детерминизации.
    # Представляет собой долю дисперсии в зависимой переменной, которая предсказуема на основе независимой переменной (переменных).
    # Модели, которые имеют худшие прогнозы, чем этот базовый уровень, будут иметь отрицательный R^2
    def r_square(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res/(SS_tot + K.epsilon()))


    def r_square_loss(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - (1 - SS_res/(SS_tot + K.epsilon()))


    # симметричный вариант MAPE
    def smape_loss(y_true, y_pred):
        from keras import backend as K
        return 100/K.sum(y_true)*K.sum(K.abs(y_true - y_pred) / (K.abs(y_true) + K.abs(y_pred)))


    def smape(y_true, y_pred):
        from keras import backend as K
        return 1 - 100/K.sum(y_true)*K.sum(K.abs(y_true - y_pred) / (K.abs(y_true) + K.abs(y_pred)))


    from sklearn.model_selection import train_test_split
    #xtrain = x_train.values.reshape(-1, 6)
    #ytrain = y_train.values.reshape(-1, 6)
    xtrain, xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.2)
    #xtest = xtest.reshape(-1, 6)
    #ytest = ytest.reshape(-1, 6)
    # Настройка модели сети
    ncols = 1
    model = Sequential()
    # Скрытые слои
    model.add(Dense(10, input_shape=(ncols,), kernel_initializer='glorot_uniform', activation='relu')) # relu or sigmoid
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Выходной слой
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=[smape_loss], metrics=[smape])
    # Сохранить структуру архитектуру НС.
    save_model_nn(model, 'result/model_nn.png')

    tensorboard_callback = create_fit_logs( "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #set early stopping monitor
    early_stopping_monitor = EarlyStopping(patience=5)

    #train model
    history = model.fit(xtrain, ytrain, validation_split=0.2, epochs=50, shuffle=False, verbose=1, callbacks=[early_stopping_monitor, tensorboard_callback])
    # Оценим качество обучения сети на тестовых данных
    scores = model.evaluate(xtest, ytest)
    print("Results fit test: %.4f " % (scores[1]*100))
    ypredict = model.predict(xtest)
    # Сохранить предсказанные значения
    save_dictionary_to_file({
            'X_test': xtest,
            'y_test': ytest,
            'y_predict': ypredict
        }, f'result/predict/recovered_data_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt')
    # Оценка предсказанных результатов
    criteria_kolmogorov_smirnov(ypredict.reshape(-1,), ytest)
    # График функции потерь по числу эпох на тренировочной, тестовой выборке
    f, a0 = plt.subplots()
    p, = a0.plot(history.history['loss'], label='train', color='tab:blue')
    q, = a0.plot(history.history['val_loss'], label='test', color='tab:cyan')
    a0.set_xlabel("epoch")
    a0.set_ylabel(r'$\epsilon$')
    a0.set_ylim(0., 10.)
    a1 = a0.twinx()
    a1.tick_params(labelcolor='tab:orange')
    r, = a1.plot(history.history['smape_loss'], label=r'$R^2$', color='tab:orange')
    a1.set_xlabel('epoch')
    a1.set_ylabel(r'$R^2$')
    a1.set_ylim(0.9, 1.0)
    plt.legend(handles=[p, q, r], loc="best")
    save_plt_figure_to_file("rigidity_loss-vs-epoch.pdf", 'pdf')

    # Нарисовать предсказанное распределение и истинное
    _,_,_=plt.hist(ytest.reshape(-1,1),bins=np.arange(0, 20 + .1, 0.1),color='blue',alpha=0.5,label="Y_test")
    _,_,_=plt.hist(xtest.reshape(-1,1),bins=np.arange(0, 20 + .1, 0.1),color='orange',alpha=0.5,label="X_test")
    _,_,_=plt.hist(ypredict.reshape(-1,1),bins=np.arange(0, 20 + .1, 0.1),histtype="step",color='black',label="Predict")
    plt.xlabel("x")
    plt.ylabel("events")
    plt.legend(frameon=False)
    plt.show()
    save_plt_figure_to_file( f'result/predict/recovered_data_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png', 'png')

    x_train = x_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    print(f'Start fit: {datetime.datetime.now()}.')
    from sklearn.model_selection import GridSearchCV
    from scikeras.wrappers import KerasRegressor


    # Функция для создания модели
    def create_model(layers=1, nodes=2, activation="sigmoid"):
        # Объявляем последовательный тип модели
        model = Sequential()
        #hidden layer(s)
        for i in range(layers):
            for j in range(nodes):
                model.add(Dense(nodes, input_shape=(ncols,), kernel_initializer='glorot_uniform', activation=activation))
        #output layer
        model.add(Dense(1))
        model.compile(optimizer='adam', loss=['mae'], metrics=['accuracy'])
        return model


    validation_split = [0.2]
    epochs = [10]
    nodes = [3, 6, 9]
    layers = [1, 2, 4]
    activation = ['relu']

    model_CV = KerasRegressor(model=create_model, epochs=10,
                              validation_split=0.2, verbose=1, nodes=3, layers=1, activation="relu")

    param_grid = dict(validation_split=validation_split, epochs=epochs, nodes=nodes, layers=layers, activation=activation)
    # cv=3 means stratified 3-Fold is used from (Stratified)KFold
    scoring = ["neg_mean_squared_error", "r2"]
    grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=4, scoring=scoring, refit='r2', cv=3)
    tensorboard_callback_grid_search_cv = create_fit_logs(f'logs/grid_search_cv/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    grid_result = grid.fit(x_train, y_train, callbacks=[tensorboard_callback_grid_search_cv])
    print(f'grid_result: {grid_result}')
    print(f'grid_result_cv: {grid_result.cv_results_}')
    print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_r2']
    stds = grid_result.cv_results_['std_test_r2']
    params = grid_result.cv_results_['params']

    # Отображение результатов перекрестной проверки с помощью тепловых карт
    cv_results = pd.DataFrame(grid_result.cv_results_)
    cv_results.to_csv('result/cv_results.csv')
    print(f'End saved cv_results: {datetime.datetime.now()}.')
    # Постройте средний балл перекрестной проверки
    create_heatmap(means, name_xlabel='Nodes', name_ylabel='Layers', xticklabels=param_grid['nodes'], yticklabels=param_grid['layers'], cmap='viridis', format="%0.4f")
    save_plt_figure_to_file(f'result/heatmap_layers_nodes_means_{datetime.date.today()}.png', 'png')
    # Постройте средний балл отклонения
    create_heatmap(stds, name_xlabel='Nodes', name_ylabel='Layers', xticklabels=param_grid['nodes'], yticklabels=param_grid['layers'], cmap='viridis', format="%0.4f")
    save_plt_figure_to_file(f'result/heatmap_layers_nodes_stds_{datetime.date.today()}.png', 'png')
    mglearn.plots.plot_cross_validation()
    print(f'Program end time : {datetime.datetime.now()}.')
    info_best_score = f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}. mean_test_r2: {means}, std_test_r2: {stds}, params: {params}.'
    save_text_to_file(f'result/example_{datetime.date.today()}.txt', info_best_score)

    results_df = pd.DataFrame(grid_result.cv_results_)
    results_df = results_df.sort_values(by=['param_nodes'])
    results_df = (
        results_df
        .set_index(results_df["params"].apply(
            lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
    )
    results_df = results_df[
        ['param_nodes', 'param_layers', 'param_epochs', 'param_validation_split', 'mean_test_r2']
    ]
    results_df['param_nodes'] = results_df['param_nodes'].astype(np.uint8)
    results_df['param_layers'] = results_df['param_layers'].astype(np.uint8)
    results_df['param_epochs'] = results_df['param_epochs'].astype(np.uint8)
    results_df['param_validation_split'] = results_df['param_validation_split'].astype(float)

    fig = plt.figure(figsize=(15, 5), dpi=80)
    ax = fig.add_subplot(121, projection='3d') # row=1 col=2 cell=1
    img = ax.scatter(results_df['param_nodes'], results_df['param_layers'], results_df['param_epochs'], c=results_df['mean_test_r2'], cmap=plt.hot())
    ax.set_xlabel("nodes")
    ax.set_ylabel("layers")
    ax.set_zlabel("epochs")
    fig.colorbar(img, pad=0.1)

    ax = fig.add_subplot(122, projection='3d')
    img = ax.scatter(results_df['param_nodes'], results_df['param_layers'], results_df['param_validation_split'], c=results_df['mean_test_r2'], cmap=plt.hot())
    ax.set_xlabel("nodes")
    ax.set_ylabel("layers")
    ax.set_zlabel("validation_split")
    fig.colorbar(img, pad=0.1)

    save_plt_figure_to_file(f'result/cosmic_rays_cv-scores_{datetime.date.today()}.pdf', 'pdf')
    print(f'End of dataframe saving: {datetime.datetime.now()}.')


def _new_method():

    def r_square(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res/(SS_tot + K.epsilon()))


    def r_square_loss(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - (1 - SS_res/(SS_tot + K.epsilon()))

    # Возвращает массив частот, длины 1000
    def distribution_mi(arr_mi):
        # создает 1000 интервалов от 0 до 100 через 0.1.
        bins = np.arange(0.0, 100.1, 4)
        result, _ = np.histogram(arr_mi, bins=bins)
        result = result / len(arr_mi)
        df_result = pd.DataFrame(np.squeeze(np.nan_to_num(result)))
        # Преобразовываем форму из (1000,1) в (1000,)
        df_result_reshape = df_result.values.reshape(-1,)
        print('df_result_reshape')
        print(np.shape(df_result_reshape))
        return df_result_reshape

    def create_sum_probabilities_pi(data):
        # Заголовки просто числа
        headers = np.arange(0, len(data), 1)
        dict_samples = dict.fromkeys(headers, {})
        # Заполним подвыборки по столбцам
        i = 0
        for k in headers:
            dict_samples[k] = data[i]
            i += 1
        # Последний столбец сумма вероятностей pi по строке
        df_samples = pd.DataFrame(dict_samples)
        df_samples['Sum'] = df_samples.sum(axis=1)
        return df_samples['Sum']



    now = datetime.datetime.now()
    print(f'Start time: {now}.')
    data_test = pd.read_csv(r"data/exp_p_1.txt", engine='python')
    data = pd.read_csv(r"data/sim_p_2.txt", sep=', ', engine='python')
    # Подготовка данных
    for i in range(708):
        data = data.append({'Rig_изм':0, 'Rig_ист':0}, ignore_index=True)
    # Шаг 1 Сформировать выборку из N элементов, для измеренного и истинного распределения
    x_train = data['Rig_изм'][:600000]
    y_train = data['Rig_ист'][:600000]
    x_test = data['Rig_изм'][600000:830000]
    y_test = data['Rig_ист'][600000:830000]
    xtrain = x_train.values.reshape(-1, 10000)
    ytrain = y_train.values.reshape(-1, 10000)
    xtest = x_test.values.reshape(-1, 10000)
    ytest = y_test.values.reshape(-1, 10000)
    xtrain_distribution = np.reshape(xtrain[0], (-1, 25))
    ytrain_distribution = np.reshape(ytrain[0], (-1, 25))
    xtest_distribution = np.reshape(xtest[0], (-1, 25))
    ytest_distribution = np.reshape(ytest[0], (-1, 25))
    print('xtrain_distribution')
    print(len(xtrain_distribution))
    for index in range(len(xtrain)):
        xtrain_distribution[index] = distribution_mi(xtrain[index])
    for index in range(len(ytrain)):
        ytrain_distribution[index] = distribution_mi(ytrain[index])
    for index in range(len(xtest)):
        xtest_distribution[index] = distribution_mi(xtest[index])
    for index in range(len(ytest)):
        ytest_distribution[index] = distribution_mi(ytest[index])


    # Шаг 3 Создание модели
    ncols = 25
    N_HIDDEN = 25
    N_MIXES = 25

    model = Sequential()
    model.add(Dense(N_HIDDEN, batch_input_shape=(None, 25), activation='relu'))
    model.add(Dense(N_HIDDEN, activation='relu'))
    model.add(Dense(N_HIDDEN, activation='relu'))
    model.add(mdn.MDN(25, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(25, N_MIXES), optimizer='adam')
    model.summary()

    tensorboard_callback = create_fit_logs("result/mdn/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    history = model.fit(x=xtest_distribution, y=ytest_distribution , batch_size=128, epochs=10, validation_split=0.2, callbacks=[tensorboard_callback])
    scores = model.evaluate(xtest_distribution, xtest_distribution)
    #print("Results fit test: %.4f " % (scores[1]))
    ypredict = model.predict(xtest_distribution)
    print(ypredict.shape)
    criteria_kolmogorov_smirnov(np.array(ypredict.reshape(-1,)[:200000]), np.array(ytest_distribution.reshape(-1,)[:200000]))
    # Сохранить предсказанные значения
    '''save_dictionary_to_file({
            'X_test': xtest_distribution,
            'y_test': ytest_distribution,
            'y_predict': ypredict
        }, f'result/new_method/recovered_data_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt')'''

    pearson_criterion(ypredict[0], ytest_distribution[0],len(ytest[0]))
    # Датафрейм, у которого по столбцам выборки, последний столбец - сумма по строке вероятностей всех pi.
    sum_pi_xtrain = create_sum_probabilities_pi(xtrain_distribution)
    sum_pi_ytrain = create_sum_probabilities_pi(ytrain_distribution)
    sum_pi_xtest = create_sum_probabilities_pi(xtest_distribution)
    sum_pi_ytest = create_sum_probabilities_pi(ytest_distribution)
    sum_pi_ypredict = create_sum_probabilities_pi(ypredict)

    # Вывести восстановленный и истинный спектр
    #_,_,_=plt.hist(ytest.reshape(-1,1),bins=np.arange(0, 1 + .1, 0.001),color='blue',alpha=0.5,label="Y_test")
    #_,_,_=plt.hist(xtest.reshape(-1,1),bins=np.arange(0, 1, 0.001),color='orange',alpha=0.5,label="X_test")
    #_,_,_=plt.hist(df_density['Sum'],bins=30,histtype="step",color='black',label="Predict", density=True)
    #_,_ = plt.bar(range(0, 1000), df_density['Sum'])
    plt.bar(range(0, 25), sum_pi_xtrain, alpha=0.5, label='X_train', width=1, color='blue')
    plt.bar(range(0, 25), sum_pi_ytrain, alpha=0.5, label='y_train',  width=1, color='orange')
    #plt.bar(range(0, 25), sum_pi_xtest, alpha=0.5, label='X_test', width=1, color='green')
    #plt.bar(range(0, 25), sum_pi_ytest, alpha=0.5, label='y_test',  width=1, color='red')
    #plt.bar(range(0, 25), sum_pi_ypredict, alpha=0.5, label='y_predict',  width=1)
    plt.step(np.arange(-0.5, 24, 1), sum_pi_ypredict, where='post', label='y_predict', color='black')
    plt.xlabel("bins")
    plt.ylabel("Sum(pi)")
    plt.legend(frameon=False)
    plt.show()



def _mdn_neural_network():
    data = pd.read_csv(r"data/sim_p_2.txt", sep=', ', engine='python')
    # Критерий Колмогорова-Смирнова для тестовой выборке
    criteria_kolmogorov_smirnov(data['Rig_изм'][600000:], np.array(data['Rig_ист'][600000:]))

    # Размер генерируемых данных:
    NSAMPLE = 600000

    y_data = data['Rig_ист'][:600000]
    print(f'y_data {y_data}')
    x_data = data['Rig_изм'][:600000]
    x_data = x_data.values.reshape((NSAMPLE, 1))
    print(f'x_data reshape: {x_data}')
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', alpha=0.3)
    plt.xlabel('Измеренные значения')
    plt.ylabel('Истинные значения')
    plt.legend(loc='upper right')
    plt.show()

    N_HIDDEN = 10 # число скрытых нейронов в скрытом слое
    N_MIXES = 5 # количество компонентов смеси
    OUTPUT_DIMS = 1 # размерность на выходе

    model = Sequential()
    model.add(Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
    model.add(Dense(N_HIDDEN, activation='relu'))
    model.add(Dense(N_HIDDEN, activation='relu'))
    model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))
    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer='adam')
    model.summary()

    tensorboard_callback = create_fit_logs("result/mdn/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    history = model.fit(x=x_data, y=y_data, batch_size=128, epochs=100, validation_split=0.2, callbacks=[tensorboard_callback])
    x_test = data['Rig_изм'][600000:]
    NTEST = x_test.size
    x_test = x_test.values.reshape(NTEST, 1)
    # Предсказание на тестовой выборке, содержит параметры смесей распределений
    y_test = model.predict(x_test)

    # Выборка из предсказанных распределений
    y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_test, 1, N_MIXES, temp=1.0)
    y_predicted = np.squeeze(y_samples)

    _, _, _ = plt.hist(data['Rig_ист'][600000:], bins=np.arange(0, 100 + .1, .1), label='true', alpha=0.5, color='orange')
    _, _, _ = plt.hist(data['Rig_изм'][600000:], bins=np.arange(0, 100 + .1, .1), label='measured', alpha=0.5, color='blue')
    _, _, _ = plt.hist(y_predicted,bins=np.arange(0, 100 + .1, .1), histtype="step", color='black', label="predict")

    plt.xlabel(r'$bins$')
    plt.ylabel("Counts / bin")
    plt.legend(loc='upper right')
    plt.show()

    _, _, _ = plt.hist(data['Rig_ист'][600000:], bins=np.arange(0, 100 + .1, 4), label='true', alpha=0.5, color='orange')
    _, _, _ = plt.hist(data['Rig_изм'][600000:], bins=np.arange(0, 100 + .1, 4), label='measured', alpha=0.5, color='blue')
    _, _, _ = plt.hist(y_predicted,bins=np.arange(0, 100 + .1, 4), histtype="step", color='black', label="predict")

    plt.xlabel(r'$bins$')
    plt.ylabel("Counts / bin")
    plt.legend(loc='upper right')
    plt.show()

    # Проверка однородности выборок К-С
    criteria_kolmogorov_smirnov(np.array(y_predicted), np.array(data['Rig_ист'][600000:]))
    # Сохранить предсказанные значения на тестовой выборке
    save_dictionary_to_file({
            'X_test': data['Rig_изм'][600000:],
            'y_test': data['Rig_ист'][600000:],
            'y_predict': y_predicted
        }, f'result/mdn/recovered_data_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt')

    # Получить параметры смеси
    # Мат. ож.
    mus = np.apply_along_axis((lambda a: a[:N_MIXES]), 1, y_test)
    # Среднекв. откл.
    sigs = np.apply_along_axis((lambda a: a[N_MIXES:2*N_MIXES]), 1, y_test)
    # Веса
    pis = np.apply_along_axis((lambda a: mdn.softmax(a[2*N_MIXES:])), 1, y_test)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_samples[:, :, 0], 'bo', alpha=0.3)
    plt.show()

    # График средних значений - это дает нам некоторое представление о том, как модель учится создавать смеси
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, mus, 'bo', alpha=0.3)
    plt.show()

    # График отклонений и весов средних значений
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_data, y_data, marker='o', c='r', alpha=0.3)
    for i in range(N_MIXES):
        ax1.scatter(x_test, mus[:, i], marker='o', s=200 * sigs[:, i] * pis[:, i], alpha=0.3)
    plt.show()

    bins = np.arange(0.0, 100.1, 4)
    count_bins_true, _ = np.histogram(data['Rig_ист'][600000:], bins=bins)
    count_bins_predict, _ = np.histogram(y_predicted, bins=bins)
    distribution_true = count_bins_true / sum(data['Rig_ист'][600000:])
    distribution_predict = count_bins_predict / sum(y_predicted)
    pearson_criterion(distribution_predict, distribution_true, len(data['Rig_ист'][600000:]))


def _omnifold():
    data_test = pd.read_csv(r"data/exp_p_1.txt", engine='python')
    data = pd.read_csv(r"data/sim_p_2.txt", sep=', ', engine='python')

    N = 600000

    # Синтетический уровень
    # theta0_G = np.random.normal(0.2, 0.8, N)  # Generator-level synthetic sample
    # theta0_S = np.array([(x + np.random.normal(0, 0.5)) for x in theta0_G])  # Добавление искажений к генератору
    # theta0_G = np.random.exponential(2.4, 600000) # Показательное распределение
    # theta0_G = np.random.lognormal(0.45, 0.75, 600000) # Логарифмическое нормальное распределение
    # theta0_G = np.random.lognormal(0.45, 1.2, 600000) # Приближает x > 4
    # theta0_G = np.random.wald(2.57,2.1, 600000) # Распределение Вальда
    # Непараметрический метод восстановления плотности Парсена-Розенблатта
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.01)
    kde_model.fit(data['Rig_ист'][:, None])
    theta0_G = np.squeeze(kde_model.sample(n_samples=600000))
    criteria_kolmogorov_smirnov(theta0_G, data['Rig_ист'])
    # theta0_G = np.random.exponential(2.7, 600000) # Показательное распределение
    # theta0_G = np.random.pareto(0.47, 600000) # Распределение Парето
    # theta0_G = data_test['R_изм'][: 600000]
    # x = np.arange(0, 20, 0.001)
    # y = pylandau.landau(x) # Распределение Ландау
    # theta0_G = np.random.choice(x, size=600000, p=y / y.sum())
    # theta0_S = np.array([(x + np.random.exponential(0.02)) for x in theta0_G])
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.01)
    kde_model.fit(data['Rig_изм'][:, None])
    theta0_S = np.squeeze(kde_model.sample(n_samples=600000))
    criteria_kolmogorov_smirnov(theta0_S, data['Rig_изм'])
    # Объединяет в массив из пар - массив [истинное Gen, измеренное Sim]
    theta0 = np.stack([theta0_G, theta0_S], axis=1)
    # Естественный уровень
    theta_unknown_G = data["Rig_ист"][: 600000]
    theta_unknown_S = data["Rig_изм"][: 600000]

    _, _, _ = plt.hist(theta0_G, bins=np.arange(0, 20 + .1, 0.1), color='blue', alpha=0.5, label="Gen, true")
    _, _, _ = plt.hist(theta0_S, bins=np.arange(0, 20 + .1, 0.1), histtype="step", color='black', ls=':', label="Sim")
    _, _, _ = plt.hist(data["Rig_ист"][: 600000].to_numpy(), bins=np.arange(0, 20 + .1, 0.1), color='orange', alpha=0.5,
                       label="Truth")
    _, _, _ = plt.hist(data["Rig_изм"][: 600000].to_numpy(), bins=np.arange(0, 20 + .1, 0.1), histtype="step",
                       color='black', label="Data")

    plt.xlabel("x")
    plt.ylabel("events")
    plt.legend(frameon=False)
    plt.show()

    inputs = Input((1,))
    hidden_layer_1 = Dense(50, activation='relu')(inputs)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
    model = Model(inputs=inputs, outputs=outputs)

    myweights = of.omnifold(theta0, theta_unknown_S, 2, model, 1)

    # Измеренный, истинный и восстановленный спектр
    _, _, _ = plt.hist(theta_unknown_S, bins=np.arange(0, 20 + .1, 0.1), color='blue', alpha=0.5, label="Measured")
    _, _, _ = plt.hist(theta_unknown_G, bins=np.arange(0, 20 + .1, 0.1), color='orange', alpha=0.5, label="Data, true")
    _, _, _ = plt.hist(theta0_G, weights=myweights[-1, 0, :], bins=np.arange(0, 20 + .1, 0.1), color='black',
                       histtype="step", label="OmniFolded", lw="2")
    plt.xlabel("x")
    plt.ylabel("events")
    plt.legend(frameon=False)
    plt.show()

    # Получить восстановленные значения
    restored_spectrum = myweights[-1, 0, :] * theta0_G
    criteria_kolmogorov_smirnov(restored_spectrum, data["Rig_ист"][: 600000])
    save_dictionary_to_file({
        'y_test': theta0_G,
        'y_predict': restored_spectrum
    }, f'result/omnifold/recovered_data_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt')
    print('End')
    f = open('result.txt', 'w+')  # открываем файл на запись
    i = 0
    for el in theta_unknown_G:
        f.write(f'{el}    {theta0_G[i]}' + '\n')
        i += 1
    f.close()


if __name__ == '__main__':
    run_algorithm = 'omnifold'

    if run_algorithm == 'neural_network':
        _main()
    elif run_algorithm == 'omnifold':
        _omnifold()
        # Поиск лучших параметров для аппроксимации плотности
        #de.density_estimation()
    elif run_algorithm == 'mdn':
        _mdn_neural_network()
    else:
        _new_method()
