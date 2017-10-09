import pandas as pd
import os
import glob
import re
import numpy as np
import scipy.io as sio
import scipy.stats as st
from datetime import datetime
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

def prepare_dataset(lookback, span, withC=False):
    '''
    :param lookback: Lookback time in hours
    :param span: Span window in hours
    :param withC: Considering C class flares as positive examples? Default False.
    :return: A dictionary of all not null mvts and corresponding labels
    '''

    print('Dataset preparation function starts...')

    if lookback == 12:
        os.chdir('/home/hamdi/GSU/Summer17/Flare_project/operational_data/16FL12S_6_12_24_29K_CXMN/')
    elif lookback == 24:
        os.chdir('/home/hamdi/GSU/Summer17/Flare_project/operational_data/16FL24S_6_12_24_29K_CXMN/')
    else:
        print('Wrong lookback entry')
        return 0
    if span > 24:
        print('Wrong span window')
        return 0

    num_rows_to_get = span * 5  # In each hour, there are 5 readings in 12 minutes cadence. 0, 12, 24, 36, 48
    # print("num_rows_to_get ", num_rows_to_get)
    num_flare_params = 16
    pattern = '*Prior' + str(lookback) + 'Span' + str(24) + '*.csv'  # always retrieve span 24 files
    files = glob.glob(pattern)
    #print(files)
    all_mvts = np.zeros((len(files), num_rows_to_get, num_flare_params))
    y = np.empty(len(files), dtype='object')
    fns = np.empty(len(files), dtype='object')  # for storing the file names
    timestamps = np.empty(len(files), dtype='object')  # for storing the the timestamps
    ctr = 0
    c_files = 0
    null_files = 0
    for file_name in files:
        df = pd.read_csv(file_name)
        df = df.loc[:, 'USFLUX':'SHRGT45']  # remove the timestamp from df
        df = df.tail(num_rows_to_get)  # retrieve number of rows according to span
        if ('no' not in file_name):  # if it is flare
            label = True
            if ('classC' in file_name):  # if it is class C flare
                c_files = c_files + 1
                if not withC:  #
                    continue
        else:
            label = False

        if df.isnull().values.any():  # if there is any NaN value in the df, that is taken out of the consideration
            null_files = null_files + 1
            continue
        # print("df.values.shape :", df.values.shape)
        # print("desired shape: ", num_rows_to_get, num_flare_params)
        assert (df.values.shape == (num_rows_to_get, num_flare_params))
        all_mvts[ctr, :, :] = df.values

        dtS = re.search("([0-9]{4}\_[0-9]{2}\_[0-9]{2}\_[0-9]{2}\_[0-9]{2})", file_name).group(0)
        YYYY = int(dtS[0:4])
        MM = int(dtS[5:7])
        DD = int(dtS[8:10])
        HH = int(dtS[11:13])
        mm = int(dtS[14:16])
        dt = datetime(year=YYYY, month=MM, day=DD, hour=HH, minute=mm)
        timestamps[ctr] = dt
        fns[ctr] = file_name
        y[ctr] = label
        ctr = ctr + 1

    all_mvts = all_mvts[0:ctr, :, :]
    fns = fns[0:ctr]
    y = y[0:ctr]
    timestamps = timestamps[0:ctr]

    print('Numpy array base and labels are ready')
    print('Shape of numpy array base: ', all_mvts.shape)
    print('Number of processed files: ', ctr)
    print('Number of c files: ', c_files)
    print('Number of null files: ', null_files)

    dataset = {"all_mvts": all_mvts, "y": y, "file_names": fns, "timestamps":timestamps, "num_c_files": c_files, "num_null_files": null_files}
    return dataset

measures_array = np.zeros((2, 12))  # 2 lookbacks, 12 spans, TSS only

for l in np.arange(12, 36, 12):
    for s in np.arange(2, 26, 2):
        # Creating the dataset
        dataset = prepare_dataset(lookback=l, span=s, withC=False)
        all_mvts = dataset["all_mvts"]
        timestamps = dataset["timestamps"]
        file_names = dataset["file_names"]
        y = dataset["y"]

        # sorting wrt timestamps
        zl = list(zip(all_mvts, timestamps, file_names, y))
        df = pd.DataFrame(data=zl, columns=['all_mvts', 'timestamps', 'file_names', 'y'])
        df_sorted = df.sort_values(['timestamps', 'file_names'], ascending=[True, True])
        all_mvts = np.stack(df_sorted['all_mvts'])  # to keep the same dimensions and type
        timestamps = np.stack(df_sorted['timestamps'])
        file_names = np.stack(df_sorted['file_names'])
        y = np.stack(df_sorted['y'])

        # # remove 2010 records ; 3 only
        # all_mvts = all_mvts[3:len(all_mvts), :, :]
        # timestamps = timestamps[3:len(timestamps)]
        # file_names = file_names[3:len(file_names)]
        # y = y[3:len(y)]

        m = all_mvts.shape[0]
        t = all_mvts.shape[1]
        n = all_mvts.shape[2]

        print('m : ', m, ' n: ', n, ' t: ', t)

        split_dt = datetime(2014, 12, 31, 23, 59, 59)
        for i in range(m):
            if (timestamps[i] > split_dt):
                split_point = i
                break
        print("Split point: ", split_point)

        mod = neighbors.KNeighborsClassifier(n_neighbors=1)
        X = np.zeros((m, 8))
        best_param = 9
        for i in range(m):
            ts = all_mvts[i, :, best_param]
            v1 = np.mean(ts)
            v2 = np.std(ts)
            v3 = st.skew(ts)
            v4 = st.kurtosis(ts)
            ts_p = np.zeros((t - 1))
            for k in range(t - 1):
                ts_p[k] = ts[k + 1] - ts[k]
            v5 = np.mean(ts_p)
            v6 = np.std(ts_p)
            v7 = st.skew(ts_p)
            v8 = st.kurtosis(ts_p)
            vect_ts = [v1, v2, v3, v4, v5, v6, v7, v8]
            X[i, :] = vect_ts

        # classifier
        print('Classification starts with only TOTUSJH...')
        print('X shape: ', X.shape)
        print('y shape: ', y.shape)
        print('Model : knn with k 1')
        # rs = 20
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = rs, stratify = y) #Holdout
        X_train = X[0:split_point]
        X_test = X[split_point:m]
        y_train = y[0:split_point]
        y_test = y[split_point:m]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()
        P = TP + FN
        N = TN + FP
        TSS = ((TP * TN) - (FP * FN)) / (P * N)
        measures_array[int(l/12)-1, int(s/2)-1] = TSS
        print('TSS in lookback ', l, ' and span ', s, ' : ', TSS)

os.chdir('/home/hamdi/GSU/Fall_2017/Flare_expts/Expt_figs/')
sio.savemat('all_lookback_spans_TSS_with_1NN.mat', {'measures_array': measures_array}) #change with lookback



x = np.arange(2, 26, 2)
y1 = measures_array[0,:]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
line1 = ax.plot(x, y1, color = 'r', linewidth = 1, marker = 'o', label='Lookback 12 hours')
y2 = measures_array[1,:]
line2 = ax.plot(x, y2, color = 'b', linewidth = 1, marker = '*', linestyle='--', label='Lookback 24 hours')
ax.set(title='TSS in 1NN with varying lookback and span time window', xlabel='Span window', ylabel='TSS')
plt.xticks(x)
ax.legend(loc='best')

from matplotlib.ticker import FuncFormatter, MaxNLocator
fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.arange(2, 26, 2)
y1 = measures_array[0,:]
y2 = measures_array[1,:]
labels = list(xs)


def format_fn(tick_val, tick_pos):
    if int(tick_val) in xs:
        return labels[int(tick_val)]
    else:
        return ''


ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(x, y1, color = 'r', linewidth = 1, marker = 'o')
ax.plot(x, y2, color = 'b', linewidth = 1, marker = '*', linestyle='--')
ax.set(title='TSS in 1NN with varying lookback and span time window', xlabel='Span window', ylabel='TSS')
plt.show()




