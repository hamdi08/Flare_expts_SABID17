import pandas as pd
import os
import glob
import re
import numpy as np
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

#Creating the dataset
dataset = prepare_dataset(lookback=24, span=24, withC=False)
all_mvts = dataset["all_mvts"]
timestamps = dataset["timestamps"]
file_names = dataset["file_names"]
y = dataset["y"]


#sorting wrt timestamps
zl = list(zip(all_mvts, timestamps, file_names, y))
df = pd.DataFrame(data=zl, columns=['all_mvts', 'timestamps', 'file_names', 'y'])
df_sorted = df.sort_values(['timestamps', 'file_names'], ascending = [True, True])
all_mvts = np.stack(df_sorted['all_mvts']) #to keep the same dimensions and type
timestamps = np.stack(df_sorted['timestamps'])
file_names = np.stack(df_sorted['file_names'])
y = np.stack(df_sorted['y'])


m = all_mvts.shape[0]
t = all_mvts.shape[1]
n = all_mvts.shape[2]

print('m : ', m, ' n: ', n, ' t: ', t)

# in 2011 - 2014 there are 2408 examples
 # Train with 2011 - 2014
 # Test with 2015 - 2016

split_dt = datetime(2014, 12, 31, 23, 59, 59)
for i in range(m):
    if(timestamps[i] > split_dt):
        split_point = i
        break
print("Split point: ", split_point)

measures = np.zeros((n, 20, 2)) #n vector dataset; 20 candidates of number of neighbors in knn; 2 measures -- acc and tss
for i in range(n):
    #make vector data with parameter i
    print('Vector dataset generation with parameter ', i)
    X = np.zeros((m, 8))
    for j in range(m):
        ts = all_mvts[j, :, i]
        v1 = np.mean(ts)
        v2 = np.std(ts)
        v3 = st.skew(ts)
        v4 = st.kurtosis(ts)
        ts_p = np.zeros((t-1))
        for k in range(t-1):
            ts_p[k] = ts[k+1] - ts[k]
        v5 = np.mean(ts_p)
        v6 = np.std(ts_p)
        v7 = st.skew(ts_p)
        v8 = st.kurtosis(ts_p)
        vect_ts = [v1, v2, v3, v4, v5, v6, v7, v8]
        X[j, :] = vect_ts
    #Vector dataset ready
    print('Classification starts with parameter ', i)
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    print('Models : knn with varying k')

    #splitting into train and test
    X_train = X[0:split_point]
    X_test = X[split_point:m]
    y_train = y[0:split_point]
    y_test = y[split_point:m]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #tss calculation in varying neighbor size of knn
    for k in np.arange(1,21,1):
        mod = neighbors.KNeighborsClassifier(n_neighbors=k)
        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred).ravel()
        tss = (TP / (TP + FN)) - (FP / (FP + TN))
        measures[i, k-1, 0] = tss
        measures[i, k-1, 1] = acc

tss_table = measures[:, :, 0].T
acc_table = measures[:, :, 1].T

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
# medianprops = dict(linestyle='-', linewidth=1.5, color='#000000')
bp = ax.boxplot(tss_table, patch_artist=True)
for box in bp['boxes']:
    box.set(color='#7570b3', linewidth=2)
    box.set(facecolor='#1b9e77')
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='o', color='#e50000', alpha=0.5)
ax.set(title = 'Individual impact of each AR parameter time series on TSS', xlabel = 'AR parameter', ylabel = 'TSS')
ax.set_xticklabels(['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD', 'TOTUSJZ', 'MEANALP',
                    'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT', 'TOTPOT', 'MEANSHR', 'SHRGT45'])
ax.set_ylim(0, 1)
fig.autofmt_xdate(rotation=90, ha='center')
os.chdir('/home/hamdi/GSU/Fall_2017/Flare_expts/Expt_figs/')
fig.savefig('TSS_based_on_indiv_AR_param.png', bbox_inches='tight')