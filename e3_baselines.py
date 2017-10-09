import pandas as pd
import os
import glob
import re
import numpy as np
import scipy.io as sio
import scipy.stats as st
from datetime import datetime
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics



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




##########################################################################################################################
#Creating the dataset
dataset = prepare_dataset(lookback=24, span=24, withC=False) #change with lookback
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

# #remove 2010 records ; 3 only
#
# all_mvts = all_mvts[3:len(all_mvts), :, :]
# timestamps = timestamps[3:len(timestamps)]
# file_names = file_names[3:len(file_names)]
# y = y[3:len(y)]


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




measures_array = np.zeros((3, 11)) # 3 baselines, 11 measures

#Baseline 1: Naive PCA of each mvts
#top principal component calculation of each training mvts
mod = neighbors.KNeighborsClassifier(n_neighbors=10)
r_all_mvts = all_mvts.reshape(m, t*n) #useless if we don't use train_test_split
#rs = 33
#r_all_mvts_train, r_all_mvts_test, y_train, y_test = train_test_split(r_all_mvts, y, test_size = 0.33, random_state = rs, stratify = y) #Holdout
r_all_mvts_train = r_all_mvts[0:split_point]
r_all_mvts_test = r_all_mvts[split_point:m]
y_train = y[0:split_point]
y_test = y[split_point:m]
train_mvts = r_all_mvts_train.reshape(r_all_mvts_train.shape[0], t , n)
mm = train_mvts.shape[0]
all_pcs = np.zeros((n,mm))
for i in range(mm):
    mvts = train_mvts[i, :, :]
    scaler = preprocessing.StandardScaler().fit(mvts)
    mvts = scaler.transform(mvts)
    pca = PCA(n_components=1)
    pca.fit(mvts)
    all_pcs[:, i] = pca.components_

#calculating the mean pca
pc_mean = np.mean(all_pcs, axis=1)
pc_mean = pc_mean.reshape(n,1)

#projection on mean pca
X = np.zeros((m, t))
for i in range(m):
    mvts = all_mvts[i, :, :]
    scaler = preprocessing.StandardScaler().fit(mvts)
    mvts = scaler.transform(mvts)
    proj_on_pc_mean = np.dot(mvts, pc_mean)
    X[i, :] = proj_on_pc_mean.T

#vector dataset ready
print('Classification starts for pca based method...')
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('Model : knn with k = 10')
#rs = 20
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = rs, stratify = y) #Holdout
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
acc = (TP + TN) / (TP + FN + TN + FP)
pr_pos = TP/(TP + FP)
pr_neg = TN/(TN + FN)
rc_pos = TP/(TP + FN)
rc_neg = TN/(TN + FP)
f1_pos = (2 * pr_pos * rc_pos) / (pr_pos + rc_pos)
f1_neg = (2 * pr_neg * rc_neg) / (pr_neg + rc_neg)
P = TP + FN
N = TN + FP
HSS1 = (TP + TN - N) / P
HSS2 = (2*((TP*TN)-(FP*FN)))/(P*(FN+TN)+(TP+FP)*N)
CH = ((TP+FP)*(TP+FN))/(P+N)
GS = (TP-CH)/(TP+FP+FN-CH)
TSS = ((TP*TN)-(FP*FN))/(P*N)
measures_array[0, :] = [acc, pr_pos, pr_neg, rc_pos, rc_neg, f1_pos, f1_neg, HSS1, HSS2, GS, TSS]
print('acc pr_pos pr_neg rc_pos rc_neg f1_pos f1_neg HSS1 HSS2 GS TSS')
print(measures_array[0, :])


#Baseline 2: Span 0 by Bobra
mod = SVC(kernel='linear', class_weight={1: 6})
X = np.zeros((m,n))
for i in range(m):
    mvts = all_mvts[i, :, :]
    X[i,:] = mvts[-1, :]

#classifier
print('Classification starts for Bobra method...')
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('Model : svm with class weights 1 to 6')
#rs = 20
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = rs, stratify = y) #Holdout
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
acc = (TP + TN) / (TP + FN + TN + FP)
pr_pos = TP/(TP + FP)
pr_neg = TN/(TN + FN)
rc_pos = TP/(TP + FN)
rc_neg = TN/(TN + FP)
f1_pos = (2 * pr_pos * rc_pos) / (pr_pos + rc_pos)
f1_neg = (2 * pr_neg * rc_neg) / (pr_neg + rc_neg)
P = TP + FN
N = TN + FP
HSS1 = (TP + TN - N) / P
HSS2 = (2*((TP*TN)-(FP*FN)))/(P*(FN+TN)+(TP+FP)*N)
CH = ((TP+FP)*(TP+FN))/(P+N)
GS = (TP-CH)/(TP+FP+FN-CH)
TSS = ((TP*TN)-(FP*FN))/(P*N)
measures_array[1, :] = [acc, pr_pos, pr_neg, rc_pos, rc_neg, f1_pos, f1_neg, HSS1, HSS2, GS, TSS]
print('acc pr_pos pr_neg rc_pos rc_neg f1_pos f1_neg HSS1 HSS2 GS TSS')
print(measures_array[1, :])

#Baseline 3 : 1nn with only TOTUSJH
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
#classifier
print('Classification starts with only TOTUSJH...')
print('X shape: ', X.shape)
print('y shape: ', y.shape)
print('Model : knn with k 1')
#rs = 20
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = rs, stratify = y) #Holdout
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
acc = (TP + TN) / (TP + FN + TN + FP)
pr_pos = TP/(TP + FP)
pr_neg = TN/(TN + FN)
rc_pos = TP/(TP + FN)
rc_neg = TN/(TN + FP)
f1_pos = (2 * pr_pos * rc_pos) / (pr_pos + rc_pos)
f1_neg = (2 * pr_neg * rc_neg) / (pr_neg + rc_neg)
P = TP + FN
N = TN + FP
HSS1 = (TP + TN - N) / P
HSS2 = (2*((TP*TN)-(FP*FN)))/(P*(FN+TN)+(TP+FP)*N)
CH = ((TP+FP)*(TP+FN))/(P+N)
GS = (TP-CH)/(TP+FP+FN-CH)
TSS = ((TP*TN)-(FP*FN))/(P*N)
measures_array[2, :] = [acc, pr_pos, pr_neg, rc_pos, rc_neg, f1_pos, f1_neg, HSS1, HSS2, GS, TSS]
print('acc pr_pos pr_neg rc_pos rc_neg f1_pos f1_neg HSS1 HSS2 GS TSS')
print(measures_array[2, :])

os.chdir('/home/hamdi/GSU/Fall_2017/Flare_expts/Expt_figs/')
sio.savemat('3_baselines_11_metrics_L24.mat', {'measures_array': measures_array}) #change with lookback