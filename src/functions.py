# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import pyreadr
import pydotplus
import time
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from bayes_opt import BayesianOptimization
np.random.seed(0)

# used to transform consistently across the eco1 encoder/ matches regions_list
REGION_CODES = ['3  TAIGA', '5  NORTHERN FORESTS', '6  NORTHWESTERN FORESTED MOUNTAINS',
                '7  MARINE WEST COAST FOREST', '8  EASTERN TEMPERATE FORESTS', '9  GREAT PLAINS',
                '10  NORTH AMERICAN DESERTS', '11  MEDITERRANEAN CALIFORNIA', '12  SOUTHERN SEMIARID HIGHLANDS',
                '13  TEMPERATE SIERRAS', '15  TROPICAL WET FORESTS']
TASK2_MODELS = ['Random Forest', 'KNN']
KNN_PARAMS = ['n_neighbors']
RF_PARAMS = ['criterion', 'max_depth', 'max_features', 'n_estimators']
DT_PARAMS = ['criterion', 'max_depth']
LDA_PARAMS = ['shrinkage']
QDA_PARAMS = ['reg_param']
SVM_PARAMS = ['C', 'gamma']
LIN_SVM_PARAMS = ['C', 'penalty']
LOG_PARAMS = ['C', 'penalty']

# create an encoding of the categorical variables to be consistent (integer valued)
def encode_vars(df, cause, season, eco1, eco3):
    df['STAT_CAUSE_DESCR'] = cause.fit_transform(df['STAT_CAUSE_DESCR'])
    df['STAT_CAUSE_DESCR'] = df['STAT_CAUSE_DESCR'].astype('category')
    df['human'] = df['human'].astype(int).astype('category')
    df['season'] = season.fit_transform(df['season'])
    df['season'] = df['season'].astype('category')
    df['disc_weekend'] = df['disc_weekend'].astype(int).astype('category')
    df['disc_holiday'] = df['disc_holiday'].astype(int).astype('category')
    df['eco1'] = eco1.fit_transform(df['eco1'])
    df['eco1'] = df['eco1'].astype('category')
    df['eco3'] = eco3.fit_transform(df['eco3'])
    df['eco3'] = df['eco3'].astype('category')

# read in the data encode the data and return the encoders
def init_data(file):
    df = pd.read_pickle(file)
    cause = preprocessing.LabelEncoder()
    state = preprocessing.LabelEncoder()
    season = preprocessing.LabelEncoder()
    eco1 = preprocessing.LabelEncoder()
    eco3 = preprocessing.LabelEncoder()
    encode_vars(df, cause, season, eco1, eco3)
    # df2 = df[df['eco1'] != eco1.transform(['3  TAIGA'])[0]]
    # df2 = df2[df2['eco1'] != eco1.transform(['2  TUNDRA'])[0]]
    # print(df['FIRE_SIZE'].min(), df['FIRE_SIZE'].max())
    # print(df['fm.mean'].min(), df['fm.mean'].max())
    # print(df['Wind.mean'].min(), df['Wind.mean'].max())

    return df, cause, season, eco1, eco3

# this will create a list of df's each representing a region. drops the eco1 column
def separate_regions(df, le_eco1, regions):
    eco = []
    for region in regions:
        eco.append(df[df['eco1']==le_eco1.transform([region])[0]].drop(['eco1'], axis=1))

    return eco

def get_percents(file, regions, le_cause, total, cause_list):
    results_df = pd.DataFrame(None, REGION_CODES, cause_list)

    file.write('***** Original Level One Observation Dispersion ***** \n')
    file.write('-'*90 + '\n')
    for i, region in enumerate(REGION_CODES):
        region_total = regions[i].shape[0]
        file.write(f"""Region {region} has {region_total/total*100:.2f}% of the original observations \n""")
    file.write('-'*90 + '\n')

    file.write('***** Percentage of Fires by Cause in each Region ***** \n')
    file.write('-'*90 + '\n')
    for i, region in enumerate(REGION_CODES):
        region_total = regions[i].shape[0]
        file.write(f'Region {region}: \n')
        for j, cause in enumerate(cause_list):
            cause_total = len(regions[i][regions[i]['STAT_CAUSE_DESCR']==le_cause.transform([cause])[0]])
            file.write(f'{cause_total/region_total*100:.2f}% of the fires were caused by {cause} \n')
            results_df.iloc[i,j] = f'{cause_total/region_total*100:.2f}%'
        file.write('-'*90 + '\n')

    return results_df

def knn_analysis(X, y):
    max_n = round(np.sqrt(X.shape[0]))
    print(max_n)
    knn_params = {'n_neighbors': list(range(3, max_n, round(max_n/7)))}
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing KNN Grid Search...')
    knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, scoring='accuracy', cv=cv, n_jobs=1)
    knn_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete KNN: {finish - start:.2f}')

    return [knn_grid.best_params_, f'{knn_grid.best_score_*100:.2f}%']

def get_knn_region_params(regions):
    knn_df = pd.DataFrame(None, REGION_CODES, KNN_PARAMS)

    for i, region in enumerate(REGION_CODES):
        if i==4:continue
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        knn_params = knn_analysis(X, y)[0]
        knn_df.loc[region] = list(knn_params.values())
        print(knn_df)

    return knn_df

def svm_analysis(X, y):
    svm_params = {
    'C': [0.01, 1],
    'gamma': [0.1]
    }
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing SVM Grid Search...')
    svm_grid = GridSearchCV(SVC(kernel='rbf'), param_grid=svm_params, scoring='accuracy', cv=cv, n_jobs=1, verbose=3)
    svm_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete SVM: {finish - start:.2f}')

    return [svm_grid.best_params_, svm_grid.best_score_]

def get_svm_region_params(regions):
    svm_df = pd.DataFrame(None, REGION_CODES, SVM_PARAMS)

    for i, region in enumerate(REGION_CODES):
        if i in [0, 1, 2, 4]:
            continue
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        print(region)
        svm_params = svm_analysis(X, y)[0]
        print(svm_params)
        svm_df.loc[region] = list(svm_params.values())
        print(svm_df)

    return svm_df

def lin_svm_analysis(X, y):
    lin_svm_params = {'penalty': ['l2'],
                  'C': [0.001, 0.1]}
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing Linear SVM Grid Search...')
    lin_svm_grid = GridSearchCV(LinearSVC(), param_grid=lin_svm_params, scoring='accuracy', cv=cv, n_jobs=1, verbose=3)
    lin_svm_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete Linear SVM: {finish - start:.2f}')

    return [lin_svm_grid.best_params_, lin_svm_grid.best_score_]

def get_lin_svm_region_params(regions):
    lin_svm_df = pd.DataFrame(None, REGION_CODES, LIN_SVM_PARAMS)

    for i, region in enumerate(REGION_CODES):
        if i in [0, 1, 2, 3, 4]:
            continue
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        print(region)
        lin_svm_params = lin_svm_analysis(X, y)[0]
        print(lin_svm_params)
        lin_svm_df.loc[region] = list(lin_svm_params.values())
        print(lin_svm_df)

    return lin_svm_df

def rf_analysis(X, y):
    rf_params = {
    'n_estimators': [100, 300, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [5, 10, 15, 20],
    'criterion' :['gini', 'entropy']
    }
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing Random Forest Grid Search...')
    rf_grid = GridSearchCV(RandomForestClassifier(), param_grid=rf_params, scoring='accuracy', cv=cv, n_jobs=1)
    rf_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete RF: {finish - start:.2f}')

    return [rf_grid.best_params_, f'{rf_grid.best_score_*100:.2f}%']

def get_rf_region_params(regions):
    rf_df = pd.DataFrame(None, REGION_CODES, RF_PARAMS)

    for i, region in enumerate(REGION_CODES):
        if i in [6, 7, 8, 9, 10]:
            print(region)
            X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
            y = regions[i]['STAT_CAUSE_DESCR']
            rf_params = rf_analysis(X, y)[0]
            rf_df.loc[region] = list(rf_params.values())
            print(rf_df)

    return rf_df

def dt_analysis(X, y):
    dt_params = {
    'max_depth' : [1, 2, 3],
    'criterion' :['gini', 'entropy']
    }
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing Decision Tree Grid Search...')
    dt_grid = GridSearchCV(DecisionTreeClassifier(), param_grid=dt_params, scoring='accuracy', cv=cv, n_jobs=1)
    dt_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete DT: {finish - start:.2f}')

    return [dt_grid.best_params_, dt_grid.best_estimator_]

def get_dt_region_params(regions, cause):
    dt_df = pd.DataFrame(None, REGION_CODES, DT_PARAMS)

    for i, region in enumerate(REGION_CODES):

        dot_data = StringIO()
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        dt_result = dt_analysis(X, y)

        importance = dt_result[1].feature_importances_
        print(importance)
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        x_vals = [x for x in range(len(importance))]
        fig1 = plt.figure(figsize=(8,6))
        ax1 = fig1.add_axes([0.1,0.15,0.9,0.75])
        plt.rc('xtick', labelsize=7)
        plt.rc('ytick', labelsize=7)
        ax1.bar(X.columns, importance)
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(X.columns)
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Importance')
        ax1.set_title(region)
        fig1.savefig(f'../figures/importance/{region}_imp.png', dpi=150)
        plt.close()
        dt_df.loc[region] = list(dt_result[0].values())
        export_graphviz(
             dt_result[1],
             out_file=dot_data,
             feature_names=X.columns, rounded=True, precision=2,
             class_names=sorted(cause.inverse_transform(y.unique())),
             filled=True)
        dt_tree = pydotplus.graph_from_dot_data(dot_data.getvalue())
        dt_tree.set_nodesep('0.025')
        dt_tree.set_fontsize('16.0')
        dt_tree.set_size('"7.75,10.25"]')
        dt_tree.set_dpi('250')
        colors = []
        nodes = dt_tree.get_node_list()
        edges = dt_tree.get_edge_list()
        for edge in edges:
            edge.set_weight('2.5')
            #edge.set_fontsize('12.0')
        dt_tree.write_png(f'../figures/trees/{region}.png')
        for node in nodes:
            continue
    return dt_df

def lda_analysis(X, y):
    lda_params = {
    'shrinkage': np.logspace(-3, 0, 20),
    }
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing LDA Grid Search...')
    lda_grid = GridSearchCV(LinearDiscriminantAnalysis(solver='lsqr'), param_grid=lda_params, scoring='accuracy', cv=cv, n_jobs=1)
    lda_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete LDA: {finish - start:.2f}')

    return [lda_grid.best_params_, f'{lda_grid.best_score_*100:.2f}%']

def get_lda_region_params(regions):
    lda_df = pd.DataFrame(None, REGION_CODES, LDA_PARAMS)

    for i, region in enumerate(REGION_CODES):
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        lda_params = lda_analysis(X, y)[0]
        temp_list = list(lda_params.values())
        lda_df.loc[region] = [round(i, 4) for i in temp_list]
        print(lda_df)

    return lda_df

def qda_analysis(X, y):
    qda_params = {
    'reg_param': np.linspace(0, 1, 40, endpoint=False),
    }
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing QDA Grid Search...')
    qda_grid = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid=qda_params, scoring='accuracy', cv=cv, n_jobs=1)
    qda_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete QDA: {finish - start:.2f}')

    return [qda_grid.best_params_, f'{qda_grid.best_score_*100:.2f}%']

def get_qda_region_params(regions):
    qda_df = pd.DataFrame(None, REGION_CODES, QDA_PARAMS)

    for i, region in enumerate(REGION_CODES):
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        qda_params = qda_analysis(X, y)[0]
        temp_list = list(qda_params.values())
        qda_df.loc[region] = [round(i, 4) for i in temp_list]
        print(qda_df)

    return qda_df

def log_analysis(X, y):
    log_params = {'penalty': ['l1', 'l2'],
                  'C': np.logspace(-2, 2, 5)}
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
    start = time.time()
    print('Performing Logistic Regression Grid Search...')
    log_grid = GridSearchCV(LogisticRegression(solver='saga', max_iter=150, tol=1e-3), param_grid=log_params, scoring='accuracy', cv=cv, n_jobs=1)
    log_grid.fit(X, y)
    finish = time.time()
    print(f'Time to complete Logistic Regression: {finish - start:.2f}')

    return [log_grid.best_params_, f'{log_grid.best_score_*100:.2f}%', log_grid.best_estimator_.coef_]

def get_log_region_params(regions):
    log_df = pd.DataFrame(None, REGION_CODES, LOG_PARAMS)

    for i, region in enumerate(REGION_CODES):
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        log_params = log_analysis(X, y)[0]
        log_df.loc[region] = list(log_params.values())
        print(log_df)

    return log_df

def get_best(regions, param_df, param_names):
    cv = RepeatedStratifiedKFold(n_splits=2, random_state=0)
    for i, region in enumerate(REGION_CODES):
        if i in [0, 1, 2, 3]:
            continue
        start = time.time()
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        y = regions[i]['STAT_CAUSE_DESCR']
        params = dict(zip(param_names, param_df.loc[region]))
        acc = np.mean(cross_val_score(LinearSVC(**params), X, y, cv=cv, verbose=3))
        finish = time.time()
        print(f'Linear SVM got an accuracy of {acc*100:.2f}% for {region} in {finish - start:.2f} seconds')

def get_log_coefs(regions, param_df, param_names, cause):

    for i, region in enumerate(REGION_CODES):
        X = regions[i].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
        X_scaled = preprocessing.StandardScaler(X)
        y = regions[i]['STAT_CAUSE_DESCR']
        log_coefs = pd.DataFrame(log_analysis(X_scaled, y)[2], index=sorted(cause.inverse_transform(y.unique())), columns=X.columns)
        print(region)
        print(log_coefs)







































#
