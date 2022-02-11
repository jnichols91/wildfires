from functions import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():

    # fires = pyreadr.read_r('../data/fires_clean3.rds')[None]
    # fires_clean = fires.loc[:,['STAT_CAUSE_DESCR',  'FIRE_SIZE', 'LATITUDE', 'LONGITUDE', 'season', 'disc_weekend',
    #                            'disc_holiday','eco1', 'eco3', 'fm.mean', 'Wind.mean', 'human']]
    # fires_clean.to_pickle('../data/fires3.pkl')
    fires_path = '../data/fires3.pkl'
    fires, le_cause, le_season, le_eco1, le_eco3 = init_data(fires_path)

    regions_list = separate_regions(fires, le_eco1, REGION_CODES)
    regions_list[0].drop(['fm.mean', 'Wind.mean'], inplace=True, axis=1)

    total_obs = fires.shape[0]
    cause_list = le_cause.inverse_transform(fires['STAT_CAUSE_DESCR'].unique())
    # results_file = open('../data/obs_pcts.txt', 'w')
    # get_percents(results_file, regions_list, le_cause, total_obs, cause_list)

    # X = regions_list[0].drop(['STAT_CAUSE_DESCR', 'human'], axis=1)
    # y = regions_list[0]['STAT_CAUSE_DESCR']
    # test_model = log_analysis(X, y)[0]
    # print(test_model)

    # knn_results = get_knn_region_params(regions_list)
    # knn_results.to_csv('../results/knn_params.csv')

    # rf_results = get_rf_region_params(regions_list)
    # rf_results.to_csv('../results/rf_params.csv')

    dt_results = get_dt_region_params(regions_list, le_cause)
    # dt_results.to_csv('../results/dt_params.csv')

    # lda_results = get_lda_region_params(regions_list)
    # lda_results.to_csv('../results/lda_params.csv')

    # qda_results = get_qda_region_params(regions_list)
    # qda_results.to_csv('../results/qda_params.csv')

    # svm_results = get_svm_region_params(regions_list)
    # svm_results.to_csv('../results/svm_params.csv')

    # lin_svm_results = get_lin_svm_region_params(regions_list)
    # lin_svm_results.to_csv('../results/lin_svm_params.csv')

    # log_results = get_log_region_params(regions_list)
    # log_results.to_csv('../results/log_params.csv')

    # knn_file = pd.read_csv('../results/knn_params.csv', index_col=0)
    # print(knn_file)
    # dt_file = pd.read_csv('../results/dt_params.csv', index_col=0)
    # lda_file = pd.read_csv('../results/lda_params.csv', index_col=0)
    # qda_file = pd.read_csv('../results/qda_params.csv', index_col=0)
    # rf_file = pd.read_csv('../results/rf_params.csv', index_col=0)
    # print(rf_file)
    # print(rf_file)
    # log_file = pd.read_csv('../results/log_params.csv', index_col=0)
    # get_log_coefs(regions_list, log_file, LOG_PARAMS)
    # svm_file = pd.read_csv('../results/log_params.csv', index_col=0)
    # lin_svm_file = pd.read_csv('../results/lin_svm_params.csv', index_col=0)
    # print(lin_svm_file)
    # # #
    # get_log_coefs(regions_list, log_file, LOG_PARAMS, le_cause)



    # print(X.columns)


if __name__ == '__main__':
    main()
















#
