


def main():

    # fires = pyreadr.read_r('../data/fires_clean3.rds')[None]
    # fires_clean = fires.loc[:,['STAT_CAUSE_CODE', 'FIRE_SIZE', 'STATE', 'season', 'disc_weekend',
    #                            'disc_holiday','eco1', 'eco3', 'fm.mean', 'Wind.mean', 'human']]
    # fires_clean.to_pickle('fires.pkl')
    fires_clean = pd.read_pickle('../data/fires2.pkl')

    le_state = preprocessing.LabelEncoder()
    le_season = preprocessing.LabelEncoder()
    le_eco1 = preprocessing.LabelEncoder()
    le_eco3 = preprocessing.LabelEncoder()
    encode_vars(fires_clean, le_state, le_season, le_eco1, le_eco3)

    fires_clean = fires_clean[fires_clean['eco3'] != 0] # 0 NOT FOUND

    X_task3 = fires_clean.drop(['STAT_CAUSE_CODE', 'human'], axis=1)
    y = fires_clean['human']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    lda_model = lda_analysis(X_train, y_train)
    #knn_model = knn_analysis(X_train, y_train)
    # svm_model = svm_analysis(X_train, y_train)

    print(lda_model)

if __name__ == '__main__':
    main()

























#
