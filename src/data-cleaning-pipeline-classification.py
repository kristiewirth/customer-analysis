from anonymizingdata import AnonymizingData
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import progressbar
import pudb


class DataCleaning(object):

    def __init__(self):
        pass

    def _intializing_data(self):
        # Importing functions to anonymize data
        self.hidden = AnonymizingData()

        # Force pandas & numpy to display all data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('max_info_columns', 100000)
        pd.set_option('max_seq_items', None)
        np.set_printoptions(threshold=np.nan)

        # Loading in all datasets
        revenue_df = pd.read_pickle('../data/all-datasets/revenue_df')
        EDD_df = pd.read_pickle('../data/all-datasets/EDD_df')
        intercom_df = pd.read_pickle('../data/all-datasets/intercom_df')
        drip_df = pd.read_pickle('../data/all-datasets/drip_df')
        hub_cust_df = pd.read_pickle('../data/all-datasets/hub_cust_df')
        hub_comp_df = pd.read_pickle('../data/all-datasets/hub_comp_df')
        turk_df = pd.read_pickle('../data/all-datasets/turk_df')
        help_scout_df = pd.read_pickle('../data/all-datasets/help_scout_df')

        # Joining all datasets
        self.df = pd.merge(revenue_df, EDD_df, left_on='revenue:id',
                           right_on='edd:customer_id', how='left')
        self.df = pd.merge(self.df, intercom_df, how='left',
                           left_on='revenue:email', right_on='intercom:email')
        self.df = pd.merge(self.df, drip_df, how='left',
                           left_on='revenue:email', right_on='drip:email')
        self.df = pd.merge(self.df, hub_cust_df, how='left', left_on='revenue:email',
                           right_on='hubcust:email')  # 41% null
        self.df = pd.merge(self.df, hub_comp_df, how='left', left_on='revenue:domain',
                           right_on='hubcomp:company_domain_name')  # 53% null
        self.df = pd.merge(self.df, turk_df, how='left', left_on='revenue:domain',
                           right_on='turk:turk_domain')  # 88% null
        self.df = pd.merge(self.df, help_scout_df, how='left', left_on='revenue:email',
                           right_on='helpscout:emails')  # 21% null

    def _cleaning_data(self):
        # Force stray unicode into strings
        categorical_vals = self.df.select_dtypes(exclude=['float64', 'int64'])
        for column in categorical_vals.columns:
            try:
                self.df[column] = self.df[column].astype(str)
            except:
                pass

        # Consolidating any duplicate customer rows
        self.df = self.df.groupby('revenue:email').first().reset_index()

        # Renaming column names for anonymity
        self.df = self.hidden.column_names(self.df)

        # Pulling out test rows
        tests = self.hidden.test_emails(self.df)
        # Dropping all test rows from the self.df
        self.df['test_emails'] = [x in tests for x in self.df['revenue:email']]
        self.df = self.df[self.df['test_emails'] == False]
        self.df.drop('test_emails', inplace=True, axis=1)

        # For dummy variables created before the merge - fill nans with zeros and recode as categorical
        self.df = self.hidden.clean_initial_dummies(self.df)

        # Resetting columns incorrectly coded as numerical
        self.df[['ga:dayofweek_x', 'ga:dayofweek_y']] = self.df[[
            'ga:dayofweek_x', 'ga:dayofweek_y']].astype(object)

        # Recoding categorical as numerical
        self.df['helpscout:number_support_tickets'] = self.df['helpscout:number_support_tickets'].astype(
            int)

        # Dropping a long list of columns that have large portions of nulls, are identifiers,
        # have only one value, those with leakage, and any date columns
        self.df = self.hidden.drop_columns(self.df)

    def _reset_data_types(self, X):
        numerical_vals = X.select_dtypes(include=['float64', 'int64'])
        for column in numerical_vals.columns:
            if len(numerical_vals[column].unique()) <= 3:
                numerical_vals.drop(column, inplace=True, axis=1)
        categorical_vals = X.drop(numerical_vals, axis=1)
        return numerical_vals, categorical_vals

    def _modeling_prep(self):
        # Creating X & y variables
        y = self.df.pop('edd:licenses_license3')
        X = self.df

        # Sorting self.df into numerical and not for scaler
        numerical_vals, categorical_vals = self._reset_data_types(X)

        # Create dummy variables
        print('Creating dummies...')
        bar = progressbar.ProgressBar()
        for column in bar(categorical_vals.columns):
            try:
                X = pd.get_dummies(X, drop_first=True, dummy_na=False, columns=[column])
            except:
                continue

        # Dropping dummy columns for nans or unknowns or specific emails
        bar = progressbar.ProgressBar()
        print('Dropping nan columns')
        for column in bar(X.columns):
            if '_nan' in column or '_unknown' in column or '@' in column:
                X.drop(column, inplace=True, axis=1)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        # Quick fill numerical null values
        numerical_vals, categorical_vals = self._reset_data_types(X)
        for column in numerical_vals.columns:
            mean = X_train[column].mean()
            X_train[column] = X_train[column].fillna(mean)
            X_test[column] = X_test[column].fillna(mean)

        # Dropping any remaining columns with nulls
        for column in X_train.columns:
            try:
                if X_train[column].isnull().sum() > 0 or X_test[column].isnull().sum() > 0:
                    X_train.drop(column, inplace=True, axis=1)
                    X_test.drop(column, inplace=True, axis=1)
            except:
                continue

        # Force y values into integers for algorithms (not floats)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Feature selection using l1 logistic regression
        # model = LogisticRegression(penalty='l2', C=15)
        # model.fit(X_train, y_train)
        # coefs = list(model.coef_)[0]
        # features = list(X_train.columns)
        # importances = []
        # for x, y in zip(features, coefs):
        #     importances.append([x, y])
        #     best_features = [x for x in importances if x[1] != 0.0]
        #     best_features = [x[0] for x in best_features]
        #     X_train_reduced = X_train[best_features]
        X_train_reduced = X_train

        # Scaling train data
        scaler = StandardScaler()
        # Resetting numerical values
        numerical_vals, categorical_vals = self._reset_data_types(X_train_reduced)
        # Scaling data
        X_train_scaled = scaler.fit_transform(X_train_reduced[numerical_vals.columns])
        X_train_scaled = np.concatenate(
            [X_train_scaled, X_train_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        # Scaling test data
        X_test_scaled = scaler.transform(X_test[numerical_vals.columns])
        X_test_scaled = np.concatenate(
            [X_test_scaled, X_test.drop(numerical_vals.columns, axis=1)], axis=1)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train

    def _model_testing(self, model, X_train_scaled, y_train, X_train):
        if model == 'LogisticRegression':
            model = LogisticRegression()
            # param_list = {'penalty': ['l1', 'l2'], 'C': [1, 5, 10, 15]}
            param_list = {'penalty': ['l1', 'l2'], 'C': np.arange(5, 25, 2)}
        elif model == 'KNeighborsClassifier':
            model = KNeighborsClassifier()
            # param_list = {'n_neighbors': [5, 10, 15]}
            param_list = {'n_neighbors': np.arange(2, 20, 2)}
        elif model == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier()
            # param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}
            param_list = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(15, 100, 2),
                          'min_samples_split': [0.1], 'max_features': np.arange(0.5, 0.9, 0.1)}
        elif model == 'SVC':
            model = SVC(kernel='linear')
            # param_list = {'C': [1, 5, 10]}
            param_list = {'C': np.arange(5, 25, 2)}
        elif model == 'SGDClassifier':
            model = SGDClassifier()
            # param_list = {'alpha': [0.001, 0.01, 0.1]}
            param_list = {'alpha': np.arange(0.00001, 0.001, 0.0001)}
        elif model == 'RandomForestClassifier':
            model = RandomForestClassifier()
            param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}
        elif model == 'BaggingClassifier':
            model = BaggingClassifier()
            param_list = {'n_estimators': [5, 10, 15],
                          'max_samples': [.5, 1], 'max_features': [.5, 1]}
        elif model == 'AdaBoostClassifier':
            model = AdaBoostClassifier()
            param_list = {'n_estimators': [5, 10, 15], 'learning_rate': [0.001, 0.01, 0.1]}

        # Grid searching hyperparameters
        g = GridSearchCV(model, param_list, scoring='f1_weighted',
                         cv=5, n_jobs=-1, verbose=10)
        g.fit(X_train_scaled, y_train)
        results = g.cv_results_
        print('\n\n')
        pprint(results)
        print('\n\n')
        print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))
        print('\n\n')

        coefs = list(g.best_estimator_.coef_)[0]
        features = list(X_train.columns)

        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])

        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        print('Coefficients:')
        for pair in importances[:600]:
            print(pair)

    def clustering(self, X_train, y_train, graph=False):
        # Creating the clusters
        cluster = KMeans(n_clusters=2)
        X_train['cluster'] = cluster.fit_predict(X_train)

        # Filtering to test a model on one cluster only
        X_train = X_train[X_train['cluster'] == 0]
        y_train = y_train[X_train.index]

        if graph:
            # For visualizing the clusters
            pca = PCA(n_components=2)
            X_train['pca1'] = pca.fit_transform(X_train)[:, 0]
            X_train['pca2'] = pca.fit_transform(X_train)[:, 1]
            X_train = X_train.reset_index()
            sns.lmplot('pca1', 'pca2', data=X_train, hue='cluster', fit_reg=False)
            plt.savefig('../images/customer_clusters.png', dpi=300)

    def run_pipeline(self, model):
        # self._intializing_data()
        # self._cleaning_data()
        # X_train_scaled, X_test_scaled, y_train, y_test, X_train = self._modeling_prep()
        #
        # # Compressing data for faster performance
        # args = {'X_train_scaled': X_train_scaled,
        #         'X_test_scaled': X_test_scaled, 'y_train': y_train, 'y_test': y_test}
        # np.savez_compressed('../data/Xycompressed_classification', **args)
        # X_train.to_pickle('../data/X_train')

        # # Loading compressed data
        npz = np.load('../data/Xycompressed_classification.npz')
        X_train_scaled = npz['X_train_scaled']
        X_test_scaled = npz['X_test_scaled']
        y_train = npz['y_train']
        y_test = npz['y_test']
        X_train = pd.read_pickle('../data/X_train')

        # self.clustering(X_train, y_train, graph=True)
        self._model_testing(model, X_train_scaled, y_train, X_train)
        return X_train


if __name__ == '__main__':
    dc = DataCleaning()
    X_train = dc.run_pipeline('DecisionTreeClassifier')

    ##################################################
    # Decision tree
    # Best Params: {'max_features': 0.59999999999999998, 'min_samples_split': 0.1, 'criterion': 'gini', 'max_depth': 23}, Best Score: 0.797351847402
    ##################################################
    # Logistic regression
    # Best Params: {'penalty': 'l2', 'C': 15}, Best Score: 0.787997856311
    ##################################################
    # SVC (linear)
    # Best Params: {'C': 5}, Best Score: 0.767679168076
    ##################################################
    # SGDClassifier
    # Best Params: {'alpha': 0.00091000000000000011}, Best Score: 0.790229391565
    ##################################################
    # KNeighborsClassifier
    # Best Params: {'n_neighbors': 5}, Best Score: 0.772965894112
