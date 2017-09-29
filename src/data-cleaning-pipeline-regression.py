from anonymizingdata import AnonymizingData
from separatedatasourcecleaning import SeparateDataSets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, SGDRegressor, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
import progressbar
import sys
import pudb
import os


class DataCleaning(object):

    def __init__(self):
        # Importing functions to anonymize data
        self.hidden = AnonymizingData()

    def intializing_data(self):
        # Reset to fix unicode errors
        reload(sys)
        sys.setdefaultencoding('utf8')

        # Force pandas & numpy to display all data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('max_info_columns', 100000)
        pd.set_option('max_seq_items', None)
        np.set_printoptions(threshold=np.nan)

        # Run file that sets up each dataset separately and pickles them for use in this main file
        sds = SeparateDataSets()
        sds.pickle_all()

        # Loading in all datasets
        revenue_df = pd.read_pickle('../data/all-datasets/revenue_df')
        EDD_df = pd.read_pickle('../data/all-datasets/EDD_df')
        intercom_df = pd.read_pickle('../data/all-datasets/intercom_df')
        drip_df = pd.read_pickle('../data/all-datasets/drip_df')
        hub_cust_df = pd.read_pickle('../data/all-datasets/hub_cust_df')
        hub_comp_df = pd.read_pickle('../data/all-datasets/hub_comp_df')
        turk_df = pd.read_pickle('../data/all-datasets/turk_df')
        help_scout_df = pd.read_csv('../data/all-datasets/help_scout_df', low_memory=False)

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

    def cleaning_data(self):
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
        self.df[['ga:dayofweek_x', 'ga:dayofweek_y', 'turk:answer.well-made_y', 'home_phone', 'mobile_phone', 'work_phone', 'other_type_email', 'work_email']] = self.df[[
            'ga:dayofweek_x', 'ga:dayofweek_y', 'turk:answer.well-made_y', 'home_phone', 'mobile_phone', 'work_phone', 'other_type_email',
            'work_email']].astype(object)

        # Recoding categorical as numerical
        self.df[['helpscout:number_support_tickets', 'ga:sessions', 'helpscout:days_open']] = self.df[['helpscout:number_support_tickets', 'ga:sessions',
                                                                                                       'helpscout:days_open']].astype(float)

        # Creating columns for revenue loss due to support tickets
        # self.df['helpscout:months_open'] = self.df['helpscout:days_open'] / 30.
        self.df['helpscout:loss_per_customer'] = self.df['helpscout:days_open'] * 10.0
        self.df['adjusted_revenue'] = self.df['revenue:purchase_value'] - \
            self.df['helpscout:loss_per_customer']

        # Saving all cleaned data (before dropping columns) for graphing purposes later
        self.df.to_csv('../data/all-datasets/cleaned_data.csv', index=False)

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

    def modeling_prep(self):
        # Creating X & y variables
        y = self.df.pop('adjusted_revenue')
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
        for column in X.columns:
            if '_nan' in column or '_unknown' in column or '@' in column:
                X.drop(column, inplace=True, axis=1)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # # Resetting indices for easier referencing
        X_train = X_train.reset_index()
        X_test = X_test.reset_index()

        # Resetting numerical columns after dropping columns
        numerical_vals, categorical_vals = self._reset_data_types(X)

        # Filling in null values using a gridsearched linear model
        for column in numerical_vals.columns:
            # Instantiating model to predict values
            el = ElasticNet()
            param_list = {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]}
            # Grid searching hyperparameters
            g = GridSearchCV(el, param_list, scoring='neg_mean_squared_error',
                             cv=5, n_jobs=3, verbose=1)
            # Getting indices in which the given column is not null
            filled = list(X_train[column].dropna().index)
            # Getting all the X train data for the rows in which the given column is not blank
            mini_training_data = X_train.drop(column, axis=1).apply(
                lambda x: x.fillna(x.mean()), axis=0).iloc[filled]
            # Getting the column to be filled data where it is not null
            mini_training_target = X_train[column].iloc[filled]
            # Fitting the model that will be used to fill the null values
            g.fit(mini_training_data, mini_training_target)
            model = g.best_estimator_
            # Getting indices in which the given column is null
            nulls = [x for x in X_train.index if x not in filled]
            # Getting all the X train data for the rows in which the given column has blank values
            mini_testing_data1 = X_train.drop(column, axis=1).apply(
                lambda x: x.fillna(x.mean()), axis=0).iloc[nulls]
            if mini_testing_data1.empty == False:
                # Predicting the values of the given column where it is blank
                predictions1 = model.predict(mini_testing_data1)
                # Filling in the values that are blank using the predictions
                X_train[column].iloc[nulls] = predictions1
            # Repeating the process for X test (but just using the already trained model)
            nulls = [x for x in X_test.index if x not in filled]
            # Getting all the X test data for the rows in which the given column has blank values
            mini_testing_data2 = X_test.drop(column, axis=1).apply(
                lambda x: x.fillna(x.mean()), axis=0).iloc[nulls]
            if mini_testing_data2.empty == False:
                # Predicting the values of the given column where it is blank
                predictions2 = model.predict(mini_testing_data2)
                # Filling in the values that are blank using the predictions
                X_test[column].iloc[nulls] = predictions2

        # Dropping any remaining columns with nulls
        for column in X_train.columns:
            try:
                if X_train[column].isnull().sum() > 0 or X_test[column].isnull().sum() > 0:
                    X_train.drop(column, inplace=True, axis=1)
                    X_test.drop(column, inplace=True, axis=1)
            except:
                continue

        # # Best features (according to Lasso)
        # model = Lasso(alpha=0.1)
        # model.fit(X_train, y_train)
        # coefs = list(model.coef_)
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

    def model_testing(self, model, X_train_scaled, y_train, X_train):
        if model == 'ElasticNet':
            model = ElasticNet()
            param_list = {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]}
        elif model == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
            param_list = {'n_neighbors': [5, 10, 20]}
        elif model == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor()
            param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}
        elif model == 'SGDRegressor':
            model = SGDRegressor()
            param_list = {'alpha': [0.001, 0.01, 0.1, 1.0], 'penalty': ['l1', 'l2']}
        elif model == 'SVR':
            model = SVR(kernel='linear')
            param_list = {'C': [1, 5, 10, 15]}
        elif model == 'RandomForestRegressor':
            model = RandomForestRegressor()
            param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}

        # Grid searching hyperparameters
        g = GridSearchCV(model, param_list, scoring='r2',
                         cv=5, n_jobs=3, verbose=10)
        g.fit(X_train_scaled, y_train)
        results = g.cv_results_
        print('\n\n')
        pprint(results)
        print('\n\n')
        print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))

        coefs = list(g.best_estimator_.coef_)
        self.print_coefficients(X_train, coefs)

    def final_model(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train):
        # model =
        model.fit(X_train_scaled, y_train)
        y_test_predicted = model.predict(X_test_scaled)
        # RMSE
        # R2
        coefs = list(model.coef_)
        # coefs = list(model.feature_importances_)
        self.print_coefficients(X_train, coefs)
        # self.print_tree(model, X_train)

    def print_coefficients(self, X_train, coefs):
        features = list(X_train.columns)

        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])

        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        print('\n\n')
        print('Coefficients:')
        for pair in importances:
            if pair[1] == 0.0:
                break
            else:
                print(pair)

    def print_tree(self, model, X_train):
        export_graphviz(model, out_file='decision-tree.dot', feature_names=X_train.columns)
        os.system('dot -Tpng decision-tree.dot -o decision-tree.png')


if __name__ == '__main__':
    dc = DataCleaning()

    dc.intializing_data()
    dc.cleaning_data()
    X_train_scaled, X_test_scaled, y_train, y_test, X_train = dc.modeling_prep()

    # Compressing data for faster performance
    args = {'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled, 'y_train': y_train, 'y_test': y_test}
    np.savez_compressed('../data/all-datasets/Xycompressed_regression', **args)
    X_train.to_pickle('../data/all-datasets/X_train_regression')

    # Loading compressed data
    npz = np.load('../data/all-datasets/Xycompressed_regression.npz')
    X_train_scaled = npz['X_train_scaled']
    X_test_scaled = npz['X_test_scaled']
    y_train = npz['y_train']
    y_test = npz['y_test']
    X_train = pd.read_pickle('../data/all-datasets/X_train_regression')

    dc.model_testing('ElasticNet', X_train_scaled, y_train, X_train)

    # dc.final_model(X_train_scaled, X_test_scaled, y_train, y_test, X_train)

    ##################
    # Negative mean squared error
    # ElasticNet
    # Best Params: {'alpha': 0.1, 'l1_ratio': 0.5}, Best Score: -317,669.47006
