from anonymizingdata import AnonymizingData
from separatedatasourcecleaning import SeparateDataSets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression, ElasticNet
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
# from imblearn.over_sampling import SMOTE
from pprint import pprint
import progressbar
import sys
import pudb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


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

        # Dropping out manual payment methods of $0 revenue - friend purchases only
        friend_purchases = self.df[(self.df['revenue:purchase_value'] == 0.0) & (
            self.df['edd:payment_method_manual_purchases'] == 1)].index
        self.df = self.df.loc[~self.df.index.isin(friend_purchases)]

        # For dummy variables created before the merge - fill nans with zeros and recode as categorical
        self.df = self.hidden.clean_initial_dummies(self.df)

        # Resetting columns incorrectly coded as numerical
        self.df[['ga:dayofweek_x', 'ga:dayofweek_y', 'turk:answer.well-made_y', 'home_phone', 'mobile_phone', 'work_phone', 'other_type_email', 'work_email']] = self.df[[
            'ga:dayofweek_x', 'ga:dayofweek_y', 'turk:answer.well-made_y', 'home_phone', 'mobile_phone', 'work_phone', 'other_type_email',
            'work_email']].astype(object)

        # Recoding categorical as numerical
        self.df[['helpscout:number_support_tickets', 'helpscout:days_open', 'turk:answer.well-made_y']] = self.df[[
            'helpscout:number_support_tickets', 'helpscout:days_open', 'turk:answer.well-made_y']].astype(float)

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
        for column in X.columns:
            if '_nan' in column or '_unknown' in column or '@' in column:
                X.drop(column, inplace=True, axis=1)

        # Dropping additional dummy columns
        X.drop(["intercom:industry_Not Online/Can't Access",
                'edd:state_none'], inplace=True, axis=1)

        # Exporting data to csv
        X.to_csv('../data/all-datasets/cleaned_data_dropped.csv')

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        # # Resetting indices for easier referencing
        X_train = X_train.reset_index()
        X_test = X_test.reset_index()

        # Resetting numerical columns after dropping columns
        numerical_vals, categorical_vals = self._reset_data_types(X)

        # Filling in null values using a gridsearched linear model
        for column in numerical_vals.columns:
            # Instantiating model to predict values
            el = ElasticNet()
            # param_list = {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]}
            param_list = {'alpha': [0.1], 'l1_ratio': [0.5]}
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

        # Force y values into integers for algorithms (not floats)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Feature selection using l1 logistic regression
        model = LogisticRegression(penalty='l1', C=5)
        model.fit(X_train, y_train)
        coefs = list(model.coef_)[0]
        features = list(X_train.columns)
        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])
            best_features = [x for x in importances if x[1] != 0.0]
            best_features = [x[0] for x in best_features]
            X_train_reduced = X_train[best_features]
            X_test_reduced = X_test[best_features]

        # Scaling train data
        scaler = StandardScaler()
        # Resetting numerical values
        numerical_vals, categorical_vals = self._reset_data_types(X_train_reduced)
        # Scaling data
        X_train_scaled = scaler.fit_transform(X_train_reduced[numerical_vals.columns])
        X_train_scaled = np.concatenate(
            [X_train_scaled, X_train_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        # # Using SMOTE to generate extra synthetic samples of the smaller class
        # X_train_resampled, y_train_resampled = SMOTE().fit_sample(X_train_scaled, y_train)

        # Scaling test data
        X_test_scaled = scaler.transform(X_test_reduced[numerical_vals.columns])
        X_test_scaled = np.concatenate(
            [X_test_scaled, X_test_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train_reduced

    def feature_selection(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train, plot=False):
        model = LogisticRegression(C=5)
        model.fit(X_train_scaled, y_train)
        # y_test_predicted = model.predict(X_test_scaled)
        best_score = np.mean(cross_val_score(model, X_train_scaled,
                                             y_train, scoring='f1_weighted', cv=5))

        change_f1 = []
        bar = progressbar.ProgressBar()
        # Cycling through all features
        for i in bar(range(len(X_train.columns))):
            model = LogisticRegression(C=5)
            # Dropping one feature to see how the model performs
            temp_X_train = np.delete(X_train_scaled, i, 1)
            # temp_X_test = np.delete(X_test_scaled, i, 1)
            # Fitting the model with all features minus one
            model.fit(temp_X_train, y_train)
            # Predicting class
            # y_test_predicted = model.predict(temp_X_test)
            # Scoring the model
            # f1 = f1_score(y_test, y_test_predicted, average='weighted')
            f1 = np.mean(cross_val_score(model, temp_X_train,
                                         y_train, scoring='f1_weighted', cv=5))
            # Creating a list of change in f1 scores based on dropping features
            # More negative numbers = f score drops by that amount without that variable
            change_f1.append(f1 - best_score)

        feature_importances = pd.Series(change_f1, index=X_train.columns)
        feature_importances = feature_importances.sort_values()

        if plot:
            mpl.rcParams.update({
                'font.size': 16.0,
                'axes.titlesize': 'large',
                'axes.labelsize': 'medium',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'legend.fontsize': 'small',
            })

            fig = plt.figure(figsize=(50, 200))
            ax = fig.add_subplot(111)
            ax.set_title('Feature Importances')
            feature_importances.plot(kind='barh')
            plt.savefig('../images/feature_importances.png')

            negative_impact = feature_importances[feature_importances < 0.0]
            fig = plt.figure(figsize=(50, 80))
            ax = fig.add_subplot(111)
            ax.set_title('Negative Impact When Dropped')
            negative_impact.plot(kind='barh')
            plt.savefig('../images/negative_impact.png')

        columns = list(X_train.columns)
        indices = [columns.index(feature) for impact, feature in zip(
            feature_importances, feature_importances.index) if impact < 0.0][:20]
        X_train_scaled = X_train_scaled[:, indices]
        X_test_scaled = X_test_scaled[:, indices]
        X_train = X_train[[columns[x] for x in indices]]

        return X_train_scaled, X_test_scaled, X_train

    def model_testing(self, model, X_train_scaled, y_train, X_train):
        if model == 'LogisticRegression':
            model = LogisticRegression()
            # param_list = {'penalty': ['l1', 'l2'], 'C': [1, 5, 10, 15]}
            # param_list = {'penalty': ['l1', 'l2'], 'C': np.arange(5, 25, 2)}
            param_list = {'C': np.arange(5, 25, 2)}
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
                         cv=5, n_jobs=3, verbose=10)
        g.fit(X_train_scaled, y_train)
        results = g.cv_results_
        print('\n\n')
        pprint(results)
        print('\n\n')
        print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))

        coefs = list(g.best_estimator_.coef_)[0]
        self.print_coefficients(X_train, coefs)

    def final_model(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train):
        model = LogisticRegression(C=5)
        model.fit(X_train_scaled, y_train)
        y_test_predicted = model.predict(X_test_scaled)
        print('F1 Score: {}'.format(f1_score(y_test, y_test_predicted, average='weighted')))
        print('Precision: {}'.format(precision_score(y_test, y_test_predicted, average='weighted')))
        print('Recall: {}'.format(recall_score(y_test, y_test_predicted, average='weighted')))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_test_predicted)))
        coefs = list(model.coef_)[0]
        self.print_coefficients(X_train, coefs)
        # coefs = list(model.feature_importances_)
        # self.print_tree(model, X_train)

    def print_coefficients(self, X_train, coefs):
        features = list(X_train.columns)

        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])

        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        print('\n\n')
        print('Coefficients:')
        for pair in importances[:1000]:
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
    np.savez_compressed('../data/all-datasets/Xycompressed_classification', **args)
    X_train.to_pickle('../data/all-datasets/X_train_classification')

    # Loading compressed data
    npz = np.load('../data/all-datasets/Xycompressed_classification.npz')
    X_train_scaled = npz['X_train_scaled']
    X_test_scaled = npz['X_test_scaled']
    y_train = npz['y_train']
    y_test = npz['y_test']
    X_train = pd.read_pickle('../data/all-datasets/X_train_classification')

    X_train_scaled, X_test_scaled, X_train = dc.feature_selection(
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, plot=True)

    # dc.model_testing('LogisticRegression', X_train_scaled, y_train, X_train)

    dc.final_model(X_train_scaled, X_test_scaled, y_train, y_test, X_train)

    #################################################
    # F1 Score: 0.755425683687
    # Precision: 0.766511464274
    # Recall: 0.818433179724
    # Accuracy: 0.818433179724
    #
    # Coefficients:
    # ['edd:state_MB', 2.6519095608306498]
    # ['drip:time_zone_America/Nassau', 2.4877312147840009]
    # ['turk:answer.industry_retail_1.0', 2.3909167870753332]
    # ['hubcomp:time_zone_Australia/Sydney', -1.8881449890797348]
    # ['intercom:country_Mexico', -1.8707820542177354]
    # ['intercom:industry_Energy & Environment', -1.7644646889223288]
    # ['edd:payment_method_paypal_standard_1.0', 1.3099815869999256]
    # ['edd:payment_method_paypal_pro_1.0', 1.3092051813194345]
    # ['hubcomp:industry_Government Administration', -1.282342635142838]
    # ['edd:payment_method_stripe_1.0', 1.1736176864671901]
    # ['hubcust:country_HKG', 1.1320574068555984]
    # ['edd:state_MD', 1.0181179263874418]
    # ['intercom:industry_Fashion & Apparel', 0.96561081246881053]
    # ['hubcomp:industry_Construction', 0.82226976042569033]
    # ['edd:country_GB', -0.62774926579134571]
    # ['drip:time_zone_Australia/Brisbane', 0.32202621380000584]
    # ['helpscout:number_support_tickets', 0.30683121060736074]
    # ['edd:country_US', 0.18009564948901546]
    # ['hubcomp:state/region_DC', -0.067191259390299268]
    # ['intercom:browser_safari', -0.057335901207023733]
    ##################################################
    # Logistic regression
    # Best Params: {'penalty': 'l2', 'C': 5}, Best Score: 0.777238452282
    ##################################################
    # Decision tree
    # Best Params: {'max_features': 0.79999999999999993, 'min_samples_split': 0.1, 'criterion': 'entropy', 'max_depth': 79}, Best Score: 0.802447822331
    ##################################################
    # SVC (linear)
    # Best Params: {'C': 5}, Best Score: 0.767679168076
    ##################################################
    # SGDClassifier
    # Best Params: {'alpha': 0.00091000000000000011}, Best Score: 0.790229391565
    ##################################################
    # KNeighborsClassifier
    # Best Params: {'n_neighbors': 2}, Best Score: 0.860791628082 -- overfit
