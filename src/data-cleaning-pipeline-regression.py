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
from sklearn.metrics import mean_squared_error
from pprint import pprint
import progressbar
import sys
import pudb
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import cross_val_score


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
        y = self.df.pop('revenue:purchase_value')
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
                'helpscout:gender_x_male', 'hubcust:original_source_drill-down_1_INTEGRATION',
                'hubcomp:original_source_data_2_contact-upsert'], inplace=True, axis=1)

        # Exporting data to csv
        X.to_csv('../data/all-datasets/cleaned_data_dropped.csv')

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # # # Resetting indices for easier referencing
        X_train = X_train.reset_index()
        X_test = X_test.reset_index()

        X_train.drop('index', inplace=True, axis=1)
        X_test.drop('index', inplace=True, axis=1)

        # Resetting numerical columns after dropping columns
        numerical_vals, categorical_vals = self._reset_data_types(X)

        # Filling in null values using a gridsearched linear model
        for column in numerical_vals.columns:
            # Getting indices in which the given column is not null
            filled = list(X_train[column].dropna().index)
            # Getting all the X train data for the rows in which the given column is not blank
            mini_training_data = X_train.drop(column, axis=1).apply(
                lambda x: x.fillna(x.mean()), axis=0).iloc[filled]
            # Getting the column to be filled data where it is not null
            mini_training_target = X_train[column].iloc[filled]
            # Instantiating model to predict values
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            model.fit(mini_training_data, mini_training_target)
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

        # Best features (according to Lasso)
        model = Lasso(alpha=0.00091)
        model.fit(X_train, y_train)
        coefs = list(model.coef_)
        features = list(X_train.columns)
        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])
        best_features = [x for x in importances if x[1] != 0.0]
        best_features = [x[0] for x in best_features]
        X_train_reduced = X_train[best_features]
        X_test_reduced = X_test[best_features]

        X_train_scaled = np.array(X_train_reduced)
        X_test_scaled = np.array(X_test_reduced)

        # # Scaling train data
        # scaler = StandardScaler()
        # # Resetting numerical values
        # numerical_vals, categorical_vals = self._reset_data_types(X_train_reduced)
        # # Scaling data
        # X_train_scaled = scaler.fit_transform(X_train_reduced[numerical_vals.columns])
        # X_train_scaled = np.concatenate(
        #     [X_train_scaled, X_train_reduced.drop(numerical_vals.columns, axis=1)], axis=1)
        #
        # # Scaling test data
        # X_test_scaled = scaler.transform(X_test_reduced[numerical_vals.columns])
        # X_test_scaled = np.concatenate(
        #     [X_test_scaled, X_test_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train_reduced

    def model_testing(self, model, X_train_scaled, y_train, X_train):
        if model == 'ElasticNet':
            model = ElasticNet()
            # param_list = {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]}
            param_list = {'alpha': np.arange(0.00001, 0.001, 0.0001),
                          'l1_ratio': np.arange(0.1, 1.0, 0.1)}
            # param_list = {'alpha': np.arange(0.00001, 0.001, 0.0001)}
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
            # param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}
            param_list = {'min_samples_split': np.arange(
                0.1, 1.0, 0.1), 'max_depth': np.arange(5, 25, 2)}

        # Grid searching hyperparameters
        g = GridSearchCV(model, param_list, scoring='neg_mean_squared_error',
                         cv=5, n_jobs=3, verbose=10)
        g.fit(X_train_scaled, y_train)
        results = g.cv_results_
        print('\n\n')
        pprint(results)
        print('\n\n')
        RMSE = (-g.best_score_)**0.5
        print('Best Params: {}, Best Score: {}'.format(g.best_params_, RMSE))

        coefs = list(g.best_estimator_.coef_)
        self.print_coefficients(X_train, coefs)

    def feature_selection(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train, plot=False):
        model = ElasticNet(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        # y_test_predicted = model.predict(X_test_scaled)
        best_score = np.mean(cross_val_score(model, X_train_scaled,
                                             y_train, scoring='neg_mean_squared_error', cv=5))
        best_score = (-best_score)**0.5

        change_rmse = []
        bar = progressbar.ProgressBar()
        # Cycling through all features
        for i in bar(range(len(X_train.columns))):
            model = ElasticNet(alpha=1.0)
            # Dropping one feature to see how the model performs
            temp_X_train = np.delete(X_train_scaled, i, 1)
            # temp_X_test = np.delete(X_test_scaled, i, 1)
            # Fitting the model with all features minus one
            model.fit(temp_X_train, y_train)
            # Predicting class
            # y_test_predicted = model.predict(temp_X_test)
            # Scoring the model
            # f1 = f1_score(y_test, y_test_predicted, average='weighted')
            rmse = np.mean(cross_val_score(model, temp_X_train,
                                           y_train, scoring='neg_mean_squared_error', cv=5))
            rmse = (-rmse)**0.5
            # Creating a list of change in f1 scores based on dropping features
            # More positive numbers = rmse increased without variable
            change_rmse.append(rmse - best_score)

        feature_importances = pd.Series(change_rmse, index=X_train.columns)
        feature_importances = feature_importances.sort_values(ascending=False)

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

            negative_impact = feature_importances[feature_importances > 0.0]
            fig = plt.figure(figsize=(50, 80))
            ax = fig.add_subplot(111)
            ax.set_title('Negative Impact When Dropped')
            negative_impact.plot(kind='barh')
            plt.savefig('../images/negative_impact.png')

        columns = list(X_train.columns)
        indices = [columns.index(feature) for impact, feature in zip(
            feature_importances, feature_importances.index) if impact > 0.0][:21]
        X_train_scaled = X_train_scaled[:, indices]
        X_test_scaled = X_test_scaled[:, indices]
        X_train = X_train[[columns[x] for x in indices]]

        return X_train_scaled, X_test_scaled, X_train

    def final_model(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train):
        model = ElasticNet(alpha=0.00081, l1_ratio=0.1)
        model.fit(X_train_scaled, y_train)
        y_test_predicted = model.predict(X_test_scaled)
        RMSE = mean_squared_error(y_test, y_test_predicted)
        RMSE = (RMSE)**0.5
        print('RMSE: ${}'.format(RMSE))
        coefs = list(model.coef_)
        self.print_coefficients(X_train, coefs)
        return y_test_predicted, y_test
        # coefs = list(model.feature_importances_)
        # self.print_tree(model, X_train)

    def print_coefficients(self, X_train, coefs):
        features = list(X_train.columns)

        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])

        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        pd.DataFrame(importances).to_csv('../data/coefficients.csv')
        print('\n')
        print('Coefficients:')
        for pair in importances[:1000]:
            if pair[1] == 0.0:
                break
            else:
                print(pair)
    #
    # def print_tree(self, model, X_train):
    #     export_graphviz(model, out_file='decision-tree.dot', feature_names=X_train.columns)
    #     os.system('dot -Tpng decision-tree.dot -o decision-tree.png')


if __name__ == '__main__':
    dc = DataCleaning()

    # dc.intializing_data()
    # dc.cleaning_data()
    # X_train_scaled, X_test_scaled, y_train, y_test, X_train = dc.modeling_prep()
    #
    # # Compressing data for faster performance
    # args = {'X_train_scaled': X_train_scaled,
    #         'X_test_scaled': X_test_scaled, 'y_train': y_train, 'y_test': y_test}
    # np.savez_compressed('../data/all-datasets/Xycompressed_regression', **args)
    # X_train.to_pickle('../data/all-datasets/X_train_regression')

    # Loading compressed data
    npz = np.load('../data/all-datasets/Xycompressed_regression.npz')
    X_train_scaled = npz['X_train_scaled']
    X_test_scaled = npz['X_test_scaled']
    y_train = npz['y_train']
    y_test = npz['y_test']
    X_train = pd.read_pickle('../data/all-datasets/X_train_regression')

    X_train_scaled, X_test_scaled, X_train = dc.feature_selection(
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, plot=True)

    # dc.model_testing('ElasticNet', X_train_scaled, y_train, X_train)

    y_test_predicted, y_test = dc.final_model(
        X_train_scaled, X_test_scaled, y_train, y_test, X_train)

    #pd.DataFrame(zip(y_test_predicted, y_test)).to_csv('../data/predictions-true.csv')
    ###################################################
    # ElasticNet - revenue
    # Best Params: {'alpha': 0.00081000000000000006, 'l1_ratio': 0.10000000000000001},
    # RMSE: $104.818570694
    #
    # Coefficients:
    # ['edd:payment_method_stripe_1.0', 113.60430712519899]
    # ['edd:payment_method_paypal_standard_1.0', 99.834188013295531]
    # ['intercom:industry_Online Education', 88.145299983318409]
    # ['edd:payment_method_paypal_pro_1.0', 84.616322091213334]
    # ['drip:time_zone_America/Chicago', 66.097734606401374]
    # ['drip:time_zone_America/Denver', 43.675445865119904]
    # ['drip:time_zone_Australia/Sydney', 38.67188755898168]
    # ['intercom:industry_Medical, Health & Fitness', 38.329615281732714]
    # ['intercom:timezone_America/Chicago', -33.711762719694377]
    # ['intercom:timezone_America/New_York', -33.527105759993134]
    # ['drip:time_zone_Europe/Berlin', 32.900408700808981]
    # ['intercom:browser_language_en', 28.475373570320833]
    # ['drip:time_zone_America/New_York', 27.0101856095146]
    # ['intercom:industry_Government & Association', 26.357576291036736]
    # ['drip:time_zone_Europe/London', 24.333782670507272]
    # ['hubcomp:industry_Higher Education', 21.645822234785584]
    # ['hubcomp:country_United States', 15.703149458786957]
    # ['edd:country_US', 8.1862614071765147]
    # ['helpscout:gender_y_male', 6.6784447142789398]
    # ['edd:country_CA', 6.0097123275508899]
    # ['helpscout:phototype_x_gravatar', -1.1768087541002885]
