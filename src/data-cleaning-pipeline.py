from anonymizingdata import AnonymizingData
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, SGDRegressor, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt


class DataCleaning(object):

    def __init__(self):
        pass

    def _intializing_data(self):
        # Importing functions to anonymize data
        self.hidden = AnonymizingData()

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

        # Dropping a long list of columns that have large portions of nulls, identifiers,
        # have only one value, those with leakage, date columns
        self.df = self.hidden.drop_columns(self.df)

        # Resetting columns incorrectly coded as numerical
        self.df[['edd:payment_method_paypal_pro',  'edd:payment_method_manual_purchases',
                 'edd:payment_method_stripe', 'edd:payment_method_paypal_standard',
                 'edd:payment_method_amazon', 'edd:payment_method_paypal_pro',
                 'edd:payment_method_paypal_standard', 'edd:payment_method_manual_purchases', 'edd:payment_method_stripe']] = self.df[[
                     'edd:payment_method_paypal_pro',  'edd:payment_method_manual_purchases',
                     'edd:payment_method_stripe', 'edd:payment_method_paypal_standard',
                     'edd:payment_method_amazon', 'edd:payment_method_paypal_pro',
                     'edd:payment_method_paypal_standard', 'edd:payment_method_manual_purchases', 'edd:payment_method_stripe']] .astype(object)

    def _modeling_prep(self):
        # Creating X & y variables
        y = self.df.pop('revenue:purchase_value')
        X = self.df

        # Sorting self.df into numerical and not for scaler
        numerical_vals = X.select_dtypes(include=['float64', 'int64'])
        categorical_vals = X.select_dtypes(exclude=['float64', 'int64'])

        # Quick fill numerical null values
        for col in numerical_vals:
            X[col] = X[col].fillna(X[col].mean())

        # Create dummy variables
        for column in categorical_vals.columns:
            try:
                X = pd.get_dummies(X, drop_first=True, dummy_na=False, columns=[column])
            except:
                pass

        # Dropping dummy columns for nans or unknowns or specific emails
        for column in X.columns:
            if '_nan' in column or '_unknown' in column or '@' in column:
                X.drop(column, inplace=True, axis=1)
            else:
                pass

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Best features (according to Lasso)
        model = Lasso(alpha=0.1)
        model.fit(X_train, y_train)
        coefs = list(model.coef_)
        features = list(X_train.columns)
        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])
            best_features = [x for x in importances if x[1] != 0.0]
            best_features = [x[0] for x in best_features]
            X_train_reduced = X_train[best_features]

        # Scaling train data
        scaler = StandardScaler()
        # Resetting numerical values
        numerical_vals = X.select_dtypes(include=['float64', 'int64'])
        X_train_scaled = scaler.fit_transform(X_train_reduced[numerical_vals.columns])
        X_train_scaled = np.concatenate(
            [X_train_scaled, X_train_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        # Scaling test data
        X_test_scaled = scaler.transform(X_test[numerical_vals.columns])
        X_test_scaled = np.concatenate(
            [X_test_scaled, X_test.drop(numerical_vals.columns, axis=1)], axis=1)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train

    def _model_testing(self, model, X_train_scaled, y_train, X_train):
        if model == 'ElasticNet':
            model = ElasticNet()
            param_list = {'alpha': [0.0001, 0.001, 0.01]}
        elif model == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
            param_list = {'n_neighbors': [5, 10, 20]}
        elif model == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor()
            param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}
        elif model == 'SGDRegressor':
            model = SGDRegressor()
            param_list = {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l1', 'l2']}
        elif model == 'SVR':
            model = SVR(kernel='linear')
            param_list = {'epsilon': [0.1]}
        elif model == 'RandomForestRegressor':
            model = RandomForestRegressor()
            param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}

        # Grid searching hyperparameters
        g = GridSearchCV(model, param_list, scoring='r2',
                         cv=5, n_jobs=-1, verbose=10)
        g.fit(X_train_scaled, y_train)
        results = g.cv_results_
        print('\n\n')
        pprint(results)
        print('\n\n')
        print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))
        print('\n\n')

        coefs = list(g.best_estimator_.coef_)
        features = list(X_train.columns)
        importances = []
        for x, y in zip(features, coefs):
            importances.append([x, y])

        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        print('Coefficients:')
        for pair in importances[:15]:
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
            X_train['x'] = pca.fit_transform(X_train)[:, 0]
            X_train['y'] = pca.fit_transform(X_train)[:, 1]
            X_train = X_train.reset_index()
            sns.lmplot('x', 'y', data=X_train, hue='cluster', fit_reg=False)
            plt.savefig('../images/customer_clusters.png', dpi=300)

    def run_pipeline(self, model='ElasticNet'):
        self._intializing_data()
        self._cleaning_data()
        X_train_scaled, X_test_scaled, y_train, y_test, X_train = self._modeling_prep()
        self._model_testing(model, X_train_scaled, y_train, X_train)


if __name__ == '__main__':
    dc = DataCleaning()
    dc.run_pipeline(model='ElasticNet')

    #########################
    # Best model so far: ElasticNet
    # Best Params: {'alpha': 0.01, 'l1_ratio': 0.5}, Best R2 Score: 0.593484819464
    #########################
