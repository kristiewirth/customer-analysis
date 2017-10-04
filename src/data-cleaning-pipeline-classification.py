from anonymizingdata import AnonymizingData
from separatedatasourcecleaning import SeparateDataSets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression, ElasticNet
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from pprint import pprint
import progressbar
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from imblearn.over_sampling import SMOTE


class DataCleaning(object):

    def __init__(self):
        # Importing functions to anonymize data
        self.hidden = AnonymizingData()

    def intializing_data(self):
        '''
        Loads all separate data sets and merges into one dataset for modeling purposes
        '''

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
                           right_on='hubcust:email')
        self.df = pd.merge(self.df, hub_comp_df, how='left', left_on='revenue:domain',
                           right_on='hubcomp:company_domain_name')
        self.df = pd.merge(self.df, turk_df, how='left', left_on='revenue:domain',
                           right_on='turk:turk_domain')
        self.df = pd.merge(self.df, help_scout_df, how='left', left_on='revenue:email',
                           right_on='helpscout:emails')

    def cleaning_data(self):
        '''
        Cleans dataset by consolidating duplicate rows, removing test purchase rows,
        recoding variable types, and dropping columns that have leakage or large proportions of nulls
        '''
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
        self.df['helpscout:loss_per_customer'] = self.df['helpscout:days_open'] * 10.0
        self.df['adjusted_revenue'] = self.df['revenue:purchase_value'] - \
            self.df['helpscout:loss_per_customer']

        # Saving all cleaned data (before dropping columns) for graphing purposes later
        self.df.to_csv('../data/all-datasets/cleaned_data.csv', index=False)

        # Dropping a long list of columns that have large portions of nulls, are identifiers,
        # have only one value, those with leakage, and any date columns
        self.df = self.hidden.drop_columns(self.df)

    def _reset_data_types(self, X):
        '''
        Resets which columns have numerical values and which have categorical values after dropping columns
        '''
        # Selecting floats and integers
        numerical_vals = X.select_dtypes(include=['float64', 'int64'])
        for column in numerical_vals.columns:
            # Numerical columns that have less than 3 values are probably actually categories
            if len(numerical_vals[column].unique()) <= 3:
                numerical_vals.drop(column, inplace=True, axis=1)
        # Categorical values are any columns that are not numerical
        categorical_vals = X.drop(numerical_vals, axis=1)
        return numerical_vals, categorical_vals

    def modeling_prep(self):
        '''
        Prepares the dataset for modeling by creating dummy variables, dropping any remaining less useful columns,
        creating a train test split, scaling the data, and filling null values
        '''
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

        # Dropping dummy columns that are not useful in the final model due to leakage, collinearity, or simply indicate unknown values
        X.drop(["intercom:industry_Not Online/Can't Access",
                'helpscout:gender_x_male', 'hubcust:original_source_drill-down_1_INTEGRATION',
                'hubcomp:original_source_data_2_contact-upsert', 'hubcust:original_source_drill-down_2_contact-upsert',
                'helpscout:phototype_y_gravatar', 'hubcomp:original_source_data_1_INTEGRATION', 'turk:answer.industry_other_1.0', 'edd:payment_method_paypal_pro_1.0',
                'edd:payment_method_paypal_standard_1.0', 'edd:payment_method_stripe_1.0'], inplace=True, axis=1)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        # # Resetting indices for easier referencing
        X_train = X_train.reset_index()
        X_test = X_test.reset_index()
        # Dropping index variable that shows up after reset
        X_train.drop('index', inplace=True, axis=1)
        X_test.drop('index', inplace=True, axis=1)

        # Resetting numerical columns after dropping columns
        numerical_vals, categorical_vals = self._reset_data_types(X)

        # Filling null values using a smaller regression model
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

        # # Scaling train data - commented out due to dropping all numerical features after l1 regression!
        # scaler = StandardScaler()
        # # Resetting numerical values
        # numerical_vals, categorical_vals = self._reset_data_types(X_train_reduced)
        # # Scaling data
        # X_train_scaled = scaler.fit_transform(X_train_reduced[numerical_vals.columns])
        # X_train_scaled = np.concatenate(
        #     [X_train_scaled, X_train_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        # # Using SMOTE to generate extra synthetic samples of the smaller class -- did not help with final scoring
        # X_train_resampled, y_train_resampled = SMOTE().fit_sample(X_train_scaled, y_train)

        # # Scaling test data
        # X_test_scaled = scaler.transform(X_test_reduced[numerical_vals.columns])
        # X_test_scaled = np.concatenate(
        #     [X_test_scaled, X_test_reduced.drop(numerical_vals.columns, axis=1)], axis=1)

        # Transforming X data into numpy arrays for easier calculations later after scaling is no longer needed
        X_train_scaled = np.array(X_train_reduced)
        X_test_scaled = np.array(X_test_reduced)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train_reduced

    def feature_selection(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train, plot=False):
        '''
        Selects only the twenty best features, defined as those variables that create the greatest drop in f1 score when left out of the model
        '''
        # Intantiating regression model
        model = LogisticRegression(C=17)
        model.fit(X_train_scaled, y_train)
        # Getting a baseline best score to compare to using cross validation
        best_score = np.mean(cross_val_score(model, X_train_scaled,
                                             y_train, scoring='f1_weighted', cv=5))

        # Creating an empty list of change in f1 score to append to
        change_f1 = []
        bar = progressbar.ProgressBar()
        # Cycling through all features
        for i in bar(range(len(X_train.columns))):
            model = LogisticRegression(C=17)
            # Dropping one feature to see how the model performs
            temp_X_train = np.delete(X_train_scaled, i, 1)
            # Fitting the model with all features minus one
            model.fit(temp_X_train, y_train)
            # Scoring the model using cross validation
            f1 = np.mean(cross_val_score(model, temp_X_train,
                                         y_train, scoring='f1_weighted', cv=5))
            # Creating a list of change in f1 scores based on dropping features
            # More negative numbers = f score drops by that amount without that variable
            change_f1.append(f1 - best_score)

        # Creating a pandas series with the features associated with their respective changes in f1 scores
        feature_importances = pd.Series(change_f1, index=X_train.columns)
        feature_importances = feature_importances.sort_values()

        # Plotting changes in f1 scores
        if plot:
            # Setting up default larger font sizes for graphs
            mpl.rcParams.update({
                'font.size': 16.0,
                'axes.titlesize': 'large',
                'axes.labelsize': 'medium',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'legend.fontsize': 'small',
            })

            # Creating a graph of all f1 score changes
            fig = plt.figure(figsize=(50, 200))
            ax = fig.add_subplot(111)
            ax.set_title('Feature Importances')
            feature_importances.plot(kind='barh')
            plt.savefig('../images/feature_importances.png')

            # Creating a graph of f1 score changes that are negative (and thus may be important in the model)
            negative_impact = feature_importances[feature_importances < 0.0]
            fig = plt.figure(figsize=(50, 80))
            ax = fig.add_subplot(111)
            ax.set_title('Negative Impact When Dropped')
            negative_impact.plot(kind='barh')
            plt.savefig('../images/negative_impact.png')

        # Getting features in list form
        columns = list(X_train.columns)
        # Grabbing the 20 features that had the biggest impact on f1 score when left out
        indices = [columns.index(feature) for impact, feature in zip(
            feature_importances, feature_importances.index) if impact < 0.0][:20]
        # Resetting the X variables to only include the top 20 features
        X_train_scaled = X_train_scaled[:, indices]
        X_test_scaled = X_test_scaled[:, indices]
        X_train = X_train[[columns[x] for x in indices]]

        return X_train_scaled, X_test_scaled, X_train

    def model_testing(self, model, X_train_scaled, y_train, X_train):
        '''
        Uses grid searching to test different models and different hyperparameters
        '''
        if model == 'LogisticRegression':
            model = LogisticRegression()
            # param_list = {'penalty': ['l1', 'l2'], 'C': [1, 5, 10, 15]}
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
        # Fitting grid searched model
        g.fit(X_train_scaled, y_train)
        # Printing detailed results from gridsearching
        results = g.cv_results_
        print('\n\n')
        pprint(results)
        print('\n\n')
        # Printing best hyperparameters and the associated weighted f1 score
        print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))

        # Printing coefficients of the features used
        coefs = list(g.best_estimator_.coef_)[0]
        self.print_coefficients(X_train, coefs)

    def final_model(self, X_train_scaled, X_test_scaled, y_train, y_test, X_train):
        '''
        Fits best model using best hyperparameters from grid search on the test data and calculates final metrics
        '''
        model = LogisticRegression(C=17)
        model.fit(X_train_scaled, y_train)
        y_test_predicted = model.predict(X_test_scaled)
        # Scoring model using weighted scores due to imbalanced class
        print('F1 Score: {}'.format(f1_score(y_test, y_test_predicted, average='weighted')))
        print('Precision: {}'.format(precision_score(y_test, y_test_predicted, average='weighted')))
        print('Recall: {}'.format(recall_score(y_test, y_test_predicted, average='weighted')))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_test_predicted)))
        # Printing coefficients of features used
        coefs = list(model.coef_)[0]
        self.print_coefficients(X_train, coefs)
        # Commented out options for decision trees
        # coefs = list(model.feature_importances_)
        # self.print_tree(model, X_train)
        return y_test_predicted, y_test

    def print_coefficients(self, X_train, coefs):
        '''
        Prints coefficients model in order of highest values
        '''
        # Creating a list of features
        features = list(X_train.columns)

        importances = []
        for x, y in zip(features, coefs):
            # Connecting features with their corresponding coefficients
            importances.append([x, y])

        # Sort coefficients in decreasing order of absolute values of the coefficients
        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        # Cycling through the list to print for nicer formatting
        print('Coefficients:')
        for pair in importances:
            if pair[1] == 0.0:
                break
            else:
                print(pair)

    # def print_tree(self, model, X_train):
    #     '''
    #     Optional function to create a visual depiction of a decision tree model
    #     '''
    #     # Exporting text form of decision tree
    #     export_graphviz(model, out_file='decision-tree.dot', feature_names=X_train.columns)
    #     # Converting text to a visual png file
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
    # np.savez_compressed('../data/all-datasets/Xycompressed_classification', **args)
    # X_train.to_pickle('../data/all-datasets/X_train_classification')

    # Loading compressed and processed data
    npz = np.load('../data/all-datasets/Xycompressed_classification.npz')
    X_train_scaled = npz['X_train_scaled']
    X_test_scaled = npz['X_test_scaled']
    y_train = npz['y_train']
    y_test = npz['y_test']
    X_train = pd.read_pickle('../data/all-datasets/X_train_classification')

    X_train_scaled, X_test_scaled, X_train = dc.feature_selection(
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, plot=True)

    # dc.model_testing('LogisticRegression', X_train_scaled, y_train, X_train)

    y_test_predicted, y_test = dc.final_model(
        X_train_scaled, X_test_scaled, y_train, y_test, X_train)

    #################################################
    # Printed Results:
    # F1 Score: 0.743584054581
    # Precision: 0.732450544689
    # Recall: 0.812903225806
    # Accuracy: 0.812903225806
    #
    # Coefficients:
    # ['turk:answer.industry_retail_1.0', 4.0302363383517417]
    # ['drip:time_zone_America/Anchorage', 3.6834776982065898]
    # ['drip:time_zone_America/Nassau', 3.6834776982065898]
    # ['drip:time_zone_Asia/Beirut', 3.0729742025156392]
    # ['intercom:industry_Consulting', 2.475049013510481]
    # ['drip:time_zone_America/Argentina/Buenos_Aires', -2.088771196421217]
    # ['turk:answer.industry_events_services_1.0', 2.0810199188826477]
    # ['drip:time_zone_UTC', 2.0810199188826477]
    # ['intercom:industry_Employment', 1.7911025393175044]
    # ['intercom:industry_Security', 1.4826125036485247]
    # ['hubcomp:industry_Recreational Facilities and Services', -1.4533954589597129]
    # ['hubcomp:industry_Industrial Automation', 1.4033048434380999]
    # ['hubcomp:industry_Insurance', 1.4033048434380999]
    # ['turk:answer.industry_non-profit_organization_management_1.0', 1.3813631898102734]
    # ['drip:time_zone_Pacific/Auckland', 1.1445723585807723]
    # ['intercom:industry_Fashion & Apparel', 1.1061725247647098]
    # ['intercom:industry_Real Estate', 0.81556118158597257]
    # ['drip:time_zone_America/Phoenix', 0.62796255167596615]
    # ['intercom:industry_Sports & Leisure', 0.42756889922925145]
    # ['intercom:industry_Events & Conferences', 0.37140063066828877]
