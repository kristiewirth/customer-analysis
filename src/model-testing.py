import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import ElasticNet, SGDRegressor, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers.core import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def scaling(X_train):
    # Separating numerical for scaling
    numerical_vals = X_train.select_dtypes(include=['float64', 'int64'])
    categorical_vals = X_train.select_dtypes(exclude=['float64', 'int64'])

    # Scaling train data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numerical_vals.columns])
    X_train_scaled = np.concatenate([X_train_scaled, X_train[categorical_vals.columns]], axis=1)
    return X_train_scaled


def clustering(X_train, y_train, graph=False):
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

    return X_train


def lasso_feature_selection(X_train_scaled, X_train, y_train):
    # # Testing white box models
    model = Lasso()
    param_list = {'alpha': [0.1]}

    # Best features (according to Lasso)
    g = GridSearchCV(model, param_list, scoring='r2',
                     cv=5, n_jobs=-1, verbose=0)
    g.fit(X_train_scaled, y_train)
    coefs = list(g.best_estimator_.coef_)
    features = list(X_train.columns)
    importances = []
    for x, y in zip(features, coefs):
        importances.append([x, y])

    best_features = [x for x in importances if x[1] != 0.0]
    best_features = [x[0] for x in best_features]

    X_train_reduced = X_train[best_features]
    X_train_reduced = scaling(X_train_reduced)
    return X_train_reduced


def create_model(num_neurons=10, optimizer='adam', kernel_initializer='uniform', activation='relu'):
    # available activation functions at: options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model = Sequential()  # sequence of layers

    # Set input_dim to the number of features
    # Dense class defines new layers
    # First argument = number of neurons in the layer
    # Set kernel_initializer to the chosen intialization method for the layer
    # Either use uniform or normal method
    # Set activation to the chosen activation function
    # Use sigmoid on the last output layer to map predictions between 0 and 1

    num_inputs = X_train.shape[1]  # number of features
    # num_classes = len(np.unique(y_train))  # number of classes of target, can be 0-9
    num_neurons_in_layer = num_neurons

    # 1st layer - input layer
    # number of neurons = number of features
    model.add(Dense(input_dim=num_inputs,
                    units=num_neurons_in_layer,
                    kernel_initializer=kernel_initializer,
                    activation=activation))

    # 2nd layer - hidden layer; very rarely would you use more than one layer
    model.add(Dense(input_dim=num_neurons_in_layer,
                    units=num_neurons_in_layer,
                    kernel_initializer=kernel_initializer,
                    activation=activation))

    # 3rd layer - output layer
    # Number of neurons = 1 for regression
    # Number of neurons = 1 for classication unless using softmax,
    # then equal to number of classes
    model.add(Dense(input_dim=num_neurons_in_layer,
                    units=1,
                    kernel_initializer=kernel_initializer,
                    activation='sigmoid'))

    # Compile model
    # Change loss and metrics for a regression problem
    # Can play with the optimizer but adam is a good place to start
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Reading in the training dataset
    train_df = pd.read_pickle('../data/train_df_pickle')

    # Separating X and y values
    y_train = train_df.pop('revenue:purchase_value')
    X_train = train_df
    X_train.pop('intercom:job_title')

    # Scaling data
    X_train_scaled = scaling(X_train)

    # Feature selection
    # X_train_reduced = lasso_feature_selection(X_train_scaled, X_train, y_train)

    # Model testing
    # model = KNeighborsRegressor()
    # param_list = {'n_neighbors': [5, 10, 20]}
    #
    # model = DecisionTreeRegressor()
    # param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}
    #
    # model = SVR(kernel='linear')
    # param_list = {'epsilon': [0.1]}
    #
    # model = SGDRegressor()
    # param_list = {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l1', 'l2']}

    # model = RandomForestRegressor()
    # param_list = {'min_samples_split': [.25, .5, 1.0], 'max_depth': [5, 10, 20]}

    # Grid searching hyperparameters
    model = KerasClassifier(build_fn=create_model, verbose=0)
    num_neurons = int(np.mean([X_train.shape[1], len(np.unique(y_train))]))
    param_list = {'epochs': [10, 20, 30], 'batch_size': [30, 40, 50],
                  'optimizer': ['Adam'],
                  'num_neurons': [num_neurons]}

    #########################
    # Best model so far: ElasticNet
    # Best Params: {'alpha': 0.01, 'l1_ratio': 0.5}, Best R2 Score: 0.593484819464
    #########################

    g = GridSearchCV(model, param_list, scoring='r2',
                     cv=5, n_jobs=-1, verbose=10)
    g.fit(X_train_scaled, y_train)
    results = g.cv_results_
    print('\n\n')
    pprint(results)
    print('\n\n')
    print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))
    print('\n\n')
    #
    # coefs = list(g.best_estimator_.coef_)
    # features = list(X_train.columns)
    # importances = []
    # for x, y in zip(features, coefs):
    #     importances.append([x, y])
    #
    # importances.sort(key=lambda row: abs(row[1]), reverse=True)
    # print('Coefficients:')
    # for pair in importances[:15]:
    #     print(pair)
