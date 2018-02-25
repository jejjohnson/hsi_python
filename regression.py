import numpy as np 
import time
import warnings
from scipy import stats
import pandas as pd
from sklearn.externals import joblib
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  LassoCV,
                                  ElasticNetCV)
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error,
                             r2_score)

warnings.simplefilter('ignore')

# TODO - Radius NN
# TODO - Boosted Random Forest
# TODO - Relevance Vector Machanes
# TODO - GPR Heteroscedastic Kernel
# TODO - Nystrom KRR
# TODO - RFF KRR
# TODO - Implement Better save strategy. (Save after each fit)
# TODO - Implement a Finish fit strategy in case of failure
# TODO - Implement with trials with different random states


class SimpleR(object):
    def __init__(self, models_list=None, n_jobs=1, random_state=None):
        self.models_list = models_list
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.predictions = dict()
        self.algorithm_names = {
            'ols': 'Linear Regresssion',
            'polyr': 'Polynomial Regression',
            'rls': 'Ridge Regression',
            'lasso': 'LASSO',
            'elastic': 'Elastic Net',
            'knn': 'K-Nearest Neighbors',
            'rnn': 'Radius Nearest Neighbors',
            'dt': 'Decision Trees',
            'bag': 'Bagged Trees',
            'boost': 'Boosted Trees',
            'rf': 'Random Forest Regression',
            'svm': 'Support Vector Regression',
            'krr': 'Kernel Ridge Regression',
            'gpr': 'Gaussian Process Regression'
        }
    
    def fit_models(self, x_train, y_train, save=True):

        self.Models = dict()
        self.fit_times = dict()
        
        # TODO: check model list for bad inputs
        # TODO: case for all algorithms
        # convert all strings in list to lowercase
        self.models_list = [x.lower() for x in self.models_list]

        # TODO: check sizes of x and y
        np.testing.assert_equal(x_train.shape[0], y_train.shape[0])

        # -------------------
        # Linear Regression
        # -------------------
        if 'ols' in self.models_list or 'all' in self.models_list:
            algorithm = 'ols'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            self.Models[algorithm] = LinearRegression()
            self.Models[algorithm].fit(x_train, y_train)
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # --------------------
        # Polynomial Features
        # --------------------
        if 'polyr' in self.models_list or 'all' in self.models_list:
            algorithm = 'polyr'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'polyfeat__degree': [1, 2, 3, 4]
            }

            pipe = Pipeline([
                ('polyfeat', PolynomialFeatures()),
                ('linregress', LinearRegression())
            ])

            grid_search = GridSearchCV(
                pipe,
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)

            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # Ridge Regression
        # ------------------
        if 'rls' in self.models_list or 'all' in self.models_list:
            algorithm = 'rls'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'alpha': np.linspace(0.01, 10, 20)
            }

            grid_search = GridSearchCV(
                Ridge(),
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # LASSO
        # ------------------
        if 'lasso' in self.models_list or 'all' in self.models_list:
            algorithm = 'lasso'
            t0 = time.time()
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            
            alphas = np.linspace(0.01, 10, 20)

            self.Models[algorithm] = LassoCV(alphas=alphas, n_jobs=self.n_jobs)

            self.Models[algorithm].fit(x_train, y_train)
            
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # Elastic Net
        # ------------------
        if 'elastic' in self.models_list or 'all' in self.models_list:
            algorithm = 'elastic'
            t0 = time.time()
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            
            alphas = np.linspace(0.01, 10, 20)

            self.Models[algorithm] = ElasticNetCV(alphas=alphas, n_jobs=self.n_jobs)

            self.Models[algorithm].fit(x_train, y_train)
            
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # --------------------
        # K-Nearest Neighbors
        # --------------------
        if 'knn' in self.models_list or 'all' in self.models_list:
            algorithm = 'knn'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'n_neighbors': [1, 2, 3, 4, 5, 10, 25, 40, 50],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }

            grid_search = GridSearchCV(
                KNeighborsRegressor(),
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # Decision Trees
        # ------------------
        if 'dt' in self.models_list or 'all' in self.models_list:
            algorithm = 'dt'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'criterion': ['mse', 'mae'],
                'splitter': ['best', 'random']
            }

            grid_search = GridSearchCV(
                DecisionTreeRegressor(random_state=self.random_state),
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # Bagged Trees
        # ------------------
        if 'bag' in self.models_list or 'all' in self.models_list:
            algorithm = 'bag'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'base_estimator__criterion': ['mse', 'mae'],
                'base_estimator__splitter': ['best', 'random'],
                'n_estimators': [10, 20, 30, 40, 50, 60]
            }

            dt_model = DecisionTreeRegressor(random_state=self.random_state)
            bagging_model = BaggingRegressor(base_estimator=dt_model)

            grid_search = GridSearchCV(
                bagging_model,
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # Adaboost Trees
        # ------------------
        if 'boost' in self.models_list or 'all' in self.models_list:
            algorithm = 'boost'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()

            param_grid = {
                'base_estimator__criterion': ['mse', 'mae'],
                'base_estimator__splitter': ['best', 'random'],
                'n_estimators': [10, 20, 30, 40, 50, 60],
                'loss': ['linear', 'square']
            }

            dt_model = DecisionTreeRegressor(random_state=self.random_state,
                                            )
            adaboost_model = AdaBoostRegressor(base_estimator=dt_model)

            grid_search = GridSearchCV(
                adaboost_model,
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')
        
        # -----------------------
        # Random Forest Regressor
        # -----------------------
        if 'rf' in self.models_list or 'all' in self.models_list:
            algorithm = 'rf'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))

            t0 = time.time()
            param_grid = {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'criterion': ['mse', 'mae']
            }

            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)

            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # SVMs
        # ------------------
        if 'svm' in self.models_list or 'all' in self.models_list:
            algorithm = 'svm'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'C': np.array( [1, 10, 100, 1000.]),
                "epsilon": np.array([ .001, .005, .01, .05, .1, .2 ]
                                    )/(np.max(y_train-np.mean(y_train)) - 
                                    np.min(y_train-np.mean(y_train))),
                'gamma': x_train.shape[1] /2*np.logspace( -6 ,6, num=10)
            }

            grid_search = GridSearchCV(
                SVR(),
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # KRRs
        # ------------------
        if 'krr' in self.models_list or 'all' in self.models_list:
            algorithm = 'krr'
            t0 = time.time()
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            param_grid = {
                'alpha': np.array([ .0001, .001, .01, .1, 1. ]) / x_train.shape[0],
                'gamma': x_train.shape[1] / 2*np.logspace( -6 ,6, num=10)
            }

            grid_search = GridSearchCV(
                KernelRidge(),
                param_grid=param_grid,
                n_jobs=self.n_jobs
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # GPs (ARD Kernel)
        # ------------------
        if 'gpr' in self.models_list or 'all' in self.models_list:
            algorithm = 'gpr'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            kernel = ConstantKernel(1., (1e-10, 1000)) * \
                RBF(length_scale=np.repeat(1.0,x_train.shape[1]), length_scale_bounds=(1e-2, 1e2)) \
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e+1))

            t0 = time.time()
            self.Models[algorithm] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)
            self.Models[algorithm].fit(x_train, y_train)
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        return self

    def save_models(self, path=None, savename=None):
        
        # TODO save models as a pickle file
        # TODO have alternative path save
        # TODO check path names provided

        # check to see if models have been fitted
        if not hasattr(self, 'Models'):
            raise ValueError('No Models have been fitted. Cannot save.')
        
        # save the simple r class to a pckl file
        if savename is None:
            savename = 'models'
        
        joblib.dump(self, str(savename) + '.pckl')

        return self

    def load_models(self, path=None):
        pass
        # # TODO save models as a pickle file
        # # TODO have alternative path save
        # # TODO check path names provided

        # # check to see if models have been fitted
        # if hasattr(self, 'Models'):
        #     raise ValueError('Models have already been fitted.')

        # if path is None:
        #     path = 'models'
        
        # self = joblib.load(str(path) + '.pckl')

        # self.Models = self.Models

        # return self

    def predict_models(self, x_test):

        # TODO: check x_test size
        self.predict_times = dict()

        # Check if Models has been fitted
        assert(hasattr(self, 'Models'))

        for algorithm in self.Models:
            t0 = time.time()
            print('Predicting with {}...'.format(self.algorithm_names[algorithm]))
            self.predictions[algorithm] = self.Models[algorithm].predict(x_test)
            self.predict_times[algorithm] = time.time() - t0
            print('Predictions done in: {:4f} secs'.format(self.predict_times[algorithm]))
            print('Done!')

        return self
    
    def get_stats(self, y_test, save=None):

        # TODO: check y_test size
        # TODO: check if model has been fitted

        # initialize dataframe dictionary
        self.best_params = dict()

        columns = [
            'Model',
            'Best Params',
            'Residuals',
            'MSE',
            'MAE',
            'R Value',
            'R2 Value',
            'P Value',
            'Pearson R',
            'Slope',
            'Intercept',
            'Standard Error',
            'Fit Times',
            'Predict Times'
        ]

        # Initialize empty dataframe with columns
        df = pd.DataFrame(columns=columns)

        for model in self.Models:

            # Bias
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(self.predictions[model], y_test)


            # Pearson Coefficients
            pearsonr, _ = \
                stats.pearsonr(self.predictions[model], y_test)
            
            # get best parameters for the model
            try:
                best_params = self.Models[model].get_params()
            except AttributeError:
                print('No parameters found.')
                best_params = {}
            
            # Params for the GPR
            if hasattr(self.Models[model], "kernel_"):
                best_params = self.Models[model].kernel_.get_params()


            # append to data save
            df = df.append({
                "Model": self.algorithm_names[model],
                "Best Params": best_params,
                "Residuals": self.predictions[model] - y_test,
                "MSE": mean_squared_error(self.predictions[model], y_test),
                "MAE": mean_absolute_error(self.predictions[model], y_test),
                "R2 Value": r2_score(self.predictions[model], y_test),
                "Pearson R": pearsonr,
                "Slope": slope,
                "Intercept": intercept,
                "R Value": r_value,
                "P Value": p_value,
                "Standard Error": std_err,
                "Fit Times": self.fit_times[model],
                "Predict Times": self.predict_times[model]
            }, ignore_index=True)


        # save dataframe
        self.results = df
        return self

    def save_stats(self, path=None, name=None):

        if not hasattr(self, 'results'):
            raise ValueError('No results dataframe. Need to get stats first.')
        
        # TODO - implement path and check if valid
        if name is None:
            self.results.to_csv("errors.csv", index=False)
        else:
            self.results.to_csv(str(name + '.csv'), index=False)


        return self

def main():

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import scale

    boston = load_boston()
    x, y = boston.data, boston.target

    x, y = scale(x), scale(y)

    random_state = 123
    train_percent = 0.8

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, train_size=train_percent,
                         random_state=random_state)

    
    # Initialize Simple R
    model_list = ['all']
    n_jobs = 2
    Models = SimpleR(model_list, n_jobs=n_jobs, 
                     random_state=random_state)

    # Fit Models
    t0 = time.time()
    Models.fit_models(x_train, y_train)
    t1 = time.time() - t0
    print('All Models fitted in: {:4f} secs'.format(t1))

    # Save Models Class
    t0 = time.time()
    Models.save_models(savename='models')
    t1 = time.time() - t0
    print('Saved models in {:.4f} secs'.format(t1))

    # Load Models (Not necessary - only to show an example)
    del Models
    t0 = time.time()
    Models = joblib.load('models.pckl')
    t1 = time.time() - t0
    print('Loaded Models in {:.4f} secs'.format(t1))

    # Predict Data
    t0 = time.time()
    Models.predict_models(x_test)
    t1 = time.time() - t0
    print('All Data Predicted in: {:4f} secs'.format(t1))

    # Get statistics
    Models.get_stats(y_test)

    # Save the stats to csv file
    Models.save_stats(name='errors')

    # print stats
    # results = pd.read_csv('errors.csv')     # From CSV File
    print(Models.results)                   # From class itself

    return None

if __name__ == "__main__":

    main()
