import numpy as np 
import time
import warnings
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  LassoCV)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_squared_error,
                             r2_score)

warnings.simplefilter('ignore')

# TODO - Polynomial Regression
# TODO - LASSO
# TODO - Elastic Net
# TODO - KNN
# TODO - Radius NN
# TODO - Decision Trees
# TODO - Bagging Trees
# TODO - AdaBoost Regression
# TODO - Boosted Random Forest
# TODO - Relevance Vector Machanes
# TODO - GPR Heteroscedastic Kernel

class SimpleR(object):
    def __init__(self, models_list=None, n_jobs=1, random_state=None):
        self.models_list = models_list
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.predictions = dict()
        self.algorithm_names = {
            'ols': 'Linear Regresssion',
            'rls': 'Ridge Regression',
            'rf': 'Random Forest Regression',
            'svm': 'Support Vector Regression',
            'krr': 'Kernel Ridge Regression',
            'gpr': 'Gaussian Process Regression'
        }
    
    def fit_models(self, x_train, y_train):

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
        if 'ols' in self.models_list:
            algorithm = 'ols'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            self.Models[algorithm] = LinearRegression()
            self.Models[algorithm].fit(x_train, y_train)
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # Ridge Regression
        # ------------------
        if 'rls' in self.models_list:
            algorithm = 'rls'
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            t0 = time.time()
            param_grid = {
                'alpha': np.linspace(0.01, 10, 20)
            }

            grid_search = GridSearchCV(
                Ridge(),
                param_grid=param_grid
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # SVMs
        # ------------------
        if 'svm' in self.models_list:
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
                param_grid=param_grid
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # KRRs
        # ------------------
        if 'krr' in self.models_list:
            algorithm = 'krr'
            t0 = time.time()
            print('Fitting {}...'.format(self.algorithm_names[algorithm]))
            param_grid = {
                'alpha': np.array([ .0001, .001, .01, .1, 1. ]) / x_train.shape[0],
                'gamma': x_train.shape[1] /2*np.logspace( -6 ,6, num=10)
            }

            grid_search = GridSearchCV(
                KernelRidge(),
                param_grid=param_grid
            )

            grid_search.fit(x_train, y_train)
            
            self.Models[algorithm] = grid_search.best_estimator_
            self.fit_times[algorithm] = time.time() - t0
            print('Fitting Done in: {:4f} secs'.format(self.fit_times[algorithm]))
            print('Done!')

        # ------------------
        # GPs (ARD Kernel)
        # ------------------
        if 'gpr' in self.models_list:
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

        # ------------------
        # LASSO
        # ------------------

        # -----------------------
        # Random Forest Regressor
        # -----------------------
        if 'rf' in self.models_list:
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

        return self

    def save_models(self, path=None, savename=None):
        
        # TODO save models as a pickle file
        # TODO have alternative path save
        # TODO check path names provided

        pass

    def load_models(self, path=None):

        pass

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
    
    def get_stats(self, y_test):

        # TODO: check y_test size
        # TODO: check if model has been fitted
        self.mse = dict()
        self.mae = dict()
        self.r2 = dict()

        for model in self.Models:
            self.mse[model] = mean_squared_error(
                self.predictions[model], y_test
            )
            self.mae[model] = mean_absolute_error(
                self.predictions[model], y_test
            )
            self.r2[model] = r2_score(
                self.predictions[model], y_test
            )

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
    model_list = ['ols', 'rls', 'rf', 'krr', 'gpr', 'svm']
    n_jobs = 2
    Models = SimpleR(model_list, n_jobs=n_jobs, 
                     random_state=random_state)

    # Fit Models
    Models.fit_models(x_train, y_train)

    # Predict Data
    Models.predict_models(x_test)

    # Get statistics
    Models.get_stats(y_test)

    # Print statistic
    print('Mean Squared Errors:')
    for key in Models.mse:
        print(key, Models.mse[key])
    print('Mean Absolute Error')
    for key in Models.mae:
        print(key, Models.mae[key])
    print('R2 Value')
    for key in Models.r2:
        print(key, Models.r2[key])

    return None

if __name__ == "__main__":
    main()
