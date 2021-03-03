import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Reducer
import time
import gc
from joblib import Parallel, delayed



class preprocessor:
    """This class preprocesses Data to make it work better for Machine Learning.
    df: Dataframe to preprocess
    y_var: Variable to predict for the regression
    cols_to_drop: Attributes which the class should drop.
    numbers_to_encode: Numbers which should be encoded. This class automatically encodes all object and categorical-types.
    method_to_encode: Method how to encode. ('onehot_encode' or 'label_encode')
    test_frac: Fraction to create Test-Data with.
    remove_skew_by_boxcox: If data should be transformed via Boxcox-Transformation to make it normally-distributed.
    features_not_to_box_cox: If some Attributes should not be transformed with Boxcox.
    standardise_number: If numbers should be standardised (If robust_scaler == False, it will change numeric values into Z-scores.)
    robust_scaler: If numbers should be standardised with the robust-scaler.

    Returns: Nothing, but Test and Train-Data can be called with preprocessor.X_test, preprocessor.y_test, preprocessor.X_train and preprocessor.y_train.
    """
    def __init__(self, df, y_var, cols_to_drop = [], numbers_to_encode = [], method_to_encode = 'onehot_encode', numbers_to_onehot_encode = [], test_frac = 0.2, remove_skew_by_boxcox = False, 
        features_not_to_box_cox = [], standardise_number = True, square_numbers = False, robust_scaler = False, verbose = 0):
        ### Tests
        assert not numbers_to_onehot_encode, 'Please change "numbers_to_onehot_encode" to "numbers_to_encode", thx.'
        assert method_to_encode in ['onehot_encode', 'label_encode'], 'For method_to_encode, only "onehot_encode" or "label_encode" is accepted.'
        assert y_var, 'Please define a y_var!'

        ### Fixed Parameter for Class:
        self.str_na = 'No Value'
        self.numeric_na = -9999

        ### Drop some rows incase they are always empty (they do not exist in the Test-Set for Kaggle)
        try:
            df = df[df['TravelTimeMiv'].notna()].copy()
        except:
            pass
        try:
            df = df[df['StateShort'].notna()].copy()
        except:
            pass
        try:
            df = df[df['gde_area_settlement_percentage'].notna()].copy()
        except:
            pass
        df.reset_index(inplace = True, drop=True)
        
        ###Save some parameters to class:
        self.y = df[y_var].copy()
        self.test_frac = test_frac
        self.numbers_to_encode = numbers_to_encode
        self.cols_to_drop = cols_to_drop
        self.standardise_number = standardise_number
        self.square_numbers = square_numbers
        self.features_not_to_box_cox = features_not_to_box_cox
        self.remove_skew_by_boxcox = remove_skew_by_boxcox
        self.robust_scaler = robust_scaler
        self.verbose = verbose


        ### Drop Columns based on y_var and cols_to_drop
        if self.verbose > 0:
            print('Columns dropped to create X: ', cols_to_drop)
        self.X = df.drop(columns = [y_var])
        self.X = self.__drop_columns(self.X)

        ### Cast Dtypes:
        self.X = self.__cast_dtypes(self.X)

        # Fit method for encoding:
        self.label_method = method_to_encode
        if method_to_encode == 'onehot_encode':
            self.d_labler = defaultdict(lambda: sklearn.preprocessing.OneHotEncoder(dtype=bool))
        else:
            self.d_labler = defaultdict(sklearn.preprocessing.LabelEncoder)

    def __drop_columns(self, df_):
        """
        This function drops some columns based on self.cols_to_drop
        df_: Dataframe to drop columns from
        """
        df = df_.copy()
        X = df.drop(columns = self.cols_to_drop)
        return X

    def __cast_dtypes(self, X_):
        """
        This function casts some numeric columns based on self.numbers_to_encode and adds a _ to not allow pandas to cast dtypes back to numeric.
        df_: Dataframe to cast columns in
        """
        X = X_.copy()
        if len(self.numbers_to_encode)>0:
            X = X.astype({k:str for k in self.numbers_to_encode}).copy()
            if self.verbose > 0:
                print("Succesfully casted Dtypes!\n",X.dtypes)
            # Dirty workaround because some pandas operations change the dtype back to numbers if possible :(
            for col in self.numbers_to_encode:
                X.loc[:,col] = X.loc[:,col] + "_"
        return X

    def __fillna(self, _df):
        df = _df.copy()
        for col in df:
            #get dtype for column
            dt = df[col].dtype
            #check if it is a number
            if is_object_dtype(dt):
                if df[col].isna().values.any():
                    df[col + '_has_no_value'] = np.where(df[col].isna(), True, False)
                    df[col] = df[col].fillna(self.str_na).copy()
            else:
                if df[col].isna().values.any():
                    df[col + '_has_no_value'] = np.where(df[col].isna(), True, False)
                    df[col] = df[col].fillna(self.mean[col]).copy()
        return df


    def __remove_skew_boxcox(self, _df):
        # Code taken from https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition and adapted to our needs.
        df = _df.copy()
        for i in self.skew_index:
            try:
                df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))
            except (TypeError, ValueError) as E:
                if self.verbose > 0:
                    print(str(E) + '.\nThus, skipping the boxcox Transformation for {}.\n----'.format(i))
        return df


    def __fit_df(self):
        """
        This function fits (one hot encoding) the categorical columns if they are of type object and
        function fits (standardscaler) the numerical columns if they are of numbers (based on select_dtypes).
        Takes:
        - _df: pandas.DataFrame
        - labler: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') or sklearn.preprocessing.LabelEncoder()
        returns:
        - encoded df as numpy.array
        """
        # Fit for encoder
        df_obj = self.X.select_dtypes(include = [object]).copy()
        df_obj = df_obj.fillna(self.str_na).copy()
        df_obj.apply(lambda x: self.d_labler[x.name].fit(x.values.reshape(-1,1)))
        
        self.obj_features = df_obj.columns
        
        # Fit for standard scaler:
        df_num = self.X_train.select_dtypes(include = ['float32', 'float64'])
        self.std = df_num.std(axis=0).fillna(1)
        self.mean = df_num.mean(axis=0).fillna(0)
        self.median = df_num.median(axis=0).fillna(0)
        self.first_quantile = df_num.quantile(0.25)
        self.third_quantile = df_num.quantile(0.75)

        self.num_features = df_num.columns
        self.features_between_0_1 = []
        for col in self.num_features:
            if self.X_train[col].between(0, 1).all():
                self.features_between_0_1.append(col)

        # Fit for Boxcox:
        if self.remove_skew_by_boxcox:
            if self.features_not_to_box_cox:
                skew_features = self.X.select_dtypes(['float32', 'float64']).drop(columns=self.features_not_to_box_cox).apply(lambda x: skew(x)).sort_values(ascending=False)
            else:
                skew_features = self.X.select_dtypes(['float32', 'float64']).apply(lambda x: skew(x)).sort_values(ascending=False)
            # Remove created boolean values:
            skew_features = skew_features.filter(regex = '.+[^_has_no_value]$')

            high_skew = skew_features[skew_features > 0.5]
            self.skew_index = high_skew.index

            # Remove values which cannot be used for Boxcox Transformation. Should be investigated why.
            for i in self.skew_index:
                try:
                    boxcox1p(self.X[i], boxcox_normmax(self.X[i] + 1))
                except (TypeError, ValueError) as E:
                    if self.verbose > 0:
                        print(str(E) + '.\nThus, skipping the boxcox Transformation for {}.\n----'.format(i))
                    self.skew_index.remove(i)
            if self.verbose > 0:
                print("There are {} numerical features with Skew > 0.5 :\n{}".format(high_skew.shape[0], high_skew))

    def __encode_transform_df(self, _df):
        """
        This function transforms the new df with the fitted encoder (one hot encoding).

        Takes:
        - _df: pandas.DataFrame
        - labler: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') or sklearn.preprocessing.LabelEncoder()

        returns:
        - transformed df as numpy.array
        """

        df = _df.copy()
        for col in self.obj_features:
            if self.label_method == 'onehot_encode':
                df_onehot_encoded = self.d_labler[col].transform(df[col].values.reshape(-1,1)).toarray()
                df_onehot_encoded_df = pd.DataFrame(df_onehot_encoded, columns = self.d_labler[col].get_feature_names([col]))
                df_onehot_encoded_df.index = df.index
                df.drop(columns = col, inplace = True)
                df = df.join(df_onehot_encoded_df)
            else:
                df[col] = self.d_labler[col].transform(df[col])
        return df

    def __add_feature_is_zero(self, _df):
        df = _df.copy()
        for col in self.features_between_0_1:
            df[col + '_is_zero'] = np.where(df[col]==0, True, False)
        return df


    def __standardise_df(self, _df):
        """
        This function transforms the new df with the fitted encoder (one hot encoding).

        Takes:
        - _df: pandas.DataFrame
        - labler: sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore') or sklearn.preprocessing.LabelEncoder()

        returns:
        - transformed df as numpy.array
        """
        df = _df.copy()
        for col in self.num_features:
            if not df[col].between(0, 1).all():
                if self.robust_scaler:
                    df[col] = ((df[col] - self.median[col]) / (self.third_quantile[col] - self.first_quantile[col])).copy()
                else:
                    df[col] = ((df[col] - self.mean[col]) / (self.std[col])).copy()
            else:
                if self.verbose > 1:
                    print("{} is between 0 and 1, not standardising.".format(col))
        return df

    def __square_df(self, _df):
        """
        This function squares numbers.

        Takes:
        - _df: pandas.DataFrame

        returns:
        - transformed df as numpy.array
        """
        df = _df.copy()
        for col in self.num_features:
            df[col + '_sqrt'] = np.square(df[col])
        return df


    def split_X_y(self, test_frac = 0.2):
        self.X_test = self.X.sample(frac = test_frac, random_state = 42)
        self.X_train = self.X.drop(index = self.X_test.index, axis = 0)
        self.y_train = self.y.iloc[self.X_train.index]
        self.y_test = self.y.iloc[self.X_test.index]


    def encode_sample(self, _sample, test_data = False):
        sample = _sample.copy()
        if test_data:
            sample = self.__drop_columns(sample).copy()
            sample = self.__cast_dtypes(sample).copy()

        if self.standardise_number:
            sample = self.__standardise_df(sample).copy()

        sample = self.__fillna(sample).copy()
        sample = self.__add_feature_is_zero(sample).copy()

        if self.remove_skew_by_boxcox:
            sample = self.__remove_skew_boxcox(sample).copy()

        sample = self.__encode_transform_df(sample).copy()

        if self.square_numbers:
            sample = self.__square_df(sample).copy()

        return sample

    def preprocess(self):
        self.split_X_y(test_frac = self.test_frac)
        self.__fit_df()

        if self.standardise_number:
            self.X_train = self.__standardise_df(self.X_train).copy()

        self.X_train = self.__fillna(self.X_train).copy()
        self.X_train = self.__add_feature_is_zero(self.X_train).copy()
        
        if self.remove_skew_by_boxcox:
            self.X_train = self.__remove_skew_boxcox(self.X_train).copy()

        if self.square_numbers:
            self.X_train = self.__square_df(self.X_train).copy()

        self.X_train = self.__encode_transform_df(self.X_train).copy()

        self.X_test = self.encode_sample(self.X_test)


"""reducing.py
Author: Kirgsn, 2018
https://www.kaggle.com/wkirgsn/fail-safe-parallel-memory-reduction
Use like this:
>>> import reducing
>>> df = reducing.Reducer().reduce(df)
"""

def measure_time_mem(func):
    def wrapped_reduce(self, df, *args, **kwargs):
        # pre
        mem_usage_orig = df.memory_usage().sum() / self.memory_scale_factor
        start_time = time.time()
        # exec
        ret = func(self, df, *args, **kwargs)
        # post
        mem_usage_new = ret.memory_usage().sum() / self.memory_scale_factor
        end_time = time.time()
        print(f'reduced df from {mem_usage_orig:.4f} MB '
              f'to {mem_usage_new:.4f} MB '
              f'in {(end_time - start_time):.2f} seconds')
        gc.collect()
        return ret
    return wrapped_reduce


class Reducer:
    """
    https://www.kaggle.com/wkirgsn/fail-safe-parallel-memory-reduction
    Class that takes a dict of increasingly big numpy datatypes to transform
    the data of a pandas dataframe into, in order to save memory usage.
    usage:
    df = Reducer().reduce(df)
    """
    memory_scale_factor = 1024**2  # memory in MB

    def __init__(self, conv_table=None, use_categoricals=True, n_jobs=-1):
        """
        :param conv_table: dict with np.dtypes-strings as keys
        :param use_categoricals: Whether the new pandas dtype "Categoricals"
                shall be used
        :param n_jobs: Parallelization rate
        """

        self.conversion_table = \
            conv_table or {'int': [np.int8, np.int16, np.int32, np.int64],
                           'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                           'float': [np.float32, ]}
        self.use_categoricals = use_categoricals
        self.n_jobs = n_jobs

    def _type_candidates(self, k):
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    @measure_time_mem
    def reduce(self, df, verbose=False):
        """Takes a dataframe and returns it with all data transformed to the
        smallest necessary types.

        :param df: pandas dataframe
        :param verbose: If True, outputs more information
        :return: pandas dataframe with reduced data types
        """
        ret_list = Parallel(n_jobs=self.n_jobs)(delayed(self._reduce)
                                                (df[c], c, verbose) for c in
                                                df.columns)

        del df
        gc.collect()
        return pd.concat(ret_list, axis=1)

    def _reduce(self, s, colname, verbose):
        # skip NaNs
        if s.isnull().any():
            if verbose: print(f'{colname} has NaNs - Skip..')
            return s
        # detect kind of type
        coltype = s.dtype
        if np.issubdtype(coltype, np.integer):
            conv_key = 'int' if s.min() < 0 else 'uint'
        elif np.issubdtype(coltype, np.floating):
            conv_key = 'float'
        else:
            if isinstance(coltype, object) and self.use_categoricals:
                # check for all-strings series
                if s.apply(lambda x: isinstance(x, str)).all():
                    if verbose: print(f'convert {colname} to categorical')
                    return s.astype('category')
            if verbose: print(f'{colname} is {coltype} - Skip..')
            return s
        # find right candidate
        for cand, cand_info in self._type_candidates(conv_key):
            if s.max() <= cand_info.max and s.min() >= cand_info.min:
                if verbose: print(f'convert {colname} to {cand}')
                return s.astype(cand)

        # reaching this code is bad. Probably there are inf, or other high numbs
        print(f"WARNING: {colname} doesn't fit the grid with \nmax: {s.max()} "
              f"and \nmin: {s.min()}")
        print('Dropping it..')
