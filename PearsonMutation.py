import pandas as pd
import numpy as np
import scipy.special as special
import warnings
import scipy
from scipy.stats import PearsonRConstantInputWarning


class PearsonMutation(object):
    """Compute Pearson correlation. Return correlation matrix of symbols in df(columns of df)"""
    def __init__(self, file_name, df, end_index, past_days=126):
        self.file_name = file_name
        self.df = df
        self.end_index = end_index
        self.past_days = past_days

    def historical_mean(self, stock_index):  # one year 253 trading days, 6 month 126 days.
        '''read data from end_index passing past_days, and only select columns showing in stock_index. Compute
        the mean of each column, then save them as a dictionary
        stock_index: showing in the selected window,
        end_index: the last index of the selected window,
        past_days: how many historical days is considered to compute the mean.'''
        df_mean = pd.read_csv(self.file_name)[self.end_index - self.past_days:self.end_index]
        df_mean = df_mean[stock_index]  # get selected stock indexes without x as a dataframe
        mean_list = df_mean.mean(axis=0)  # if it contains 'NaN', mean will ignore the value of 'NaN'.
        mean_dict = {}
        for i in stock_index:
            mean_dict[i] = mean_list[i]
        return mean_dict


    def _chk_asarray(self, a, axis):
        if axis is None:
            a = np.ravel(a)
            outaxis = 0
        else:
            a = np.asarray(a)
            outaxis = axis
        return a, outaxis


    def ss(self, a, axis=0):
        """
        Squares each element of the input array, and returns the sum(s) of that.
        Parameters
        ----------
        a : array_like
            Input array.
        axis : int or None, optional
            The axis along which to calculate. If None, use whole array.
            Default is 0, i.e. along the first axis.
        Returns
        -------
        ss : ndarray
            The sum along the given axis for (a**2).
        See also
        --------
        30.0
        """
        a, axis = self._chk_asarray(a, axis)
        return np.sum(a * a, axis)


    def betai(self, a, b, x):
        x = np.asarray(x)
        x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
        return special.betainc(a, b, x)


    def pearson_mutation(self, x, y, xmean, ymean, end_index):
        n = len(x)
        if n != len(y):
            raise ValueError('x and y must have the same length.')

        if n < 2:
            raise ValueError('x and y must have length at least 2.')

        x = np.asarray(x)
        y = np.asarray(y)
        #if (x == x[0]).all() or (y == y[0]).all():
        if (x == xmean).all() or (y == ymean).all():
            warnings.warn(PearsonRConstantInputWarning())
            return np.nan, np.nan
        if n == 2:
            return dtype(np.sign(x[1] - x[0]) * np.sign(y[1] - y[0])), 1.0

        xm = x - xmean
        ym = y - ymean
        r_num = np.add.reduce(xm * ym)
        r_den = np.sqrt(self.ss(xm) * self.ss(ym))
        r = r_num / r_den

        # Presumably, if abs(r) > 1, then it is only some small artifact of floating
        # point arithmetic.
        r = max(min(r, 1.0), -1.0)
        d = n - 2
        if abs(r) == 1.0:
            prob = 0.0
        else:
            t_squared = r * r * (d / ((1.0 - r) * (1.0 + r)))
            prob = self.betai(0.5 * d, 0.5, d / (d + t_squared))
        return r, prob


    def pearson_correlation(self):
        '''df: without all columns have NaN. df: 10'''
        stock_index = self.df.columns[1:]
        # print(stock_index)
        mean_dict = self.historical_mean(stock_index)
        corr_matrix = np.ones((len(stock_index), len(stock_index)), dtype=float)
        for i in range(0, len(stock_index)):
            for j in range(i + 1, len(stock_index)):
                xmean = mean_dict.get(stock_index[i])
                ymean = mean_dict.get(stock_index[j])
                corr, prob = self.pearson_mutation(self.df[stock_index[i]].values, self.df[stock_index[j]].values,
                                              xmean, ymean, self.end_index)

                corr_matrix[i][j] = corr
                corr_matrix[j][i] = corr
        return corr_matrix