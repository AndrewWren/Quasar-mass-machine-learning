# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:54:11 2020

@author: andre
"""

import numpy as np
import os
import pandas as pd



def display_n_epochs(quasars):
    """ A utility function not used in getting the Chen data, but may assist
    understanding of the set of quasars and their spectra, for example
    to define the stratify categories in
    get_Chen_data.Xy.objs_train_test_val()"""
    
    values, edges = np.histogram(quasars['N_EPOCHS'],
                                 bins=list(np.arange(0.5,11))
                                 + [20, 60, 80, np.inf])
    bin_strings = list(range(1,11)) + ['10-20', '20-60', '60-80', '80+']
    return np.transpose((bin_strings, values))


def y_prediction(model, X):
    y_pred = np.squeeze(model.predict(np.expand_dims(X.to_numpy(), -1)))
    return y_pred


def create_y_obj(y):
    y_series = pd.Series(y, index=pd.MultiIndex.from_tuples(y.index))
    y_obj = y_series.groupby(level=0).mean()
    return y_obj

    
def y_obj_mean_prediction(model, X):
    y_pred = y_prediction(model, X)
    y_pred_series = pd.Series(y_pred,
                              index=pd.MultiIndex.from_tuples(X.index))
    y_pred_obj = y_pred_series.groupby(level=0).mean()
    return y_pred_obj


def bin_objs_mse_r2(quasars, y_obj, y_pred_obj, n_epoch_bin):
    quasars_binned_all = quasars.index[((quasars['N_EPOCHS'] >= n_epoch_bin[0])
                      & (quasars['N_EPOCHS'] < n_epoch_bin[1]))]
    #print(quasars_binned_all.sum())
    quasars_binned = quasars_binned_all.intersection(y_obj.index)
    #print(quasars_binned_all.shape, quasars_binned.shape)
    mse = (((y_obj[quasars_binned] - y_pred_obj[quasars_binned])**2)
           .mean())
    r2 = 1 - mse/y_obj.var()
    return quasars_binned.shape[0], mse, r2 


def n_epoch_bin_table(quasars, y_obj, y_pred_obj, n_epoch_bin_edges,
                      title=False):
    n_epoch_bins = [[n_epoch_bin_edges[i], n_epoch_bin_edges[i+1]]
                    for i, _ in enumerate(n_epoch_bin_edges[:-1])]
    if title is not False:
        print(title)
    print('\n   Bin\t    N\t  MSE\t   R2\n')
    for n_epoch_bin in n_epoch_bins:
        bin_name = str(n_epoch_bin[0])
        if n_epoch_bin[1] != n_epoch_bin[0] + 1:
            bin_name += ('-' + str(n_epoch_bin[1]))
        res = bin_objs_mse_r2(quasars, y_obj, y_pred_obj, n_epoch_bin)
        print(f'{bin_name:>6}\t{res[0]:>5}\t{res[1]:.03f}\t{res[2]:.03f}')
    print('=' * 29)
    res = bin_objs_mse_r2(quasars, y_obj, y_pred_obj, [1, np.inf])
    print(f'   All\t{res[0]:>5}\t{res[1]:.03f}\t{res[2]:.03f}\n')
    
    
def model_mean_predict_bins(quasars, model, X, y, n_epoch_bin_edges,
                            title=False):
    y_obj = create_y_obj(y)
    y_pred_obj = y_obj_mean_prediction(model, X)
    n_epoch_bin_table(quasars, y_obj, y_pred_obj, n_epoch_bin_edges,
                      title=False)
