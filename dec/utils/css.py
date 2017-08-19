#!/usr/bin/python

""" Classes and functions for estimating Compressive Spatial Summation pRF model with filtering added """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve, savgol_filter
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask
from popeye.css import CompressiveSpatialSummationModel


class CompressiveSpatialSummationModelFiltered(CompressiveSpatialSummationModel):
    r"""
    A Compressive Spatial Summation population receptive field model class,
    adapted by TK (3/9/2017) to filter the regressors using the same filtering regime
    as had been used by the fMRI preprocessing pipeline.

    """

    def __init__(self, stimulus, hrf_model, nuissance=None, sg_filter_window_length=127, sg_filter_order=3):
        r"""
        A Compressive Spatial Summation population receptive field model [1]_.

        Paramaters
        ----------

        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.

        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`

        References
        ----------

        .. [1] Kay KN, Winawer J, Mezer A, Wandell BA (2014) Compressive spatial
        summation in human visual cortex. Journal of Neurophysiology 110:481-494.

        """

        CompressiveSpatialSummationModel.__init__(
            self, stimulus, hrf_model, nuissance)

        # the window_length should be integers, will not calculate from TR.
        # the present standard argument implements the window_length value
        # that would be good for a 120 s window and .945 s TR.
        self.sg_filter_window_length = sg_filter_window_length
        self.sg_filter_order = sg_filter_order

    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma, n, beta, baseline):

        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x0, self.stimulus.deg_y0)

        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x0[0, 0:2])**2)

        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr0, rf)

        # compression
        response **= n

        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)

        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]

        # at this point, add filtering with a savitzky-golay filter
        model = model - savgol_filter(model, window_length=self.sg_filter_window_length, polyorder=self.sg_filter_order,
                                      deriv=0, mode='nearest')

        # scale it by beta
        model *= beta

        # offset
        model += baseline

        return model

    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, n, beta, baseline):

        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)

        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)

        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)

        # compression
        response **= n

        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)

        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]

        # at this point, add filtering with a savitzky-golay filter
        model = model - savgol_filter(model, window_length=self.sg_filter_window_length, polyorder=self.sg_filter_order,
                                      deriv=0, mode='nearest')

        # scale it by beta
        model *= beta

        # offset
        model += baseline

        return model
