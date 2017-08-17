############################################################################################################################################
#   imports
############################################################################################################################################

import numpy as np
import tables
import ctypes
from popeye.spinach import generate_og_receptive_fields
# from popeye.css import CompressiveSpatialSummationModel
from popeye.visual_stimulus import VisualStimulus
from hrf_estimation.hrf import spmt
from scipy.signal import savgol_filter, fftconvolve
import matplotlib.pyplot as pl

# import taken from own nPRF package. 
# this duplicates that code, which is unhealthy but should be fine for now
# in order to keep this repo self-contained.
from utils.utils import roi_data_from_hdf, create_visual_designmatrix_all, get_figshare_data
from utils.css import CompressiveSpatialSummationModelFiltered


# indices into prf output array:
#	0:	X
#	1:	Y
#	2:	s (size, standard deviation of gauss)
#	3:	n (nonlinearity power)
#	4:	a (amplitude)
#	5:	b (baseline, intercept value)
#	6:	rsq per-run
#	7:	rsq across all
#

############################################################################################################################################
#   parameters of the reconstructions
############################################################################################################################################

# parameters of analysis
extent=[-5, 5]
stim_radius=5.0
n_pix=100
rsq_threshold = 0.3

# settings that have to do with the data and experiment
nr_prf_parameters = 8
TR = 0.945
screen_distance = 225
screen_width = 39
nr_TRs = 462
timepoints = np.arange(nr_TRs) * TR

hdf5_file = get_figshare_data('data/V1.h5')

############################################################################################################################################
#   getting the data
############################################################################################################################################

# timecourses are single-run psc data, either original or leave-one-out. 
timecourse_data_single_run = roi_data_from_hdf(['*psc'],'rh.V1', hdf5_file,'psc').astype(np.float64)
timecourse_data_loo = roi_data_from_hdf(['*loo'],'rh.V1', hdf5_file,'loo').astype(np.float64)
timecourse_data_all_psc = roi_data_from_hdf(['*av'],'rh.V1', hdf5_file,'all_psc').astype(np.float64)
# prfs are per-run, as fit using the loo data
all_prf_data = roi_data_from_hdf(['*all'],'rh.V1', hdf5_file,'all_prf').astype(np.float64)
prf_data = roi_data_from_hdf(['*all'],'rh.V1', hdf5_file,'prf').astype(np.float64).reshape((all_prf_data.shape[0], -1, all_prf_data.shape[-1]))

# get design matrix, could create new one from utils.utils.create_visual_designmatrix_all
h5file = tables.open_file(hdf5_file, mode="r")
dm_n = h5file.get_node(
                where='/', name='dm', classname='Group')
dm = dm_n.dm.read()
h5file.close()

# voxel subselection, using the 'all' rsq values
rsq_mask = all_prf_data[:,-1] > rsq_threshold

############################################################################################################################################
#   setting up prf timecourses - NOTE, this is for the 'all' situation, so should be really done on a run-by-run basis using a run's 
#	loo data and prf parameters. A test set would then be taken from the single_run data as this hasn't been used for that run's fit.
############################################################################################################################################

# set up model with hrf etc.
def my_spmt(delay, tr):
    return spmt(np.arange(0, 33, tr))

# we're going to use these popeye convenience functions 
# because they are fast, and because they were used in the fitting procedure
stimulus = VisualStimulus(
    dm, screen_distance, screen_width, 1.0 / 3.0, TR, ctypes.c_int16)
css_model = CompressiveSpatialSummationModelFiltered(stimulus, my_spmt)
css_model.hrf_delay = 0

# construct predicted signal timecourses in an ugly for loop
# this already convolves with the standard hrf, so we don't have to convolve by hand
prf_predictions = np.zeros((rsq_mask.sum(),nr_TRs))
for i, vox_prf_pars in enumerate(all_prf_data[rsq_mask]):
    prf_predictions[i] = css_model.generate_prediction(
        x=vox_prf_pars[0], y=vox_prf_pars[1], sigma=vox_prf_pars[2], n=vox_prf_pars[3], beta=vox_prf_pars[4], baseline=vox_prf_pars[5])

# and take the residuals of these with the actual data
all_residuals = timecourse_data_all_psc[rsq_mask] - prf_predictions

############################################################################################################################################
#   setting up prf spatial profiles for subsequent covariances, again for 'all' data. 
############################################################################################################################################

deg_x, deg_y = np.meshgrid(np.linspace(extent[0], extent[1], n_pix, endpoint=True), np.linspace(
    extent[0], extent[1], n_pix, endpoint=True))

rfs = generate_og_receptive_fields(
    all_prf_data[rsq_mask, 0], all_prf_data[rsq_mask, 1], all_prf_data[rsq_mask, 2], np.ones((rsq_mask.sum())), deg_x, deg_y)

############################################################################################################################################
#   setting up covariances
############################################################################################################################################

stimulus_covariance = np.cov(rfs.reshape((-1,rfs.shape[-1])).T)
all_residual_covariance = np.cov(all_residuals)
all_residual_covariance_diagonal = np.eye(all_residual_covariance.shape[0]) * all_residual_covariance # in-place multiplication



