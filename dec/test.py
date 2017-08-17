############################################################################################################################################
#   imports
############################################################################################################################################

import numpy as np
import tables
from popeye.spinach import generate_og_receptive_fields

# import taken from own nPRF package. 
# this duplicates that code, which is unhealthy but should be fine for now
# in order to keep this repo self-contained.
from utils.utils import roi_data_from_hdf, create_visual_designmatrix_all, get_figshare_data

# indices into prf output array:
#	0:	X
#	1:	Y
#	2:	s (size, standard deviation of gauss)
#	3:	n (nonlinearity power)
#	4:	a (amplitude)
#	5:	b (baseline, intercept value)
#	6:	rsq
#	7:	
#

############################################################################################################################################
#   parameters of the reconstructions
############################################################################################################################################

extent=[-5, 5]
stim_radius=5.0
n_pix=100
nr_prf_parameters = 8

hdf5_file = get_figshare_data('data/V1.h5')

############################################################################################################################################
#   getting the data
############################################################################################################################################

# timecourses are single-run psc data, either original or leave-one-out. 
timecourse_data_single_run = roi_data_from_hdf(['*psc'],'rh.V1', hdf5_file,'psc').astype(np.float64)
timecourse_data_loo = roi_data_from_hdf(['*loo'],'rh.V1', hdf5_file,'loo').astype(np.float64)
# prfs are per-run, as fit using the loo data
all_prf_data = roi_data_from_hdf(['*all'],'rh.V1', hdf5_file,'all_prf').astype(np.float64)
prf_data = roi_data_from_hdf(['*all'],'rh.V1', hdf5_file,'prf').astype(np.float64).reshape((all_prf_data.shape[0], -1, all_prf_data.shape[-1]))

# get design matrix, could create new one from utils.utils.create_visual_designmatrix_all
h5file = tables.open_file(hdf5_file, mode="r")
dm_n = h5file.get_node(
                where='/', name='dm', classname='Group')
dm = dm_n.dm.read()
h5file.close()

############################################################################################################################################
#   setting up prfs
############################################################################################################################################

deg_x, deg_y = np.meshgrid(np.linspace(extent[0], extent[1], n_pix, endpoint=True), np.linspace(
    extent[0], extent[1], n_pix, endpoint=True))

rfs = generate_og_receptive_fields(
    all_prf_data[:, 0], all_prf_data[:, 1], all_prf_data[:, 2], np.ones((all_prf_data.shape[0])), deg_x, deg_y)

# # raise to the n power
# rfs **= all_prf_data[:, 3]
# # multiply by amplitude parameter
# rfs *= all_prf_data[:, 4]
# # add baseline offset : intercept
# rfs += all_prf_data[:, 5]
