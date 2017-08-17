############################################################################################################################################
#   imports
############################################################################################################################################

import numpy as np
from popeye.spinach import generate_og_receptive_fields

# import taken from own nPRF package. 
# this duplicates that code, which is unhealthy but should be fine for now
# in order to keep this repo self-contained.
from .utils.utils import roi_data_from_hdf, create_visual_designmatrix_all

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

hdf5_file = '/Users/knapen/disks/ae_S/2016/visual/prf/all_B0/sub-012/masks/V1.h5'

############################################################################################################################################
#   getting the data
############################################################################################################################################

# timecourses are single-run psc data. 
timecourse_data = roi_data_from_hdf(['*psc'],'rh.V1', hdf5_file,'psc').astype(np.float64)
prf_data = roi_data_from_hdf(['*all'],'rh.V1', hdf5_file,'prf').astype(np.float64)

# get design matrix
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
    prf_data[:, 0], prf_data[:, 1], prf_data[:, 2], np.ones((prf_data.shape[0])), deg_x, deg_y)


