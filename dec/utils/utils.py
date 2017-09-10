from __future__ import division
from math import *
import numpy as np
import urllib.request
import os


class PRFModelTrial(object):
    """docstring for PRFModelTrial"""

    def __init__(self, orientation, n_elements, n_samples, sample_duration, bar_width=0.1, ecc_test=1.0):
        super(PRFModelTrial, self).__init__()
        self.orientation = orientation
        self.n_elements = n_elements
        self.n_samples = n_samples
        self.sample_duration = sample_duration
        self.bar_width = bar_width * 2.

        self.rotation_matrix = np.matrix(
            [[cos(self.orientation), -sin(self.orientation)], [sin(self.orientation), cos(self.orientation)]])

        x, y = np.meshgrid(np.linspace(-1, 1, self.n_elements),
                           np.linspace(-1, 1, self.n_elements))
        self.xy = np.matrix([x.ravel(), y.ravel()])
        self.rotated_xy = np.array(self.rotation_matrix * self.xy)
        self.ecc_test = (np.array(self.xy) ** 2).sum(axis=0) <= ecc_test

        if ecc_test == None:
            self.ecc_test = np.ones_like(self.ecc_test, dtype=bool)

    def in_bar(self, time=0):
        """in_bar, a method, not Ralph."""
        # a bar of self.bar_width width
        position = 2.0 * ((time * (1.0 + self.bar_width / 2.0)
                           ) - (0.5 + self.bar_width / 4.0))
        # position = 2.0 * ((time * (1.0 + self.bar_width)) - (0.5 + self.bar_width / 2.0))
        extent = [-self.bar_width / 2.0 + position,
                  self.bar_width / 2.0 + position]
        # rotating the xy matrix itself allows us to test only the x component
        return ((self.rotated_xy[0, :] >= extent[0]) * (self.rotated_xy[0, :] <= extent[1]) * self.ecc_test).reshape((self.n_elements, self.n_elements))
        # return ((self.rotated_xy[0,:] >= extent[0]) * (self.rotated_xy[0,:] <= extent[1])).reshape((self.n_elements, self.n_elements))

    def pass_through(self):
        """pass_through models a single pass-through of the bar, 
        with padding as in the padding list for start and end."""

        self.pass_matrix = np.array(
            [self.in_bar(i) for i in np.linspace(0.0, 1.0, self.n_samples, endpoint=True)])


def create_visual_designmatrix_all(
        bar_dur_in_TR=32,
        iti_duration=2,
        bar_width=0.1,
        n_pixels=100,
        thetas=[-1, 0, -1, 45, 270, -1,  315,
                180, -1,  135,   90, -1,  225, -1],
        nr_timepoints=462):
    ITIs = np.zeros((iti_duration, n_pixels, n_pixels))
    all_bars = []
    for x in thetas:
        all_bars.extend(ITIs)
        if x == -1:
            all_bars.extend(np.zeros((bar_dur_in_TR, n_pixels, n_pixels)))
        else:
            pmt = PRFModelTrial(orientation=-np.radians(x) - np.pi / 2.0, n_elements=n_pixels,
                                n_samples=bar_dur_in_TR + 1, sample_duration=1, bar_width=bar_width)
            pmt.pass_through()
            pmt.pass_matrix = pmt.pass_matrix[:-1]
            all_bars.extend(pmt.pass_matrix)

    # swap axes for popeye:
    visual_dm = np.transpose(np.array(all_bars), [1, 2, 0])
    visual_dm = np.round(visual_dm[:, :, :nr_timepoints]).astype(np.int16)

    return visual_dm

def roi_data_from_hdf(data_types_wildcards, roi_name_wildcard, hdf5_file, folder_alias):
    """takes data_type data from masks stored in hdf5_file

    Takes a list of 4D fMRI nifti-files and masks the
    data with all masks in the list of nifti-files mask_files.
    These files are assumed to represent the same space, i.e.
    that of the functional acquisitions. 
    These are saved in hdf5_file, in the folder folder_alias.

    Parameters
    ----------
    data_types_wildcards : list
        list of data types to be loaded.
        correspond to nifti_names in mask_2_hdf5
    roi_name_wildcard : str
        wildcard for masks. 
        corresponds to mask_name in mask_2_hdf5.
    hdf5_file : str
        absolute path to hdf5 file.
    folder_alias : str
        name of the folder in the hdf5 file from which data
        should be loaded.

    Returns
    -------
    output_data : list
        list of numpy arrays corresponding to data_types and roi_name_wildcards
    """
    import tables
    import itertools
    import fnmatch
    import numpy as np

    h5file = tables.open_file(hdf5_file, mode="r")

    try:
        folder_alias_run_group = h5file.get_node(
            where='/', name=folder_alias, classname='Group')
    except NoSuchNodeError:
        # import actual data
        print('No group ' + folder_alias + ' in this file')
        # return None

    all_roi_names = h5file.list_nodes(
        where='/' + folder_alias, classname='Group')
    roi_names = [
        rn._v_name for rn in all_roi_names if roi_name_wildcard in rn._v_name]
    if len(roi_names) == 0:
        print('No rois corresponding to ' +
              roi_name_wildcard + ' in group ' + folder_alias)
        # return None

    data_arrays = []
    for roi_name in roi_names:
        try:
            roi_node = h5file.get_node(
                where='/' + folder_alias, name=roi_name, classname='Group')
        except tables.NoSuchNodeError:
            print('No data corresponding to ' +
                  roi_name + ' in group ' + folder_alias)
            pass
        all_data_array_names = h5file.list_nodes(
            where='/' + folder_alias + '/' + roi_name)
        data_array_names = [adan._v_name for adan in all_data_array_names]
        selected_data_array_names = list(itertools.chain(
            *[fnmatch.filter(data_array_names, dtwc) for dtwc in data_types_wildcards]))

        # if sort_data_types:
        selected_data_array_names = sorted(selected_data_array_names)
        if len(data_array_names) == 0:
            print('No data corresponding to ' + str(selected_data_array_names) +
                  ' in group /' + folder_alias + '/' + roi_name)
            pass
        else:
            print('Taking data corresponding to ' + str(selected_data_array_names) +
                  ' from group /' + folder_alias + '/' + roi_name)
            data_arrays.append([])
            for dan in selected_data_array_names:
                data_arrays[-1].append(
                    eval('roi_node.__getattr__("' + dan + '").read()'))

            # stack across timepoints or other values per voxel
            data_arrays[-1] = np.hstack(data_arrays[-1])
    # stack across regions to create a single array of voxels by values (i.e. timepoints)
    all_roi_data_np = np.vstack(data_arrays)

    h5file.close()

    return all_roi_data_np


def get_figshare_data(localpath = 'data/V1.h5', remotepath='https://ndownloader.figshare.com/files/9183091'):
    if os.path.isfile(localpath):
        print('data file found, returning local file %s'%localpath)
        pass 
    else:
        print('downloading data from figshare: %s to: %s'%(remotepath, localpath))
        try:
            os.makedirs(os.path.split(localpath)[0])
        except OSError:
            pass
        urllib.request.urlretrieve(remotepath, localpath)
    return localpath

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

    