import numpy as np
import scipy as sp
import tables
import ctypes
import matplotlib.pyplot as pl

from hrf_estimation.hrf import spmt
from scipy.signal import savgol_filter, fftconvolve, deconvolve
from scipy.ndimage.interpolation import rotate

from popeye.spinach import generate_og_receptive_fields
from popeye.visual_stimulus import VisualStimulus

from .utils import roi_data_from_hdf, create_visual_designmatrix_all, get_figshare_data, create_circular_mask
from .css import CompressiveSpatialSummationModelFiltered
from .fit import *

from tqdm import tqdm



def setup_data_from_h5(data_file, 
                        n_pix, 
                        extent=[-5,5], 
                        screen_distance=225, 
                        screen_width=69.0, 
                        rsq_threshold=0.5,
                        TR=0.945,
                        cv_fold=1,
                        n_folds=6,
                        use_median=True):
        
    hdf5_file = get_figshare_data(data_file)

    ############################################################################################################################################
    #   getting the data
    ############################################################################################################################################

    # timecourses are single-run psc data, either original or leave-one-out. 
    timecourse_data_single_run = roi_data_from_hdf(['*psc'],'V1', hdf5_file,'psc').astype(np.float64)
    timecourse_data_loo = roi_data_from_hdf(['*loo'],'V1', hdf5_file,'loo').astype(np.float64)
    timecourse_data_all_psc = roi_data_from_hdf(['*av'],'V1', hdf5_file,'all_psc').astype(np.float64)
    # prfs are per-run, as fit using the loo data
    all_prf_data = roi_data_from_hdf(['*all'],'V1', hdf5_file,'all_prf').astype(np.float64)
    prf_data = roi_data_from_hdf(['*all'],'V1', hdf5_file,'prf').astype(np.float64).reshape((all_prf_data.shape[0], -1, all_prf_data.shape[-1]))

    dm=create_visual_designmatrix_all(n_pixels=n_pix)
    if use_median:
        dm_crossv = dm
    else:
        dm_crossv = np.tile(dm,(1,1,n_folds-1))

    # apply pixel mask to design matrix, as we also do this to the prf profiles.    
    mask = dm.sum(axis = -1, dtype = bool)
    dm = dm[mask,:]
    
    # voxel mask for crossvalidation
    # only count those voxels here that have positive rsq
    rsq_crossv = np.mean(prf_data[:,:,-1], axis=1) * np.sign(np.mean(prf_data[:,:,4], axis=1))
    rsq_mask_crossv = rsq_crossv > rsq_threshold
    
    # determine amount of trs
    nr_TRs = int(timecourse_data_single_run.shape[-1] / n_folds)

    # now, mask the data and select for this fold
    test_data = timecourse_data_single_run[rsq_mask_crossv,nr_TRs*cv_fold:nr_TRs*(cv_fold+1)]
    if use_median:
        train_data = timecourse_data_loo[rsq_mask_crossv,nr_TRs*cv_fold:nr_TRs*(cv_fold+1)]
    else:
        train_data = np.delete(timecourse_data_single_run[rsq_mask_crossv,:], np.s_[nr_TRs*cv_fold:nr_TRs*(cv_fold+1)], axis=1)

    # set up the prf_data variable needed for decoding
    prf_cv_fold_data = prf_data[rsq_mask_crossv,cv_fold]

    ############################################################################################################################################
    #   setting up prf timecourses - NOTE, this is for the 'all' situation, so should be really done on a run-by-run basis using a run's 
    #   loo data and prf parameters. A test set would then be taken from the single_run data as this hasn't been used for that run's fit.
    ############################################################################################################################################

    # set up model with hrf etc.
    def my_spmt(delay, tr):
        return spmt(np.arange(0, 33, tr))

    # we're going to use these popeye convenience functions 
    # because they are fast, and because they were used in the fitting procedure
    stimulus = VisualStimulus(dm_crossv, 
                            screen_distance, 
                            screen_width, 
                            1.0, 
                            TR, 
                            ctypes.c_int16)
    css_model = CompressiveSpatialSummationModelFiltered(stimulus, my_spmt)
    css_model.hrf_delay = 0

    ############################################################################################################################################
    #   setting up prf spatial profiles for subsequent covariances, now some per-run stuff was done
    ############################################################################################################################################
    
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
    deg_x, deg_y = np.meshgrid(np.linspace(extent[0], extent[1], n_pix, endpoint=True), np.linspace(
        extent[0], extent[1], n_pix, endpoint=True))

    rfs = generate_og_receptive_fields( prf_data[rsq_mask_crossv, cv_fold, 0], 
                                        prf_data[rsq_mask_crossv, cv_fold, 1], 
                                        prf_data[rsq_mask_crossv, cv_fold, 2], 
                                        np.ones((rsq_mask_crossv.sum())), 
                                        deg_x, 
                                        deg_y)
    
    cov_rfs = np.copy(rfs)
    # scale the rfs according to prf_cv_fold_data
    cov_rfs **= prf_cv_fold_data[:, 3]
    cov_rfs *= prf_cv_fold_data[:, 4]
    cov_rfs += prf_cv_fold_data[:, 5]

    #this step is used in the css model
    rfs /= ((2 * np.pi * prf_data[rsq_mask_crossv, cv_fold, 2]**2) * 1 /np.diff(css_model.stimulus.deg_x[0, 0:2])**2)

    # convert to 1D array and mask with circular mask
    rfs = rfs.reshape((np.prod(mask.shape),-1))[mask.ravel(),:]
    cov_rfs = cov_rfs.reshape((np.prod(mask.shape),-1))[mask.ravel(),:]

    ############################################################################################################################################
    #   setting up prf spatial profiles for the decoding step, creating linear_predictor
    ############################################################################################################################################

    # and then we try to use this:
    W=rfs.T
    linear_predictor=np.zeros((W.shape[0], W.shape[1]+1))
    linear_predictor[:,1:]=np.copy(W)
    #do some rescalings. This affects decoding quite a lot!
    linear_predictor **= np.tile(prf_cv_fold_data[:, 3], (linear_predictor.shape[1],1)).T
    #at this point (after power raising but before multiplication/subtraction) the css model convolves with hrf.
    linear_predictor *= np.tile(prf_cv_fold_data[:, 4], (linear_predictor.shape[1],1)).T
    linear_predictor += np.tile(prf_cv_fold_data[:, 5], (linear_predictor.shape[1],1)).T

    ############################################################################################################################################
    #   create covariances and stuff for omega fitting
    ############################################################################################################################################

    prediction= np.dot(rfs.T,dm_crossv[mask])

    css_prediction=np.zeros((rsq_mask_crossv.sum(),train_data.shape[1]))
    for g, vox_prf_pars in enumerate(prf_cv_fold_data):
        css_prediction[g] = css_model.generate_prediction(
            x=vox_prf_pars[0], y=vox_prf_pars[1], sigma=vox_prf_pars[2], n=vox_prf_pars[3], beta=vox_prf_pars[4], baseline=vox_prf_pars[5])

    all_residuals_css = train_data - css_prediction
    
    # some quick visualization
    f = pl.figure(figsize=(17,5))
    s = f.add_subplot(211)
    pl.plot(css_prediction[np.argmax(rsq_crossv[rsq_mask_crossv])], label='prediction')
    pl.plot(train_data[np.argmax(rsq_crossv[rsq_mask_crossv])], label='data')
    pl.plot(all_residuals_css[np.argmax(rsq_crossv[rsq_mask_crossv])], label='resid')
    pl.legend()
    s.set_title('best voxel')
    s = f.add_subplot(212)
    pl.plot(css_prediction[np.argmin(rsq_crossv[rsq_mask_crossv])], label='prediction')
    pl.plot(train_data[np.argmin(rsq_crossv[rsq_mask_crossv])], label='data')
    pl.plot(all_residuals_css[np.argmin(rsq_crossv[rsq_mask_crossv])], label='resid')
    pl.legend()
    s.set_title('worst voxel given this threshold')
    
    stimulus_covariance_WW = np.dot(rfs.T,rfs)
    all_residual_covariance_css = np.cov(all_residuals_css) 

    return prf_cv_fold_data, rfs, linear_predictor, all_residuals_css, all_residual_covariance_css, stimulus_covariance_WW, test_data, mask


def decode_cv_prfs(n_pix, rsq_threshold, use_median, n_folds, data_file, extent, screen_distance, screen_width, TR, **kwargs):
    
    # for key, value in kwargs.iteritems():
    #         key = value


    # set up results variables
    cv_decoded_image, cv_reshrot_recon, cv_reshrot_recon_m, \
    cv_omega, cv_estimated_tau_matrix, \
    cv_estimated_rho, cv_estimated_sigma = [], [], [], [], [], [], []

    for i in tqdm(range(n_folds)):
        # get the data
        (prf_cv_fold_data, rfs, linear_predictor, 
         all_residuals_css, all_residual_covariance_css, 
         stimulus_covariance_WW, test_data, mask) = setup_data_from_h5(
                                data_file = data_file, 
                                n_pix=n_pix, 
                                extent=extent, 
                                screen_distance=screen_distance, 
                                screen_width=screen_width, 
                                rsq_threshold=rsq_threshold,
                                TR=TR,
                                cv_fold=i,
                                n_folds=n_folds,
                                use_median=False)

        # estimate the covariance structure, which outputs all parameters
        (estimated_tau_matrix, estimated_rho, 
         estimated_sigma, omega, 
         omega_inv, logdet) = fit_model_omega(
                                        observed_residual_covariance=all_residual_covariance_css, 
                                        featurespace_covariance=stimulus_covariance_WW,
                                        verbose=1
                                        )

         # set up result array:
        dm_pixel_logl_ratio = np.zeros((mask.sum(),test_data.shape[1]))

        # and loop across timepoints
        for t, bold in enumerate(test_data.T):
            dm_pixel_logl_ratio[:,t] = firstpass_decoder_independent_Ws(
                                                bold=bold, 
                                                logdet=logdet,
                                                omega_inv=omega_inv,
                                                linear_predictor=linear_predictor)

        decoded_image = np.zeros((mask.sum(),test_data.shape[1]))  
        for t, bold in enumerate(test_data.T):
        # for t, bold in enumerate(tqdm(test_data.T)):
            
            starting_value=dm_pixel_logl_ratio[:,t]
            prf_data=prf_cv_fold_data
            bnds=[(-100,100) for elem in rfs]


            final_result=sp.optimize.minimize(
                                            calculate_bold_loglikelihood, 
                                            starting_value, 
                                            args=(  rfs,
                                                    prf_data, 
                                                    logdet, 
                                                    omega_inv, 
                                                    bold), 
                                            method='L-BFGS-B', 
                                            bounds=bnds,
                                            tol=1e-04,
                                            options={'disp':False})
            decoded_image[:,t] = final_result.x
            logl = -final_result.fun

        # fill in the mask
        recon = np.zeros([decoded_image.shape[1]]+list(mask.shape) )
        for t in range(decoded_image.shape[1]):
            recon[t,mask] = decoded_image[:,t]

        # rotate reconstructions to bar orientation
        thetas = [-1, 0, -1, 45, 270, -1,  315,  180, -1,  135,   90, -1,  225, -1]
        rotated_recon = np.copy(recon).T

        hrf_delay = 0
        block_delimiters = np.r_[np.arange(2, 462, 34) + hrf_delay, 462]
        reshrot_recon = np.zeros((8, rotated_recon.shape[0], rotated_recon.shape[1], 38))
        bar_counter = 0
        for i in range(len(block_delimiters) - 1):
            if thetas[i] != -1:
                rotated_recon[:, :, block_delimiters[i]:block_delimiters[i + 1] + 4] = rotate(rotated_recon[:, :, block_delimiters[i]:block_delimiters[i + 1] + 4],
                                                                                              axes=(
                    0, 1),
                    angle=thetas[i],
                    reshape=False,
                    mode='nearest')
                reshrot_recon[bar_counter] = rotated_recon[:, :,
                                                           block_delimiters[i]:block_delimiters[i + 1] + 4]
                bar_counter += 1

        reshrot_recon_m = np.median(reshrot_recon, axis=0)

        ##############################
        #   Save out results
        ##############################

        cv_decoded_image.append(decoded_image)
        cv_reshrot_recon.append(reshrot_recon)
        cv_reshrot_recon_m.append(reshrot_recon_m)
        cv_omega.append(omega)
        cv_estimated_tau_matrix.append(estimated_tau_matrix)
        cv_estimated_rho.append(estimated_rho)
        cv_estimated_sigma.append(estimated_sigma)


    cv_decoded_image = np.array(cv_decoded_image)
    cv_reshrot_recon = np.array(cv_reshrot_recon)
    cv_reshrot_recon_m = np.array(cv_reshrot_recon_m)
    cv_omega = np.array(cv_omega)
    cv_estimated_tau_matrix = np.array(cv_estimated_tau_matrix)
    cv_estimated_rho = np.array(cv_estimated_rho)
    cv_estimated_sigma = np.array(cv_estimated_sigma)

    return cv_decoded_image, cv_reshrot_recon, cv_reshrot_recon_m, cv_omega, cv_estimated_tau_matrix, cv_estimated_rho, cv_estimated_sigma
