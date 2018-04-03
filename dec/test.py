############################################################################################################################################
#   imports
############################################################################################################################################

import numpy as np
import scipy as sp
import tables
import ctypes
from popeye.spinach import generate_og_receptive_fields
# from popeye.css import CompressiveSpatialSummationModel
from popeye.visual_stimulus import VisualStimulus
from hrf_estimation.hrf import spmt
from scipy.signal import savgol_filter, fftconvolve, deconvolve
import matplotlib.pyplot as pl
import matplotlib.animation as animation

from matplotlib import animation
import os
import itertools

# import taken from own nPRF package. 
# this duplicates that code, which is unhealthy but should be fine for now
# in order to keep this repo self-contained.
from utils.utils import roi_data_from_hdf, create_visual_designmatrix_all, get_figshare_data, createCircularMask
from utils.css import CompressiveSpatialSummationModelFiltered
from fit import fit_model_omega, calculate_bold_loglikelihood, maximize_loglikelihood

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
stim_radius=9.0
n_pix=21
rsq_threshold = 0.5

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
timecourse_data_single_run = roi_data_from_hdf(['*psc'],'V1', hdf5_file,'psc').astype(np.float64)
timecourse_data_loo = roi_data_from_hdf(['*loo'],'V1', hdf5_file,'loo').astype(np.float64)
timecourse_data_all_psc = roi_data_from_hdf(['*av'],'V1', hdf5_file,'all_psc').astype(np.float64)
# prfs are per-run, as fit using the loo data
all_prf_data = roi_data_from_hdf(['*all'],'V1', hdf5_file,'all_prf').astype(np.float64)
prf_data = roi_data_from_hdf(['*all'],'V1', hdf5_file,'prf').astype(np.float64).reshape((all_prf_data.shape[0], -1, all_prf_data.shape[-1]))

# get design matrix, could create new one from utils.utils.create_visual_designmatrix_all
# h5file = tables.open_file(hdf5_file, mode="r")
# dm_n = h5file.get_node(
#                 where='/', name='dm', classname='Group')
# dm = dm_n.dm.read()
# h5file.close()

dm=create_visual_designmatrix_all(n_pixels=n_pix)
dm_crossv=np.tile(dm,(1,1,5))

# voxel subselection, using the 'all' rsq values
#rsq_mask = all_prf_data[:,-1] > rsq_threshold

#mask for crossvalidation
rsq_mask_crossv = np.mean(prf_data[:,:,-1], axis=1) > rsq_threshold
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
    dm_crossv, screen_distance, screen_width, 1.0 / 3.0, TR, ctypes.c_int16)
css_model = CompressiveSpatialSummationModelFiltered(stimulus, my_spmt)
css_model.hrf_delay = 0

############################################################################################################################################
#   setting up prf spatial profiles for subsequent covariances, now some per-run stuff was done
############################################################################################################################################
#use this number to choose which section to use in crossvalidate setup
crossv=0


deg_x, deg_y = np.meshgrid(np.linspace(extent[0], extent[1], n_pix, endpoint=True), np.linspace(
    extent[0], extent[1], n_pix, endpoint=True))

rfs = generate_og_receptive_fields(prf_data[rsq_mask_crossv, crossv, 0], prf_data[rsq_mask_crossv,crossv, 1], prf_data[rsq_mask_crossv,crossv, 2], np.ones((rsq_mask_crossv.sum())), deg_x, deg_y)

#this step is used in the css model
rfs /= ((2 * np.pi * prf_data[rsq_mask_crossv,crossv, 2]**2) * 1 /np.diff(css_model.stimulus.deg_x[0, 0:2])**2)


#rfs **= prf_data[rsq_mask2,i, 3]
#rfs *= prf_data[rsq_mask2,i, 4]
#rfs += prf_data[rsq_mask2,i, 5]

############################################################################################################################################
#   setting up covariances
############################################################################################################################################

#residuals for the actually linear model (weights*stimulus) (doesn't work)
test_data = timecourse_data_single_run[rsq_mask_crossv,nr_TRs*crossv:nr_TRs*(crossv+1)]

train_data = np.delete(timecourse_data_single_run[rsq_mask_crossv,:], np.s_[nr_TRs*crossv:nr_TRs*(crossv+1)], axis=1)

prediction= np.dot(rfs.reshape((-1,rfs.shape[-1])).T,dm_crossv.reshape((-1,dm_crossv.shape[-1])))


css_prediction=np.zeros((rsq_mask_crossv.sum(),train_data.shape[1]))
for g, vox_prf_pars in enumerate(prf_data[rsq_mask_crossv,crossv]):
    css_prediction[g] = css_model.generate_prediction(
        x=vox_prf_pars[0], y=vox_prf_pars[1], sigma=vox_prf_pars[2], n=vox_prf_pars[3], beta=vox_prf_pars[4], baseline=vox_prf_pars[5])

#could try to deconvolve bold to get neural response
#neural_response=np.copy(train_data)

#for time in range(prediction.shape[1]):
    #prediction[:,time] **= prf_data[rsq_mask_crossv,crossv, 3]
    #at this point (after power raising but before multiplication/subtraction) the css model convolves with hrf.
    #prediction[:,time] *= prf_data[rsq_mask_crossv,crossv, 4]
    #prediction[:,time] += prf_data[rsq_mask_crossv,crossv, 5]
    #try to go in the opposite direction from bold to neural response
    #neural_response[:,time] -= prf_data[rsq_mask2,i, 5]
    #neural_response[:,time] /= prf_data[rsq_mask2,i, 4]
    #neural_response[:,time] += savgol_filter(neural_response[:,time], window_length=css_model.sg_filter_window_length, polyorder=css_model.sg_filter_order,deriv=0, mode='nearest')
    #neural_response[:,time] **= (1/prf_data[rsq_mask2,i, 3]) 


#all_residuals_simple=train_data-prediction #2-simple     dm1 dm_crossv     rmask2 - rmask_crossv
all_residuals_css=train_data-css_prediction #1-css

#fig = pl.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(np.arange(all_residuals1.shape[0]*all_residuals2.shape[1]),np.ravel(all_residuals1-all_residuals2), s=1, c='k', marker="o")
#ax1.plot(np.arange(200),np.arange(200),'r')
#ax1.set_ylim(-100000,100000)
#pl.show()

#hrf = css_model.hrf_model(css_model.hrf_delay, css_model.stimulus.tr_length)+0.001
#bold=deconvolve(neural_response[0],hrf)
#pl.plot(bold[1][:200])
 
########moiving average attempt to deconvolve boldsimilar to berger
#def moving_average(array):
#    new_array=np.zeros((array.shape[0],array.shape[1]))
#    for i in range(array.shape[1]-8):
#        new_array[:,i]=(array[:,i+4]+array[:,i+5]+array[:,i+6]+array[:,i+7]+array[:,i+8])/5#+array[:,i+7]+array[:,i+8])/7
#    new_array[:,-8:-1]=np.zeros((new_array.shape[0],7))    
#    return new_array  
#  
#berger_test_bold=moving_average(sp.stats.zscore(test_data, axis=1))
#pl.plot(berger_test_bold[0,:])
#pl.plot(test_data[0,:]) 
#pl.plot(prediction[0,:462],'k',berger_test_bold[0,:], 'r')
#pl.plot(test_data[0,:])
#pl.plot(css_prediction[0,:])
#pl.plot(prediction[0,:])

#whatever=np.arange(train_data.shape[1])
#voxel_nr=300
#timesteps=400       
#pl.plot(whatever[:timesteps], train_data[voxel_nr,:timesteps], 'k', whatever[:timesteps], prediction[voxel_nr,:timesteps], 'r', whatever[:timesteps], css_prediction[voxel_nr,:timesteps], 'b')

#pl.plot(np.ravel(lols),np.ravel(train_data[:,:-8]),marker='ko',ms=1)#, np.arange(200),np.arange(200),'r')

#this does not work
#stimulus_covariance = np.cov(rfs.reshape((-1,rfs.shape[-1])).T)

#this is W*W.T=W_matrix. note that W is not rfs, but it is rfs recast in shape and transposed
#this matrix has information on how much the receptive fields of any two voxels overlap. this sort of works
#stimcovr2->stimcovar_WW
stimulus_covariance_WW = np.dot(rfs.reshape((-1,rfs.shape[-1])).T,rfs.reshape((-1,rfs.shape[-1])))

#normalized attempt, did not work
#stimulus_covariance3 = np.corrcoef(rfs.reshape((-1,rfs.shape[-1])).T)


#important: which residuals(==model) to use?
all_residual_covariance_css = np.cov(all_residuals_css) #allresidcovar-allresidcovar_css
#all_residual_covariance_simple = np.cov(all_residuals_simple)     #allresidcovar2-allresidcovar_simple

#pl.plot(np.ravel(all_residual_covariance),np.ravel(all_residual_covariance2),'ko',ms=1)

#initializing tau guess from variance does not help
#all_residual_variance_css = np.var(all_residuals_css, axis=1)

#old
#all_residual_covariance_diagonal = np.eye(all_residual_covariance.shape[0]) * all_residual_covariance # in-place multiplication


    
model_omega_inv,logdet_mo=fit_model_omega(all_residual_covariance_css, stimulus_covariance_WW)


mask = createCircularMask(n_pix,n_pix,radius=stim_radius)

#having a look at a sample for the term in the gaussian exponent. Omega inverse as expected has very large entries
#model might still work with a good estimate of omega
#omega_inv=np.linalg.inv(model_omega) 
#np.dot(all_residuals[:,1],np.dot(omega_inv,all_residuals[:,1]))




#Wmat=rfs.reshape((-1,rfs.shape[-1])).T
#linear_predictor=np.dot(W,np.ravel(dm_crossv[:,:,46]))

#Sanity check:
#pass!! maybe...
#vals=[55,120,160,220,255,326,360,427]
#maxpostindex=[]
#diff=[]
#p=np.zeros((462,len(vals)))
#for z in range(p.shape[1]):
#    for k in range(p.shape[0]):
        #conceptually, we are calculating the log-likelihood of all stimuli in DM having caused the bold response observed at time 50    
#        logl=calculate_bold_loglikelihood(timecourse_data_all_psc[rsq_mask,vals[z]],model_omega,rfs,dm[:,:,k],k)
#        print(str(vals[z])+" "+str(k)+" "+str(logl))    
#        p[k,z]=logl
#    maxpostindex.append(np.argmax(p[:,z]))
#    diff.append(p[vals[z],z]-np.amax(p[:,z]))    
#    print("Time of highest posterior for stimulus presented at t="+str(vals[z])+": "+str(np.argmax(p[:,z])))
#    print("Difference between exact stimulus posterior and max: "+str(diff[z]))


#STEPS/REASONING FOR FAST FIRSTPASS DECODER FUNCTION
#no need to create the many matrices of independent pixels. for each pixel
#the linear predictor would be a n_voxels sized vector where each voxel value
#is simply the value of its receptive field at the position of the pixel examined.
#(in the following, by n_pixels, I mean the total number of pixels, not the side of the square)
#therefore:
#1) create a matrix of linear predictors of size (n_voxels,n_pixels). This is simply the W matrix itself!
#(up to the CSS rescalings, so do those too)
#2) find the residuals. bold signal is a vector of size n_voxels. use np.tile
#to obtain a bold "matrix" so we can just do bold-W and that should give the residuals.
#3) now we want to calculate the stuff in the exponent of the gaussian distrib.
#this has to be done carefully, paying attention not just to the matrix sizes but 
#what they mean. It is possible to dot omega inv_with W. this will give again something of size
#(n_voxels,n_pixels). now, if we intuitively multiply the transposed residuals by this matrix,
#we will get something of size (n_pixels,n_pixels). However! we are only interested in the diagonal values
#because those would be the ones we look at when we do the normal procedure. all other values are
#not even calculated normally. are they useful? they represent
#some kind of "cross likelihood" among pixels, but has this any meaning? I dont know. but worth checking later
#however, skipping this full calculation and just calculating the diagonal elements should reduce computational time
#by quite a lot. lets do without for now. use this simple linear algebra trick to calculate only the needed
#elements.
#trial=(RESID * model_omega_inv.dot(RESID)).sum(0)
def firstpass_decoder_independent_pixels(bold,logdet,omega_inv,rfs,prf_dataa):
    # logdet=np.linalg.slogdet(omega)
    if logdet[0]!=1.0:
        print('Error: model covariance has negative or zero determinant')
        return
    const=-0.5*(logdet[1]+omega_inv.shape[0]*np.log(2*np.pi))
    W=rfs.reshape((-1,rfs.shape[-1])).T
    #important change: use the actual model prediction (prf_predictions contains the model predicted bold response for the given dm matrix. Could replace dm matrix with random
    #stimuli as further test
    
    #important change here: until now, we were fitting the previous stuff on residuals from the "css" model and then trying to predict based on a simple
    #linear model Weights*Stimulus. Upon inspection, css and simple models have different residuals wrt data so they are different in contrast to what I was told.
    #I also tried to redo all fitting and testing with the simple linear model. did not work
    linear_predictor=np.zeros((W.shape[0], W.shape[1]+1))
    linear_predictor[:,1:]=np.copy(W)
    #do some rescalings. This affects decoding quite a lot!
    linear_predictor **= np.tile(prf_dataa[rsq_mask_crossv,crossv, 3], (linear_predictor.shape[1],1)).T
    #at this point (after power raising but before multiplication/subtraction) the css model convolves with hrf.
    linear_predictor *= np.tile(prf_dataa[rsq_mask_crossv,crossv, 4], (linear_predictor.shape[1],1)).T
    linear_predictor += np.tile(prf_dataa[rsq_mask_crossv,crossv, 5], (linear_predictor.shape[1],1)).T
    #bold=np.random.rand(444)
    resid=np.tile(bold,(linear_predictor.shape[1],1)).T-linear_predictor
    # I'm tired, I should check this
    log_likelihood_indep_pixels=const - 0.5 * (resid * omega_inv.dot(resid)).sum(0)
    baseline=log_likelihood_indep_pixels[0]
    firstpass_image=np.reshape(baseline/log_likelihood_indep_pixels[1:],(rfs.shape[0],rfs.shape[1]))
    firstpass_image_normalized=(firstpass_image-np.min(firstpass_image[mask]))/(np.max(firstpass_image[mask])-np.min(firstpass_image[mask]))
    #apply circular mask
    firstpass_image_normalized[~mask]=0
    return firstpass_image_normalized


#np.shape(np.tile(prf_data[rsq_mask_crossv,crossv, 3], (901,1)).T) 
#Create a matrix to test each pixel independently        
#total_pixels=n_pix*n_pix
#dm_independent_pixels=np.zeros((n_pix,n_pix,total_pixels+1))
#ff=1
#for i in np.arange(n_pix):
#    for j in np.arange(n_pix):
#        dm_independent_pixels[i,j,ff]=1
#        ff+=1
#        
#Select the data to decode. Not bad!
#################################################################################
#   Later this section will be the test.py and eall functions above moved to other files
#################################################################################

start=0
end=460
test_data_decode=np.copy(test_data[:,start:end])
#decode and plot
#dm_pixel_logl = np.zeros((n_pix,n_pix,test_data_decode.shape[1]))
dm_pixel_logl_ratio = np.zeros((n_pix,n_pix,test_data_decode.shape[1]))
decoded_image = np.zeros((n_pix,n_pix,test_data_decode.shape[1]))  
  
#dm_pixel_logl_ratio2 = np.zeros((n_pix,n_pix,test_data_decode.shape[1]))  

  
#baseline_=np.zeros(test_data_decode.shape[1]) 
result_corrcoef=np.zeros(test_data_decode.shape[1])#-4) 

for t in range(test_data_decode.shape[1]):
    
#    baseline_[t]=calculate_bold_loglikelihood(test_data_decode[:,t],logdet_mo,model_omega_inv,rfs,dm_independent_pixels[:,:,0],prf_data)
    print(str(int(t*100/test_data_decode.shape[1]))+'% Completed')
#    ff=1
#    for i in np.arange(n_pix):
#        for j in np.arange(n_pix):
#            dm_pixel_logl[i,j,t]=calculate_bold_loglikelihood(test_data_decode[:,t],logdet_mo,model_omega_inv,rfs,dm_independent_pixels[:,:,ff],prf_data)
#            ff+=1
#            dm_pixel_logl_ratio2[i,j,t] = baseline_[t]/dm_pixel_logl[i,j,t]
#            
    dm_pixel_logl_ratio[:,:,t]=firstpass_decoder_independent_pixels(test_data_decode[:,t],logdet_mo,model_omega_inv,rfs,prf_data)
    logl, decoded_image[:,:,t]=maximize_loglikelihood(dm_pixel_logl_ratio[:,:,t],test_data_decode[:,t],logdet_mo,model_omega_inv,rfs,prf_data)
    print(logl)
    # pl.imshow(dm_crossv[:,:,start+t])
    # pl.show()
    #pl.imshow(dm_pixel_logl_ratio[:,:,t])
    #pl.show()
    # pl.imshow(decoded_image[:,:,t])
    # pl.show()

#    pl.imshow(dm_pixel_logl_ratio2[:,:,t])
#    pl.show()

# np.save(file='data/decoded_image',arr=decoded_image)
# np.save(file='data/dm_pixel_logl_ratio',arr=dm_pixel_logl_ratio)

decoded_image = np.load('data/decoded_image.npy')
dm_pixel_logl_ratio = np.load('data/dm_pixel_logl_ratio.npy')

# set up figure
fig = pl.figure(figsize=(6, 6))
ims = []

for t in range(test_data_decode.shape[1]):
   
    #the video animation is currently not working for me. not sure how to fix it
    # fig.gca().add_patch(pl.Circle((int(n_pix/2), int(n_pix/2)), radius=8, facecolor='w',
    #                             edgecolor='k', fill=False, linewidth=3.0))
    extentz=[int(n_pix/2)-stim_radius,int(n_pix/2)+stim_radius,int(n_pix/2)-stim_radius,int(n_pix/2)+stim_radius,]
    # im = pl.imshow(dm_pixel_logl_ratio[:,:,t], animated=True, clim=[dm_pixel_logl_ratio.min(
    # ), dm_pixel_logl_ratio.max()], cmap='viridis', alpha=0.95)#, extent=extentz)    
    im = pl.imshow(decoded_image[:,:,t], animated=True, clim=[decoded_image.min(
    ), decoded_image.max()], cmap='viridis', alpha=0.95)#, extent=extentz)
    pl.axis('off')
    ims.append([im])

ani = animation.ArtistAnimation(
    fig, ims, interval=5, blit=True, repeat_delay=150)
# writer=writer, , codec='hevc'
ani.save('data/out.mp4', dpi=150, bitrate=1800, codec='hevc')

#try to calculate correlation between decoded and actual image. If bold not deconvolved, need to account for hemodynamic delay
delay=-4            
dm_roll=np.roll(np.flip(dm_pixel_logl_ratio,axis=1),delay,axis=2)            
for i in np.arange(result_corrcoef.shape[0]):     
    result_corrcoef[i] = np.corrcoef(np.ravel(dm_roll[:,:,i]),np.ravel(dm_crossv[:,:,start+i]))[0,1]
pl.plot(result_corrcoef)
np.nanmean(result_corrcoef)    
#next: "smart" function to optimize the posterior. (hierarchical prior? flip-and-keep with continuous values? flip-and-keep +proximity-biased search?)
#Perhaps choose a different approach i.e. define receptive fields that cover the screen  
#is it possible to improve the CSS model itself?
#also look into whether it is possible to improve the omega fit (other terms in the model), and whether deconvolving the bold helps.  
#also: currently we do the first pass by taking ratio of baseline/logl because this value increases as
#the pixel activation becomes more likely. But this is not the only way; could use difference instead of ratio
#could use an absolute scale rather than a relative one; could use a cutoff and discrete values; etc etc
            