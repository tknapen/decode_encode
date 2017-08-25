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

#residuals for the actually linear model (weights*stimulus) (doesn't work)
#all_residuals2=timecourse_data_all_psc[rsq_mask]-np.dot(rfs.reshape((-1,rfs.shape[-1])).T,dm.reshape((-1,dm.shape[-1])))

#this does not work
#stimulus_covariance = np.cov(rfs.reshape((-1,rfs.shape[-1])).T)
#this is W*W.T=W_matrix. note that W is not rfs, but it is rfs recast in shape and transposed
#this matrix has information on how much the receptive fields of any two voxels overlap. this sort of works
stimulus_covariance2 = np.dot(rfs.reshape((-1,rfs.shape[-1])).T,rfs.reshape((-1,rfs.shape[-1])))
#normalized attempt, did not work
#stimulus_covariance3 = np.corrcoef(rfs.reshape((-1,rfs.shape[-1])).T)



all_residual_covariance = np.cov(all_residuals)
#initializing tau guess from variance does not help
#all_residual_variance = np.var(all_residuals, axis=1)
#all_residual_covariance_diagonal = np.eye(all_residual_covariance.shape[0]) * all_residual_covariance # in-place multiplication


############################################################################################################################################
#   Defining the distance function between residual covariance and model covariance following van Bergen et al. 2015
#   The model covariance here has terms for voxel-unique noise; shared noise; feature-space noise.
#   This function is defined to be minimized according to the scipy.optimize.minimize syntax. 
#   Takes as argument
#   x: one dimensional vector of length 2+#voxels, the parameters to be optimized. (rho,sigma,tau vector)
#   omega: matrix of size #voxels x #voxels. This is the covariance matrix estimated from the data
#   W_matrix: matrix of size #voxels x #voxels. This is the matrix product between the weight matrix and its own transpose
#   (the weight matrix (fitted receptive fields here) has size #pixels x #voxels)    
############################################################################################################################################

#initial guess and boundaries
#number of minimization attempts initial guesses
initial_guesses=2

x0=np.zeros((all_residual_covariance.shape[0]+2,initial_guesses))+0.5
#none of these work very well
#x0[:,1]=np.random.rand(x0.shape[0])
#x0[:,2]=5*np.random.rand(x0.shape[0])
#x0[:,3]=10*np.random.rand(x0.shape[0])

#stuiable start values determined experimentally
for s in range(x0.shape[1]):
    x0[1,s]=0.025
    x0[0,s]=0.5

#load the result of the previous minimization    
x0[:,0]=np.load("outfile.npy")

#suitable boundaries determined experimenally    
bnds = [(-5,15) for xs in x0[:,0]]
bnds[0]=(0,1)
bnds[1]=(0,1)
def f(x, residual_covariance, W_matrix):
    rho=x[0]
    sigma=x[1]
    #tried to use the all_residual_covariance as tau_matrix: optimization fails (maybe use it as initial values for search. tried & failed)
    #tried to use stimulus_covariance as W_matrix: search was interrupted as it becomes several order of magnitudes slower.
    tau_matrix = np.outer(x[2:],x[2:])
    return np.sum(np.square(residual_covariance-rho*tau_matrix-(1-rho)*np.multiply(np.identity(residual_covariance.shape[0]),tau_matrix)-sigma**2*W_matrix))

#minimize distance between model covariance and observed covariance
#This routine allows computation starting from multiple different initial conditions, in an attempt to avoid local minima
best_fun=0
for k in range(x0.shape[1]-1):
    result=sp.optimize.minimize(f, x0[:,k], args=(all_residual_covariance,stimulus_covariance2), method='TNC', bounds=bnds,tol=1e-02,options={'disp':True})
    if k==0:
        best_fun=result.fun
    if result.fun <= best_fun:
        best_fun=result.fun
        best_result=result
        
better_result=sp.optimize.minimize(f, best_result.x, args=(all_residual_covariance,stimulus_covariance2), method='L-BFGS-B', bounds=bnds,options={'disp':True,'maxfun': 15000000, 'factr': 10})

#extract model covariance parameters and build omega
estimated_tau_matrix=np.outer(better_result.x[2:],better_result.x[2:])
estimated_rho=better_result.x[0]
estimated_sigma=better_result.x[1]

#print some details about omega for inspection and save
print(str(np.max(better_result.x[2:]))+" "+str(np.min(better_result.x[2:])))
print(str(estimated_sigma)+" "+str(estimated_rho))
np.save("outfile",better_result.x)
model_omega=estimated_rho*estimated_tau_matrix+(1-estimated_rho)*np.multiply(np.identity(estimated_tau_matrix.shape[0]),estimated_tau_matrix)+estimated_sigma**2*stimulus_covariance2


#How good is the result?
np.sum(np.square(all_residual_covariance-model_omega))
#The first test-optimization of parameters was done with a very rough 0.01 precision (distance ~7*10^5)
#0.001 precision increased computational time and reduced distance (now ~6*10^5)
#on server: ~3.9*10^5

#Some sanity checks. 
#Notice that determinants of data covariance and model covariance are extremely small, need to take log to make them manageable
print(np.linalg.slogdet(all_residual_covariance))
print(np.linalg.slogdet(model_omega))
#having a look at a sample for the term in the gaussian exponent. Omega inverse as expected has very large entries
#model might still work with a good estimate of omega
#omega_inv=np.linalg.inv(model_omega) 
#np.dot(all_residuals[:,1],np.dot(omega_inv,all_residuals[:,1]))

############################################################################################################################################
#   This function calculates the probability of a hypothetical bold pattern, given some stimulus expressed pixel by pixel.
#   The entire model is captured by the receptive fields and the model covariance matrix (omega) which depends on rho,sigma,taus)
#   If instead the bold is measured and the stimulus is hypothetical, the value returned by this function
#   is proportional to the posterior probability of that stimulus having produced the observed bold response.
#   up to a normalization constant.
#   Calculate log-likelihood (logp) instead of p to deal with extremely small/large values.
############################################################################################################################################

def calculate_bold_loglikelihood(bold,omega,rfs,stimulus,tt):
    logdet=np.linalg.slogdet(omega)
    if logdet[0]!=1.0:
        print('Error: model covariance has negative or zero determinant')
        return
    const=-0.5*(logdet[1]+omega.shape[0]*np.log(2*np.pi))
    W=rfs.reshape((-1,rfs.shape[-1])).T
    #important change: use the actual model prediction (prf_predictions contains the model predicted bold response for the given dm matrix. Could replace dm matrix with random
    #stimuli as further test
    
    #important change here: until now, we were fitting the previous stuff on residuals from the "css" model and then trying to predict based on a simple
    #linear model Weights*Stimulus. Upon inspection, css and simple models have different residuals wrt data so they are different in contrast to what I was told.
    #I also tried to redo all fitting and testing with the simple linear model. did not work
    linear_predictor=prf_predictions[:,tt] #np.dot(W,stimulus.reshape(W.shape[1]))
    resid=bold-linear_predictor
  
    log_likelihood=const-0.5*np.dot(resid,np.dot(np.linalg.inv(omega),resid))
    return log_likelihood

#Sanity check:
#pass!!
p=np.zeros(462)
for k in range(462):
    #conceptually, we are calculating the log-likelihood of all stimuli in DM having caused the bold response observed at time 50    
    logl=calculate_bold_loglikelihood(timecourse_data_all_psc[rsq_mask,50],model_omega,rfs,dm[:,:,k],k)
    print(str(k)+" "+str(logl))    
    #p[k]=np.exp(logl)
 
#next: train on loo data and test on left-out. Ask Tomas for details on data setup & css model.
    
#next: "smart" function to optimize the posterior. (hierarchical prior? flip-and-keep with continuous values? flip-and-keep +proximity-biased search?)
#problem: finding the normalization constant would require 2^(n_pixels) calculations, which is not feasible.
#Perhaps choose a different approach i.e. define receptive fields that cover the screen      
            