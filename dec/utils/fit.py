import numpy as np
import scipy as sp


############################################################################################################################################
#   Defining the function to fit residual covariance and model covariance following van Bergen et al. 2015
#   The model covariance here has terms for voxel-unique noise; shared noise; feature-space noise.
#   This function is defined to be minimized according to the scipy.optimize.minimize syntax. 
#   Takes as argument
#   observed_residual_covariance: (n_voxels,n_voxels) matrix. the observed covariance of residuals, to be calculated
#   in advance, depending on the model used
#   featurespace_covariance: that is W.dot(W.T) where W is a n_voxel * n_features matrix
#   in our case is the receptive fields covariance
############################################################################################################################################


def fit_model_omega(observed_residual_covariance, featurespace_covariance, infile=None, outfile=None, verbose=0):

   # or if possible load the result of the previous minimization
    if infile != None:
        x0=np.load(infile)
        initial_guesses = 1
    else:   # initial guesses around Van Bergen values
        initial_guesses = 2
        x0=np.zeros((observed_residual_covariance.shape[0]+2,initial_guesses))
        x0[0,:] = 0.1 # rho
        x0[1,:] = 0.3 # sigma
        x0[2:,:] = 0.7 * np.ones((observed_residual_covariance.shape[0], initial_guesses)) + \
                0.1 * np.random.randn( observed_residual_covariance.shape[0], initial_guesses)
#        x0[2:,:] = np.zeros((observed_residual_covariance.shape[0], initial_guesses))

    
    #suitable boundaries determined experimenally    
    bnds = [(-500,500) for xs in x0[:,0]]
    bnds[0]=(0,1)
    # bnds[1]=(0,1)
    
    def f(x, residual_covariance, W_matrix):
        rho=x[0]
        sigma=x[1]
        #tried to use the all_residual_covariance as tau_matrix: optimization fails (maybe use it as initial values for search. tried & failed)
        #tried to use stimulus_covariance as W_matrix: search was interrupted as it becomes several order of magnitudes slower.
        tau_matrix = np.outer(x[2:],x[2:])
        
        unique_variance = np.eye(tau_matrix.shape[0]) * (1-rho)
        shared_variance = tau_matrix * rho
        
        omega = shared_variance + unique_variance + sigma**2 * W_matrix
        
        return np.sum(np.square(residual_covariance - omega))
    
    #minimize distance between model covariance and observed covariance
    #This routine allows computation starting from multiple different initial conditions, in an attempt to avoid local minima
    best_fun=0
    for k in range(x0.shape[1]):
        result=sp.optimize.minimize(f, 
                                    x0[:,k], 
                                    args=(observed_residual_covariance, featurespace_covariance), 
                                    method='L-BFGS-B', 
                                    bounds=bnds,
                                    tol=1e-02,
                                    options={'disp':True})
        if k==0:
            best_fun=result.fun
            best_result=result
        if result.fun <= best_fun:
            best_fun=result.fun
            best_result=result
            
    better_result=sp.optimize.minimize(f, 
                                       best_result['x'], 
                                       args=(observed_residual_covariance, featurespace_covariance), 
                                       method='L-BFGS-B', 
                                       bounds=bnds, 
                                       tol=1e-06, 
                                       options={'disp':True,'maxfun': 15000000, 'factr': 10})
    
    #extract model covariance parameters and build omega
    x=better_result.x
    estimated_tau_matrix=np.outer(x[2:],x[2:])
    estimated_rho=x[0]
    estimated_sigma=x[1]
        
    model_omega=estimated_rho*estimated_tau_matrix+(1-estimated_rho)*np.multiply(np.identity(estimated_tau_matrix.shape[0]),estimated_tau_matrix)+estimated_sigma**2*featurespace_covariance
    model_omega_inv = np.linalg.inv(model_omega)
    logdet = np.linalg.slogdet(model_omega)


    if outfile != None:
        np.save(outfile,x)

    if verbose > 0:
        #print some details about omega for inspection and save
        print("max tau: "+str(np.max(x[2:]))+" min tau: "+str(np.min(x[2:])))
        print("sigma: "+str(estimated_sigma)+" rho: "+str(estimated_rho))
        #How good is the result?
        print("summed squared distance: "+str(np.sum(np.square(observed_residual_covariance-model_omega))))
        #Some sanity checks. 
        #Notice that determinants of data covariance and model covariance are extremely small, need to take log to make them manageable
        #print(np.linalg.slogdet(all_residual_covariance_css))
        #print(np.linalg.slogdet(model_omega))
    
    #The first test-optimization of parameters was done with a very rough 0.01 precision (distance ~7*10^5)
    #0.001 precision increased computational time and reduced distance (now ~6*10^5)
    #on server: ~3.9*10^5

    return estimated_tau_matrix, estimated_rho, estimated_sigma, model_omega, model_omega_inv, logdet



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
def firstpass_decoder_independent_Ws(   bold, 
                                            logdet,
                                            omega_inv,
                                            linear_predictor_ip):
    if logdet[0]!=1.0:
        print('Error: model covariance has negative or zero determinant')
        return
    const=-0.5*(logdet[1]+omega_inv.shape[0]*np.log(2*np.pi))

    # difference between bold response and linear predictor is residuals
    resid=np.tile(bold,(linear_predictor_ip.shape[1],1)).T-linear_predictor_ip

    # actual calculation here
    log_likelihood_indep_Ws=const - 0.5 * (resid * omega_inv.dot(resid)).sum(0)
    
    # all ll relative to 0, the empty screen
    baseline=log_likelihood_indep_Ws[0]

    # firstpass_image=np.reshape(baseline/log_likelihood_indep_pixels[1:],(mask.shape[0],mask.shape[1]))
    firstpass_image=baseline/log_likelihood_indep_Ws[1:]

    firstpass_image_normalized=(firstpass_image-np.min(firstpass_image))/(np.max(firstpass_image)-np.min(firstpass_image))
    
    return firstpass_image_normalized


############################################################################################################################################
#   This function calculates the probability of a hypothetical bold pattern, given some stimulus expressed pixel by pixel.
#   The entire model is captured by the receptive fields and the model covariance matrix (omega) which depends on rho,sigma,taus)
#   If instead the bold is measured and the stimulus is hypothetical, the value returned by this function
#   is proportional to the posterior probability of that stimulus having produced the observed bold response.
#   up to a normalization constant.
#   Calculate log-likelihood (logp) instead of p to deal with extremely small/large values.
############################################################################################################################################

def calculate_bold_loglikelihood(   stimulus,
                                    rfs,
                                    prf_data,
                                    logdet,
                                    omega_inv,
                                    bold):

    const=-0.5*(logdet[1]+omega_inv.shape[0]*np.log(2*np.pi))

    linear_predictor = np.dot(rfs.T,stimulus)
    #do some rescalings. This affects decoding quite a lot!
    # linear_predictor **= prf_data[:, 3]
    # #at this point (after power raising but before multiplication/subtraction) the css model convolves with hrf.
    # linear_predictor *= prf_data[:, 4]
    # linear_predictor += prf_data[:, 5]
    
    resid=bold-linear_predictor

    log_likelihood=const-0.5*np.dot(resid,np.dot(omega_inv,resid))

    return -log_likelihood

#simple function using Python built-in minimizer to get a more accurate reconstruction
def maximize_loglikelihood( starting_value,
                            bold,
                            logdet,
                            omega_inv,
                            rfs,
                            prf_data):
    bnds=[(0,1) for elem in rfs]

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
                                    tol=1e-01,
                                    options={'disp':True})
    decoded_stimulus = final_result.x
    logl = -final_result.fun
    return logl, decoded_stimulus


