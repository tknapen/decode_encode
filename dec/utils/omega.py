import numpy as np
import scipy as sp
from scipy.sparse.linalg import arpack

############################################################################################################################################
#   Defining the function to fit residual covariance and model covariance following van Bergen et al. 2015
#   The model covariance here has terms for voxel-unique noise; shared noise; feature-space noise.
#   This function is defined to be minimized according to the scipy.optimize.minimize syntax. 
#   Takes as argument
#   observed_residual_covariance: (n_voxels,n_voxels) matrix. the observed covariance of residuals differences
#   between the model and the data.
#   WWT: that is W.dot(W.T) where W is the n_voxel * n_features matrix obtained in previous model fitting procedure
#   D: voxels by voxels distance matrix in some chosen matrix. Must be a distance so all positive values and zeroes on the diagonal.
#   infile: load a vector of rho, sigma and tau parameters (which define omega) from a previous saved omega calculation
#   returns
#   
############################################################################################################################################


def fit_model_omega(observed_residual_covariance, WWT, D=None, infile=None, outfile=None, verbose=0):
    if D!=None:
        if not isPSD(D, tol = 1e-3):
            print("Please check the distance matrix provided. It appears to not be suitable.")
            return None
    
    if not isPSD(observed_residual_covariance, tol = 1e-3):
        print("Please check the residual covaricne matrix provided. It appears to not be a suitable covariance matrix.")
        return None
    
   # or if possible load the result of the previous minimization
    if infile != None:
        x0=np.load(infile)
        initial_guesses = 1
    else:   # initial guesses around Van Bergen values
        initial_guesses = 2
        x0=np.zeros((observed_residual_covariance.shape[0]+3,initial_guesses))
        x0[0,:] = 0.5 #alpha
        x0[1,:] = 0.1 # rho
        x0[2,:] = 0.3 # sigma
        x0[3:,:] = 0.7 * np.ones((observed_residual_covariance.shape[0], initial_guesses)) + \
                0.1 * np.random.randn( observed_residual_covariance.shape[0], initial_guesses)
#        x0[2:,:] = np.zeros((observed_residual_covariance.shape[0], initial_guesses))

    
    #suitable boundaries determined experimenally    
    bnds = [(-500,500) for xs in x0[:,0]]
    if D==None:
        bnds[0]=(0,0)
        
    bnds[1]=(0,1)
    bnds[2]=(0,500)
    
    def f(x, residual_covariance, WWT, Distance):
        alpha=x[0]
        rho=x[1]
        sigma=x[2]
        #tried to use the all_residual_covariance as tau_matrix: optimization fails (maybe use it as initial values for search. tried & failed)
        #tried to use stimulus_covariance as WWT: search was interrupted as it becomes several order of magnitudes slower.
        tau_matrix = np.outer(x[3:],x[3:])
        
        unique_variance = np.eye(tau_matrix.shape[0]) * (1-rho) * tau_matrix
        shared_variance = tau_matrix * rho
        
        
        if Distance==None:
            omega = shared_variance + unique_variance + (sigma**2) * WWT
        else:
            distance_variance = alpha * Distance * tau_matrix
            omega = distance_variance + shared_variance + unique_variance + (sigma**2) * WWT
        
        return np.sum(np.square(residual_covariance - omega))
    
    #minimize distance between model covariance and observed covariance
    #This routine allows computation starting from multiple different initial conditions, in an attempt to avoid local minima
    best_fun=0
    for k in range(x0.shape[1]):
        result=sp.optimize.minimize(f, 
                                    x0[:,k], 
                                    args=(observed_residual_covariance, WWT,D), 
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
                                       args=(observed_residual_covariance, WWT,D), 
                                       method='L-BFGS-B', 
                                       bounds=bnds, 
                                       tol=1e-06, 
                                       options={'disp':True,'maxfun': 15000000, 'factr': 10})
    
    #extract model covariance parameters and build omega
    x=better_result.x
    estimated_tau_matrix=np.outer(x[3:],x[3:])
    estimated_alpha=x[0]
    estimated_rho=x[1]
    estimated_sigma=x[2]
    
    if D==None:
        model_omega=estimated_rho*estimated_tau_matrix+(1-estimated_rho)*np.multiply(np.identity(estimated_tau_matrix.shape[0]),estimated_tau_matrix)+(estimated_sigma**2)*WWT
    else:
        model_omega=estimated_alpha*D*estimated_tau_matrix + estimated_rho*estimated_tau_matrix+(1-estimated_rho)*np.multiply(np.identity(estimated_tau_matrix.shape[0]),estimated_tau_matrix)+(estimated_sigma**2)*WWT
    
    model_omega_inv = np.linalg.inv(model_omega)
    logdet = np.linalg.slogdet(model_omega)
    
    if not isPSD(model_omega, tol = 1e-3):
        print("The fit model omega appears to not be a suitable covariance matrix.")
        return None

    if outfile != None:
        np.save(outfile,x)

    if verbose > 0:
        #print some details about omega for inspection and save
        print("max tau: "+str(np.max(x[3:]))+" min tau: "+str(np.min(x[3:])))
        print("sigma: "+str(estimated_sigma)+" rho: "+str(estimated_rho)+" alpha: "+str(estimated_alpha))
        #How good is the result?
        print("summed squared distance: "+str(np.sum(np.square(observed_residual_covariance-model_omega))))
        #Some sanity checks. 
        #Notice that determinants of data covariance and model covariance are extremely small, need to take log to make them manageable
        #print(np.linalg.slogdet(all_residual_covariance_css))
        #print(np.linalg.slogdet(model_omega))
    
    #The first test-optimization of parameters was done with a very rough 0.01 precision (distance ~7*10^5)
    #0.001 precision increased computational time and reduced distance (now ~6*10^5)
    #on server: ~3.9*10^5

    return estimated_tau_matrix, estimated_rho, estimated_sigma, estimated_alpha, model_omega, model_omega_inv, logdet

#function for some sanity checks within the omega estimation procedure

def isPSD(A, tol = 1e-8):
    vals = np.linalg.eigvalsh(A) # return the ends of spectrum of A
    return np.all(vals > -tol)