import numpy as np
import scipy as sp



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

############################################################################################################################################
#   Method to obtain a decoding starting point from scratch and with no prior. Based on the assumption that channels are independent of each other.
#   Basis for the full decoding procedure.
#   The function allows the user to specify a nonlinear mapping relation on top of the standard linear model.
#   Takes as argument
#   bold: the observed bold signal that is to be decoded.
#   logdet: log of the determinant of model omega, one of the outputs of the fit_omega function
#   omega_inv: inverse of the model omega, one of the outputs of the fit_omega function
#   W: this is simply the W (features) matrix itself derived in the model fitting procedure. Size must be (n_voxels,n_features)
#   W can be interpreted as a simple linear prediction assuming all channels are independent of each other
#   mapping_relation: 'None', 'linear', 'power_law', 'cosine'. Or a list of these to be applied in sequence to the linear model,
#   in the same fashion as was done in the model fitting procedure. Parameters must be provided for all these transformation.
#   'linear' and 'cosine' require two parameters for each voxel. (slope and intercept for linear), (amplitude and phase for cosine)
#   returns
#   firstpass_image_normalized: firstpass decoded result
############################################################################################################################################


def firstpass_decoder_independent_channels( W,
                                            bold, 
                                            logdet,
                                            omega_inv,
                                            mapping_relation=None,
                                            mapping_parameters=[]):
    if logdet[0]!=1.0:
        print('Error: model covariance has negative or zero determinant')
        return
    const=-0.5*(logdet[1]+omega_inv.shape[0]*np.log(2*np.pi))
    
    #1 extra column for empty screen baseline
    non_linear_predictor_independent_channels =  np.zeros((W.shape[0], W.shape[1]+1))
    non_linear_predictor_independent_channels[:,1:]=np.copy(W)
    # possible mappings to implement nonlinear transformation
    if mapping_relation != None:
        if type(mapping_relation) == list:         
            for i, mr in enumerate(mapping_relation):
                non_linear_predictor_independent_channels = mapping(non_linear_predictor_independent_channels, mapping_relation=mr, parameters=mapping_parameters[i])
        else:
            non_linear_predictor_independent_channels = mapping(non_linear_predictor_independent_channels, mapping_relation=mapping_relation, parameters=mapping_parameters)
            

    # difference between bold response and linear predictor is residuals
    resid=np.tile(bold,(non_linear_predictor_independent_channels.shape[1],1)).T-non_linear_predictor_independent_channels

    # actual calculation here
    log_likelihood_indep_Ws=const - 0.5 * (resid * omega_inv.dot(resid)).sum(0)
    
    # all ll relative to 0, the empty screen
    baseline=log_likelihood_indep_Ws[0]

    # firstpass_image=np.reshape(baseline/log_likelihood_indep_pixels[1:],(mask.shape[0],mask.shape[1]))
    firstpass_image=baseline/log_likelihood_indep_Ws[1:]

    firstpass_image_normalized=(firstpass_image-np.min(firstpass_image))/(np.max(firstpass_image)-np.min(firstpass_image))
    
    return firstpass_image_normalized



def mapping(data, mapping_relation='linear', parameters=[]):
    """ mapping converts the linear model W* through a given mapping.
    mapping_relation indicates which type of transformation, 
    parameters describe the parameters to be used for each voxel, or W element.
    The parameters must be specified as a matrix of size (nr_voxels, nr_parameters)
    Where nr_parameters is 1 for the power_law mapping and 2 for linear and cosine transformation
    slightly counterintuitive notation thing: parameters0 defaults to 1 and parameters1 defaults to 0.
    This ensures that if parameters are not specified, the transformation is just an identity. (And a warning is printed)
    """
   
    if parameters == []:
        print("Warning: the mapping parameters were not specified. Using default values.")
        parameters = np.r_['1,2,0', np.ones(data.shape[0]), np.zeros(data.shape[0])]
        
    parameters0=np.ones(data.shape)
    parameters1=np.zeros(data.shape)
              
    if len(data.shape)!=1:
        if len(parameters.shape)!=1:
            parameters0 = np.tile(parameters[:,0],(data.shape[1],1)).T
            parameters1 = np.tile(parameters[:,1],(data.shape[1],1)).T
        else:
            parameters0=np.tile(parameters,(data.shape[1],1)).T
    else:
        if len(parameters.shape)!=1:
            parameters0=parameters[:,0]
            parameters1 = parameters[:,1]
        else:
            parameters0=parameters
        
    if mapping_relation == 'linear':
        return data * parameters0 + parameters1
    elif mapping_relation == 'power_law':
        return data ** parameters0
    elif mapping_relation == 'cosine':
        return parameters0*np.cos(data + parameters1)
    elif mapping_relation == 'exponential':
        return np.exp(parameters0 * data)
    elif mapping_relation == 'log':
        return np.log(data)



############################################################################################################################################
#   Strictly speaking this function calculates the probability of a hypothetical bold pattern, given some stimulus expressed as feature by feature combination.
#   If instead the bold signal is measured and the stimulus is hypothetical, the value returned by this function
#   is proportional to the posterior probability of that stimulus having produced the observed bold response, up to a normalization constant.
#   It is then used for calculating the (log)likelihood of the hypohtesized stimulus, given a measured bold.
#   arguments
#   stimulus: features x features stimulus, either hypothetical (if bold is measured) or real (if we want to evaluate the probability of a hypothetical bold pattern)
#   W: this is simply the W (features) matrix itself derived in the model fitting procedure. Size must be (n_voxels,n_pixels)
#   bold: the (observed or hypothetical) bold signal
#   logdet: log of the determinant of model omega, one of the outputs of the fit_omega function
#   omega_inv: inverse of the model omega, one of the outputs of the fit_omega function   
#   mapping_relation: 'None', 'linear', 'power_law', 'cosine'. Or a list of these to be applied in sequence to the linear model,
#   in the same fashion as was done in the model fitting procedure. Parameters must be provided for all these transformation.
#   'linear' and 'cosine' require two parameters for each voxel. (slope and intercept for linear), (amplitude and phase for cosine)
#   returns            
#   -log_likelihood of the hypothesized stimulus being produced by the observed bold signal (or viceversa)        
############################################################################################################################################

def calculate_bold_loglikelihood(   stimulus,
                                    W,
                                    bold,
                                    logdet,                                    
                                    omega_inv,
                                    mapping_relation=None,
                                    mapping_parameters=[]):

    const=-0.5*(logdet[1]+omega_inv.shape[0]*np.log(2*np.pi))

    linear_predictor = np.dot(W,stimulus)

    # possible mappings to implement nonlinear transformation
    if mapping_relation != None:
        if type(mapping_relation) == list:
            non_linear_predictor = linear_predictor
            for i, mr in enumerate(mapping_relation):
                non_linear_predictor = mapping(non_linear_predictor, mapping_relation=mr, parameters=mapping_parameters[i])
        else:
            non_linear_predictor = mapping(linear_predictor, mapping_relation=mapping_relation, parameters=mapping_parameters)
    else:
        non_linear_predictor = linear_predictor
    
    resid = bold - non_linear_predictor

    log_likelihood = const - 0.5 * np.dot(resid, np.dot(omega_inv, resid))

    return -log_likelihood

#simple function using Python built-in minimizer to get a more accurate reconstruction
#returns: optimized decoded stimulus and associated loglikelihood.    
def maximize_loglikelihood( starting_value,
                            W,                           
                            bold,
                            logdet,
                            omega_inv,                            
                            mapping_relation,
                            mapping_parameters):
    bnds=[(0,1) for elem in starting_value]

    final_result=sp.optimize.minimize(
                                    calculate_bold_loglikelihood, 
                                    starting_value, 
                                    args=(  W,
                                            bold,
                                            logdet,
                                            omega_inv,                            
                                            mapping_relation,
                                            mapping_parameters), 
                                    method='L-BFGS-B', 
                                    bounds=bnds,
                                    tol=1e-01,
                                    options={'disp':True})
    decoded_stimulus = final_result.x
    logl = -final_result.fun
    return logl, decoded_stimulus


