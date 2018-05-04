# decode_encode

This repo contains code to optimally decode stimulus features from a previously fit encoding model. 

The first use-case is for the reconstruction of visual images from brain activations after population receptive field mapping of these voxels. 



## requirements
numpy, scipy, lmfit, popeye, pytables, hrf_estimation


## Steps

1. fit pRF profiles
2. Generate residual timecourses for fitted data:
    a. leave-one-out, predict timecourses of loo separate runs (concatenated) based on pRF profiles
4. We use the covariance models originally developed in: van Bergen, R. S., Ma, W. J., Pratte, M. S., & Jehee, J. F. M. (2015). Sensory uncertainty decoded from visual cortex predicts behavior. Nature Neuroscience, 18(12), 1728–1730. http://doi.org/10.1038/nn.4150, to optimally capture voxel covariance
5. Standard Bayesian conditional/posterior definitions for multivariate Gaussian residuals (As in van Bergen or Nishimoto)
5. "Firstpass" independent-pixels decoder to avoid combinatorial explosion and prior-bias. 
6. Standard LL minimization to obtain best available (time-independent) decoding.

###TODO
Handle time dependency


#### Leave-one-out separation

loo fits were done on loo data: run_1 is the median over runs 2-6. For residuals we have to concatenate runs 2-6, and use the prf prediction from the prf run_1. Then, the test set is the psc/run_1 data.




