# decode_encode

This repo contains code to optimally decode stimulus features from a previously fit encoding model. 

The first use-case is for the reconstruction of visual images from brain activations after population receptive field mapping of these voxels. 



## requirements
numpy, scipy, lmfit, popeye, pytables, hrf_estimation


## Steps

1. fit pRF profiles (done before)
2. Generate residual timecourses for fitted data:
    a. leave-one-out, predict timecourses of loo separate runs (concatenated) based on pRF profiles
4. Use those residuals for estimation of $\Omega$
5. Using (6) in van Bergen, take the instantaneous activation pattern, and calculate for each pixel that pixel value that maximizes the probability of this bold pattern. This should be a parallel problem over pixels. 




## To DO:

#### Different options for the posterior:

- start assuming independence; calculate maximal posterior for all elements of the feature space separately.
- start with dot-product reconstruction; then iterate from this 'readout'
- hierarchical reconstruction: subsample the feature space, calculate posterior, upsample the feature space, calculate posterior using previous subsampled feature space solution:
    + 4x4 pixels posterior -> 8x8 pixels posterior -> 16x16 pixels posterior ...
- smart sampling: NUTS or something?

#### Leave-one-out separation

loo fits were done on loo data: run_1 is the median over runs 2-6. For residuals we have to concatenate runs 2-6, and use the prf prediction from the prf run_1. Then, the test set is the psc/run_1 data.




