# Configuration Documentation

This file explains the parameters used in the config.json file.

## normalise-cloud
This option accepts boolean input. With "0", clouds are not normalized.  
Here normalizing means scaling all the points in both the source and the target clouds to bring them in [0 1] range.  
For velodyne scans, we need to set this option to 1 for optimizers to work because of their large range.  
For kinect scans, it is better not to normalize the clouds. But optimizers work with both options.


## method

This option lets us choose the optimizer. Currently we have following options:  

Currently we have two choices of same algorithm (preconditioned SGLD or Bayesian ICP) which differ in how gradients are passed to the optimizers.

### preconditioned_sgld 
This option implements algorithm 1 of <https://arxiv.org/pdf/1512.07666.pdf>. Here preconditioner is obtained using the gradients of the likelihood alone. The gradients of prior is added directly in the update equation of the optimizer.

All the parameters for this method are standard except the *"adjust_noise"* parameter which we might need to adjust the level of noise in MCMC chain for very small step size.   

### preconditioned_sgld2
This option implements the algorithm 2 of <http://openaccess.thecvf.com/content_cvpr_2016/papers/Li_Learning_Weight_Uncertainty_CVPR_2016_paper.pdf>.  With this option gradients of prior along with gradients of likelihhod function together are used for getting the preconditioner.  

These two options implement **Bayesian ICP**  method <https://arxiv.org/pdf/2004.07973.pdf> which provides us samples of the distributions of pose parameters.
In Bayesian ICP amount of noise depends on the step size. In case of very small step size, we may need to amplify the amount of noise to explore full parameter space with small number of particles (~1500 to 2000). We may need to tune *"adjust_noise"* parameter.


Following optimizers implement non-Bayesian ICP method (**SGD-ICP**) <https://arxiv.org/pdf/1907.09133.pdf>. 

1. **fixed**
2. **adam**
3. **adadelta**
4. **momentum**
5. **rmsprop**


## icp

*batchsize* : batch size defines the number of points in a mini-batch. For SGD-ICP sensible range is 50 to 300.      
*max-match-distance* : this option lets us set the distance beyond which we want to ignore the correspondences which might be beneficial in case of partial overlap.    
*max-iterations* : for Bayesian ICP ("preconditioned_sgld"), number of iterations = number of samples. For SGD-ICP, "max-iterations" depends on the initial offset between the two clouds.     

*convergence-steps* : if change in pose is less than or equal to "translation-convergence" and "rotation-convergence" for "convergence-step" times, then alignment process is terminated even number of iterations are not exhausted.   
*translational-convergence* : translation threshold (change in translation pose parameters between iterations).    
*rotational-convergence* : rotational threshold (change in rotation pose parameters between iterations)   
*filter* : If set to 1, multiple correspondences from source cloud to target cloud is ignored.   

If we use Bayesian ICP, we do not want to terminate the alignment process untill we get the desired number of samples. For this, we need to set "convergence-steps" to a large number (eg. 20), and set "translation-convergence" and "rotational-convergence" to very very small number (eg. 0.000000001).

For SGD-ICP, the sensible values for "convergence-steps", "translational-convergence", and "rotational-convergence" are 5, 0.005, 0.005 respectively.
