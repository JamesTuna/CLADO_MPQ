# QAVAT_PLUS
## Updates 2022-09-26
- Two more package needed
- ```conda install -c conda-forge pyscipopt=3.5.0```
- ```pip install cvxpy-base```
- Hyperparameter tuning is no longer needed in second phase: optimization
- We formulate the optimization as a Mixed Integer Quadratic Constraint Programming (MIQCP)
- We use the CVXPY solver to solve the MIQCP, PSD approximation of ```cached_grad``` was neccessary for MIQCP and was found empirically useful on CIFAR100 sanity check
- Issue of cvxpy MIQCP solver: depends on pyscipopt, may run forever for some unknown reasons on CIFAR100 experiments. 
- pyscipopt=3.5.0: no problem for ```cached_grad``` = CachedGrad_a248w248c100_resnet56.pkl; stuck for ```cached_grad``` = CachedGrad_a248w248c100_resnet56KL.pkl
- pyscipopt=3.1.0: stuck for ```cached_grad``` = CachedGrad_a248w248c100_resnet56.pkl

### 1. CLADO_XXX.ipynb
- binaryWeight: only consider binary quantization (quantize or not) for weights (tested on CIFAR10/100, superiority over naive method confirmed)
- multiWeight: multi-scheme weight quantization (e.g., 2,4,8 bits) for weights (tested on CIFAR10, superiority over naive method confirmed)
- general: multi-scheme (weight,activation) quantization. Problems encountered in sanity check on CIFAR100. 

### 2. How CLADO_XXX.ipynb works
- CLADO_general.ipynb is a more general version of the other two and we should use it
- it perturb pairs of layers and save the change of loss to a pkl file
- the pkl file (e.g.,generala248w248_c10resnet56_calib_kl) contains a 2D matrix capturing the change of loss when perturbing layer (x,y)
- the map of layer index (x,y) to quantization decision of layers are also stored in the pkl file
- the pkl file will be loaded and used to solve the constraint optimization problem
- naive argument, when set to True, cross-layer terms won't be used in optimization
- KL argument, when set to True, perturbation on KL divergence instead of loss will be used as cached gradient in optimization
- so when set KL and naive to be both True, CLADO becomes ZeroQ

### 3. Sanity check problem
- Ideally, when set constraint to null, the optimziation should return an 8-bit model, or model that have close-to-FP performance.
- However, we found that the sanity check doesn't pass for CIFAR100-RESNET56 when considering 3x3 options for each layer (A/W 2,4,8 bits)
- The problem is not only for CLADO, but also for ZeroQ
- Negative estimated perturbed loss/KL observed, very likely due to that overly large quantization error is not well captured by first order (ZeroQ) and second order Taylor (CLADO)
