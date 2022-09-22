#!/usr/bin/env python3
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.utils.state import disable_all
from mqbench.advanced_ptq import ptq_reconstruction

def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m

mqb_mix_model = deepcopy(mqb_8bits_model)
disable_all(mqb_mix_model)
evaluate(test,mqb_mix_model)


# ## CLADO

# In[15]:


import torch.nn.functional as F
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
def kldiv(quant_logit,fp_logit):
    inp = F.log_softmax(quant_logit,dim=-1)
    tar = F.softmax(fp_logit,dim=-1)
    return kl_loss(inp,tar)

def perturb_loss(perturb_scheme,ref_metric=ref_metric,
                 eval_data=calib_data,printInfo=False,KL=False):

    global mqb_mix_model
    mqb_mix_model.eval()

    with torch.no_grad():
        # perturb layers
        perturb(perturb_scheme)

        # do evaluation
        if not KL:
            res = evaluate(eval_data,mqb_mix_model)
            perturbed_loss = res[ref_metric[0]] - ref_metric[1]
        else:
            perturbed_loss = []

            for (data,fp_out) in zip(calib_data,calib_fp_output):
                img,label = data
                quant_out = mqb_mix_model(img.cuda())
                perturbed_loss.append(kldiv(quant_out,fp_out))
            #print(perturbed_loss)
            perturbed_loss = torch.tensor(perturbed_loss).mean()

        if printInfo:
            print(f'use kl {KL} perturbed loss {perturbed_loss}')

        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)

    return perturbed_loss


# In[ ]:


# perturb loss functionality check
# del layer_input_map['conv1']
# del layer_input_map['fc']

# for layer in layer_input_map:
#     for a_bits in MPQ_scheme:
#         for w_bits in MPQ_scheme:
#             print(f'{layer} (a:{a_bits} bits,w:{w_bits} bits))')
#             p = perturb_loss({layer:(a_bits,w_bits)},eval_data=test,printInfo=True,KL=False)
#             #print(f'{layer} (a:{a_bits} bits,w:{w_bits} bits), accuracy degradation: {p*100:.2f}%')


# ## Build Cached Grad if not done before

# In[ ]:


del layer_input_map['conv1']
del layer_input_map['fc']

import time
s_time = time.time()
cached = {}
aw_scheme = []
for a_bits in MPQ_scheme:
    for w_bits in MPQ_scheme:
        aw_scheme.append((a_bits,w_bits))

aw_scheme = [(8,2),(8,4),(8,8)]


# In[ ]:


aw_scheme


# In[ ]:


KL=True
for n in layer_input_map:
    for m in layer_input_map:
        for naw in aw_scheme:
            for maw in aw_scheme:
                if (n,m,naw,maw) not in cached:
                    if n == m:
                        if naw == maw:

                            p = perturb_loss({n:naw},ref_metric,calib_data,KL=KL)
                            print(f'perturb layer {n} to A{naw[0]}W{naw[1]} p={p}')
                        else:
                            p = 0

                    else:

                        p = perturb_loss({n:naw,m:maw},ref_metric,calib_data,KL=KL)
                        print(f'perturb layer {n} to A{naw[0]}W{naw[1]} and layer {m} to A{maw[0]}W{maw[1]} p={p}')

                    cached[(n,m,naw,maw)] = cached[(m,n,maw,naw)] = p

print(f'{time.time()-s_time:.2f} seconds elapsed')


# In[ ]:


cached


# In[ ]:


layer_index = {}
cnt = 0
for layer in layer_input_map:
    for s in aw_scheme:
        layer_index[layer+f'{s}bits'] = cnt
        cnt += 1
L = cnt


# In[ ]:


layer_index


# In[ ]:


import numpy as np
hm = np.zeros(shape=(L,L))
for n in layer_input_map:
    for m in layer_input_map:
        for naw in aw_scheme:
            for maw in aw_scheme:
                hm[layer_index[n+f'{naw}bits'],layer_index[m+f'{maw}bits']] = cached[(n,m,naw,maw)]


# In[ ]:


cached_grad = np.zeros_like(hm)


# In[ ]:


import pickle
with open('generala248w248_i1kresnet50_calib','wb') as f:
    pickle.dump({'Ltilde':hm,'layer_index':layer_index},f)
