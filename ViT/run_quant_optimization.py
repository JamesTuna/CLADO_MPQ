#!/usr/bin/env python

import torch
import pickle
import numpy as np

from utils import (
    format_space,
    clado_proxy_post_process,
    qip_optimizer
)

def feintLady():

    proxy = torch.load("./cached_files/clado_proxy.pt")
    layers_to_quant = torch.load("./cached_files/layers_to_quant.pt")
    layer_size = torch.load("./cached_files/layer_size.pt")
    layer_bitops = torch.load("./cached_files/layer_bitops.pt")

    proxy_matrix, _ = clado_proxy_post_process(
        proxy=proxy,
        layers_to_quant=layers_to_quant,
        format_space=format_space,
    )

    size_bounds = np.append(np.linspace(20.5, 35, 10), np.linspace(35.5, 45, 20))
    size_bounds = np.append(size_bounds, np.linspace(45.5, 81.5, 10))
    n_samples = 1024
    seed = 43
    
    clado_res, clado_objective = [], []
    for size_bound in size_bounds:
        print(f'Set size bound to {size_bound} MB')
        # clado
        v1, objective_val = qip_optimizer(
                                            cached_proxy=proxy_matrix,
                                            layer_bitops=layer_bitops,
                                            layer_size=layer_size,
                                            schemes_per_layer=3,
                                            bitops_bound=np.inf,
                                            size_bound=size_bound,
                                            diagonal=False,
                                            printInfo=True
                                        )
        clado_res.append(v1)
        clado_objective.append(objective_val)

    with open(f'./clado_mpqco_results/CLADO/Clado_optimal_decisions_ViT/sample_size{n_samples}/clado_8_w8-4-2_seed{seed}_optimization.pkl','wb') as f:
        pickle.dump({'clado_res': clado_res}, f)
    with open(f'./clado_mpqco_results/CLADO/Clado_optimal_objectives_ViT/sample_size{n_samples}/clado_a8_w8-4-2_seed{seed}_optimization_objective.pkl','wb') as f:
        pickle.dump({'clado_objectives': clado_objective}, f)  

def main():
    feintLady()


if __name__ == "__main__":
    main()
