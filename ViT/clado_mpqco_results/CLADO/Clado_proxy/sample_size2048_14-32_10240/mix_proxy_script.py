import torch

#for layers 14 and 32 that have high variance, pick the proxy values from 10K sample size
lnames = ['vit.encoder.layer.0.intermediate.dense', 'vit.encoder.layer.1.intermediate.dense']
proxy_matrix_10k = torch.load(f"../sample_size10240/clado_proxy_run0.pt")

n_runs = 5
for run in range(n_runs):
    proxy_matrix = torch.load(f"../sample_size2048/clado_proxy_run{run}.pt")
    for lconfig in proxy_matrix.keys():
        layer_i, layer_j, q_width_i, q_width_j = lconfig
        # ('vit.encoder.layer.0.intermediate.dense', (A:XP[8,0](CSN), W:XP[2,0](CSN)))
        # or ('vit.encoder.layer.1.intermediate.dense', (A:XP[8,0](CSN), W:XP[2,0](CSN)))
        if (layer_i in lnames and q_width_i == 2) or (layer_j in lnames and q_width_j == 2):
            proxy_matrix[lconfig] = proxy_matrix_10k[lconfig]
    torch.save(proxy_matrix, f"clado_proxy_run{run}.pt")