import torch
# p0 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run0.pt")
# p1 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run1.pt")
# p2 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run2.pt")
# p3 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run3.pt")
# p4 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run4.pt")
# p = torch.load("/homes/xwang/results/mpq/vit_base_patch16_224/clado_proxy.pt")

p5 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run0.pt")
p6 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run1.pt")
p7 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run2.pt")
p8 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run3.pt")
p9 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run4.pt")

p_4k = {}
p_4k_2 = {}
p_6k = {}
p_8k = {}
# res = []
# for lconfig in p:
#     res.append(torch.equal(p[lconfig][1024*0:1024*1].mean(), p0[lconfig]))
#     res.append(torch.equal(p[lconfig][1024*1:1024*2].mean(), p1[lconfig]))
#     res.append(torch.equal(p[lconfig][1024*2:1024*3].mean(), p2[lconfig]))
#     res.append(torch.equal(p[lconfig][1024*3:1024*4].mean(), p3[lconfig]))
#     res.append(torch.equal(p[lconfig][1024*4:1024*5].mean(), p4[lconfig]))
#     res.append(torch.equal(p[lconfig][2048*0:2048*1].mean(), p5[lconfig]))
#     res.append(torch.equal(p[lconfig][2048*1:2048*2].mean(), p6[lconfig]))
#     res.append(torch.equal(p[lconfig][2048*2:2048*3].mean(), p7[lconfig]))
#     res.append(torch.equal(p[lconfig][2048*3:2048*4].mean(), p8[lconfig]))
#     res.append(torch.equal(p[lconfig][2048*4:2048*5].mean(), p9[lconfig]))
for lconfig in p5:
    p_4k[lconfig] = (p5[lconfig] + p6[lconfig] ) / 2
    p_4k_2[lconfig] = (p7[lconfig] + p8[lconfig] ) / 2
    p_6k[lconfig] = (p5[lconfig] + p6[lconfig] + p7[lconfig]) / 3
    p_8k[lconfig] = (p5[lconfig] + p6[lconfig] + p7[lconfig] + p8[lconfig] ) / 4

torch.save(p_4k, "/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size4096/clado_proxy_run0.pt")
torch.save(p_4k_2, "/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size4096/clado_proxy_run1.pt")
torch.save(p_6k, "/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size6144/clado_proxy_run0.pt")
torch.save(p_8k, "/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size8192/clado_proxy_run0.pt")
breakpoint()