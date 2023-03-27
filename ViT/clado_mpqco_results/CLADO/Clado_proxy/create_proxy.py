import torch
p0 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run0.pt")
p1 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run1.pt")
p2 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run2.pt")
p3 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run3.pt")
p4 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size1024/clado_proxy_run4.pt")
p = torch.load("/homes/xwang/results/mpq/vit_base_patch16_224/clado_proxy.pt")

p5 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run0.pt")
p6 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run1.pt")
p7 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run2.pt")
p8 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run3.pt")
p9 = torch.load("/homes/sayehs/CLADO_MPQ/ViT/clado_mpqco_results/CLADO/Clado_proxy/sample_size2048/clado_proxy_run4.pt")


res = []
for lconfig in p:
    res.append(torch.equal(p[lconfig][1024*0:1024*1].mean(), p0[lconfig]))
    res.append(torch.equal(p[lconfig][1024*1:1024*2].mean(), p1[lconfig]))
    res.append(torch.equal(p[lconfig][1024*2:1024*3].mean(), p2[lconfig]))
    res.append(torch.equal(p[lconfig][1024*3:1024*4].mean(), p3[lconfig]))
    res.append(torch.equal(p[lconfig][1024*4:1024*5].mean(), p4[lconfig]))
    res.append(torch.equal(p[lconfig][2048*0:2048*1].mean(), p5[lconfig]))
    res.append(torch.equal(p[lconfig][2048*1:2048*2].mean(), p6[lconfig]))
    res.append(torch.equal(p[lconfig][2048*2:2048*3].mean(), p7[lconfig]))
    res.append(torch.equal(p[lconfig][2048*3:2048*4].mean(), p8[lconfig]))
    res.append(torch.equal(p[lconfig][2048*4:2048*5].mean(), p9[lconfig]))
breakpoint()