mkdir Ltilde_resnet50
mkdir DELTAL_resnet50
# Goal: get 300 batches Ltilde

# if on T4 : one run per gpu, roughly 1batch/h
# python3 CLADO_MPQCO_r50.py --start-batch 0 --end-batch 19 --nthreads 8 &> batch0_19.log
# if on v100 : two run per gpu (because GPU util is low for one run) roughly 2batches/h
# python3 CLADO_MPQCO_r50.py --start-batch 0 --end-batch 19 --nthreads 12 &> batch0_19.log &
# python3 CLADO_MPQCO_r50.py --start-batch 20 --end-batch 39 --nthreads 12 &> batch20_39.log &

# if on 4 x v100: roughly 8batches/h
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 288 --end-batch 290 --nthreads 8 &> batch290.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 221 --end-batch 230 --nthreads 8 &> batch221.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 231 --end-batch 240 --nthreads 8 &> batch231.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 241 --end-batch 250 --nthreads 8 &> batch241.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 251 --end-batch 260 --nthreads 8 &> batch251.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 261 --end-batch 270 --nthreads 8 &> batch261.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 271 --end-batch 280 --nthreads 8 &> batch271.log 
# CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 281 --end-batch 290 --nthreads 8 &> batch281.log 
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 291 --end-batch 300 --nthreads 8 &> batch300.log 
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 301 --end-batch 310 --nthreads 8 &> batch310.log
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 311 --end-batch 320 --nthreads 8 &> batch320.log 
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 321 --end-batch 330 --nthreads 8 &> batch330.log 
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 331 --end-batch 340 --nthreads 8 &> batch340.log 
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 341 --end-batch 350 --nthreads 8 &> batch350.log 
#CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 20 --end-batch 39 --nthreads 12 &> batch20_39.log &
#CUDA_VISIBLE_DEVICE=1 python3 CLADO_MPQCO_r50.py --start-batch 40 --end-batch 59 --nthreads 12 &> batch40_59.log &
#CUDA_VISIBLE_DEVICE=1 python3 CLADO_MPQCO_r50.py --start-batch 60 --end-batch 79 --nthreads 12 &> batch60_79.log &
#CUDA_VISIBLE_DEVICE=2 python3 CLADO_MPQCO_r50.py --start-batch 80 --end-batch 99 --nthreads 12 &> batch80_99.log &
#CUDA_VISIBLE_DEVICE=2 python3 CLADO_MPQCO_r50.py --start-batch 100 --end-batch 119 --nthreads 12 &> batch100_119.log &
#CUDA_VISIBLE_DEVICE=3 python3 CLADO_MPQCO_r50.py --start-batch 120 --end-batch 139 --nthreads 12 &> batch120_139.log &
#CUDA_VISIBLE_DEVICE=3 python3 CLADO_MPQCO_r50.py --start-batch 140 --end-batch 159 --nthreads 12 &> batch140_159.log &
