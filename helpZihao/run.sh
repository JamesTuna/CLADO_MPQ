mkdir Ltilde_resnet50
mkdir DELTAL_resnet50
# Goal: get 300 batches Ltilde

# if on T4 : one run per gpu, roughly 1batch/h
# python3 CLADO_MPQCO_r50.py --start-batch 0 --end-batch 19 --nthreads 8 &> batch0_19.log
# if on v100 : two run per gpu (because GPU util is low for one run) roughly 2batches/h
# python3 CLADO_MPQCO_r50.py --start-batch 0 --end-batch 19 --nthreads 12 &> batch0_19.log &
# python3 CLADO_MPQCO_r50.py --start-batch 20 --end-batch 39 --nthreads 12 &> batch20_39.log &

# if on 4 x v100: roughly 8batches/h
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 0 --end-batch 19 --nthreads 12 &> batch0_19.log &
CUDA_VISIBLE_DEVICE=0 python3 CLADO_MPQCO_r50.py --start-batch 20 --end-batch 39 --nthreads 12 &> batch20_39.log &
CUDA_VISIBLE_DEVICE=1 python3 CLADO_MPQCO_r50.py --start-batch 40 --end-batch 59 --nthreads 12 &> batch40_59.log &
CUDA_VISIBLE_DEVICE=1 python3 CLADO_MPQCO_r50.py --start-batch 60 --end-batch 79 --nthreads 12 &> batch60_79.log &
CUDA_VISIBLE_DEVICE=2 python3 CLADO_MPQCO_r50.py --start-batch 80 --end-batch 99 --nthreads 12 &> batch80_99.log &
CUDA_VISIBLE_DEVICE=2 python3 CLADO_MPQCO_r50.py --start-batch 100 --end-batch 119 --nthreads 12 &> batch100_119.log &
CUDA_VISIBLE_DEVICE=3 python3 CLADO_MPQCO_r50.py --start-batch 120 --end-batch 139 --nthreads 12 &> batch120_139.log &
CUDA_VISIBLE_DEVICE=3 python3 CLADO_MPQCO_r50.py --start-batch 140 --end-batch 159 --nthreads 12 &> batch140_159.log &
