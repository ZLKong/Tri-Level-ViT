
data_path="/home/zlkong/Documents/ImageNet"
# data_path="/home/haoyum/imagenet/ILSVRC/Data/CLS-LOC/"



# Deit-Small + Sparse Token Score + 0.5 Sparsity
# save_path="exp-deit-small-imgt1K-score-05"
# mkdir -p $save_path


# CUDA_VISIBLE_DEVICES="0,1,2,3" \
# nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=1112 \
#         --use_env main.py \
#         --model vit_small_patch16_224_token_pruning \
#         --batch-size 256 \
#         --data-path ${data_path} \
#         --sparsity-token 0.5 \
#         --output_dir ${save_path} > full_log.txt 2>&1 &

# Deit-Small + Attn-Level 50% + Token-Level 0.7
# save_path="exp-deit-small-100-Attn-Token"
# mkdir -p $save_path
# CUDA_VISIBLE_DEVICES="0" \
# python -m torch.distributed.launch --nproc_per_node=1 \
#         --use_env main.py \
#         --model deit_small_patch16_224_attn_dst \
#         --batch-size 384 \
#         --data-path ${data_path} \
#         --keep_ratio 0.7 \
#         --output_dir ${save_path}


# Attention Only
save_path="exp-deit-small-100-Attn-Token"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3" \
nohup python -m torch.distributed.launch --nproc_per_node=4 \
        --use_env main_wo_ep.py \
        --model deit_small_patch16_224_attn_dst \
        --batch-size 384 \
        --data-path ${data_path} \
        --keep_ratio 1.0 \
        --attn_ratio 0.2 \
        --output_dir ${save_path} > full_log.txt 2>&1 &
