DATA_PATH=~/Shares1/wanghuijuan03/wikikg90m/  # Path to wikikg90m-v2
CANDIDATE_PATH=$DATA_PATH/val_hrt_cand_2w.npy  # Path to dev candidate file
TEST_CANDIDATE_PATH=$DATA_PATH/val_hrt_cand_2w.npy  # Path to test candidate file


# TransE
CUDA_VISIBLE_DEVICES=6 python train.py \
    --model_name TransE \
    --data_path $DATA_PATH \
    --candidate_path $CANDIDATE_PATH \
    --test_candidate_path $TEST_CANDIDATE_PATH \
    --data_name wikikg90mv2 \
    --embed_dim 200 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 16 \
    --lr 2e-4 --cpu_lr 0.15 \
    --optimizer adam --valid_percent 0.2 \
    --max_steps 2000000 \
    --batch_size 2000 --neg_sample_size 2000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 16 \
    --print_on_screen --save_path ~/Shares1/wanghuijuan03/wikikg90m/checkpoints_shallow/ \
    --test --valid --eval_interval 20000 --save_interval 20000

    # --async_update --use_feature \
    --init_from_ckpt ~/Shares1/wanghuijuan03/wikikg90m/checkpoints/transe_wikikg90mv2_d_200_g_8.0_e_cpu_r_gpu_l_Logsigmoid_lr_2e-05_0.15_KGE/ \
#python -m paddle.distributed.launch --gpus 1,2,6,7 dist_train.py \

# RotatE
    --model_name RotatE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 100 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.15 \
    --optimizer adam --valid_percent 0.1 \
    --max_steps 2000000 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_rotate/ \
    --test --valid --eval_interval 20000

# OTE
# python train.py \
    --model_name OTE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 200 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.15 \
    --optimizer adam --valid_percent 0.1 \
    --max_steps 2000000 \
    --ote_size 20 --ote_scale 2 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_ote/ \
    --test --valid --valid_percent 0.1 --eval_interval 20000
