DATA_PATH=./data/nfs/

# TransE
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name TransE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 200 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.1 \
    --optimizer adam --cpu_optimizer adagrad \
    --max_steps 2000000 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_transe_1202/ \
    --valid --eval_interval 50000 --valid_percent 0.01

# RotatE
# python train.py \
    --model_name RotatE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 100 --gamma 8. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.1 \
    --optimizer adam --cpu_optimizer adagrad \
    --max_steps 2000000 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_rotate/ \
    --test --valid --eval_interval 50000 --valid_percent 0.01

# OTE
# python train.py \
    --model_name OTE \
    --data_path $DATA_PATH \
    --data_name wikikg90m \
    --embed_dim 200 --gamma 12. --reg_coef 1e-9 \
    -adv --mix_cpu_gpu --num_workers 4 \
    --lr 0.00002 --cpu_lr 0.1 \
    --optimizer adam --cpu_optimizer adagrad \
    --max_steps 4000000 \
    --ote_size 40 --ote_scale 0 \
    --batch_size 1000 --neg_sample_size 1000 \
    --neg_sample_type 'chunk' \
    --log_interval 1000 --num_process 1 \
    --async_update --use_feature \
    --print_on_screen --save_path output/wikikg90m_ote/ \
    --test --valid --valid_percent 0.01 --eval_interval 50000
