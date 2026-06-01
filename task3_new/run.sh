source /home/ljh_xzp/test/.venv/bin/activate
cd /home/ljh_xzp/test/task3

torchrun --standalone --nproc_per_node=8 run_task3_lm.py \
  --ddp auto --amp auto \
  --max_minutes 30 --time_check_interval 20 \
  --eval_every_epochs 20 --eval_iters 200 \
  --epochs 999999 --steps_per_epoch 0 \
  --batch_size 512 --seq_len 256 \
  --d_model 384 --nhead 8 --num_layers 6 --d_ff 1536 \
  --val_ratio 0.05 --test_ratio 0.05 \
  --out_dir outputs_lm_30min_new \
  --num_workers 128