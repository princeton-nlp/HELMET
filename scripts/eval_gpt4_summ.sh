for i in {0..15}; do python scripts/eval_gpt4_summ.py --num_shards 16 --shard_idx $i & done
