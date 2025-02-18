shards=10; for i in $(seq 0 $shards); do python scripts/eval_gpt4_summ.py --num_shards $shards --shard_idx $i & done
