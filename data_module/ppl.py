from datasets import load_dataset
from transformers import AutoTokenizer

from data_module.base import register_task

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maps length suffix to (token count, stride)
_PPL_VARIANTS = {
    "32k":  (32768,   8192),
    "64k":  (65536,   8192),
    "128k": (131072,  8192),
    "256k": (262144,  8192),
    "512k": (524288,  8192),
    "1m":   (1048576, 8192),
}

_TEST_FILE = "data/longmino/524288_10.jsonl"


def load_ppl(dataset: str, task_config, data_args):
    """Load and preprocess data for perplexity evaluation.

    Chunks documents into sliding windows of (input_ids, labels) using
    the model's own tokenizer. Each chunk has length tokens with a stride
    for evaluation.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)
    tokenizer_name = getattr(data_args, "tokenizer", None)
    if tokenizer_name is None:
        raise ValueError("data_args.tokenizer must be set for PPL evaluation (usually the model name)")

    # Parse length and stride from dataset name (e.g. ppl_longmino_32768_8192)
    parts = dataset.split("_")
    length = int(parts[-2])
    stride = int(parts[-1])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("json", data_files=task_config.test_file)["train"]

    def preprocess_example(examples):
        output = []
        all_input_ids = tokenizer(examples['context'], add_special_tokens=True)['input_ids']
        for idx in range(len(all_input_ids)):
            input_ids = all_input_ids[idx]
            if len(input_ids) < length:
                continue

            for i in range(0, len(input_ids) - length + 1, stride):
                chunk = {
                    "input_ids": input_ids[i:i + length],
                    "labels": input_ids[i + 1:i + length + 1],
                    "domain": examples['domain'][idx],
                    'created': examples['created'][idx],
                    'length': len(input_ids),
                    "stride": stride,
                }
                # handle the case where the labels do not match the input_ids because we reach the end of the text
                if i + length + 1 > len(input_ids):
                    # -100 is ignored in loss calculation
                    chunk['labels'].append(-100)
                assert len(chunk['input_ids']) == len(chunk['labels'])
                output.append(chunk)

        return {
            "input_ids": [o['input_ids'] for o in output],
            "labels": [o['labels'] for o in output],
            "domain": [o['domain'] for o in output],
            "created": [o['created'] for o in output],
            "stride": [o['stride'] for o in output],
            "length": [o['length'] for o in output],
        }

    ds = ds.map(preprocess_example, batched=True, remove_columns=ds.column_names)
    logger.info(f"Loaded {len(ds)} samples from {dataset}")

    if max_test_samples is not None and max_test_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_test_samples))

    # PPL evaluation is handled by eval_ppl.py, not the standard post_process pipeline.
    # Return a dummy post_process that won't be called during normal PPL eval.
    def post_process(output, example):
        mets = {"perplexity": 0.0, "primary_metric": 0.0}
        return mets, {}

    return {
        "data": ds,
        "prompt_template": "{context}",
        "user_template": "",
        "system_template": "",
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register PPL variants
# ---------------------------------------------------------------------------

for _length_name, (_length, _stride) in _PPL_VARIANTS.items():
    register_task(
        f"ppl_longmino_{_length}_{_stride}",
        loader=load_ppl,
        input_max_length=_length,
        test_file=_TEST_FILE,
        generation_max_length=0,
        use_chat_template=False,
        primary_metric="perplexity",
    )
