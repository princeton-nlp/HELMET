import json
from difflib import SequenceMatcher

import tiktoken
from datasets import load_dataset

from data_module.base import register_task

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_MRCR_LENGTHS = {
    "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536,
    "128k": 131072, "256k": 262144, "512k": 524288, "1m": 1048576,
}


def load_mrcr(dataset: str, task_config, data_args):
    """Load and preprocess the MRCR (Multi-Round Coreference Resolution) dataset.

    https://huggingface.co/datasets/openai/mrcr
    MRCR is already formatted as chat conversations, so we return is_chat=True
    instead of prompt templates.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)

    # Parse n_needles and length from dataset name (e.g. mrcr_8_128k)
    parts = dataset.split("_")
    n_needles = int(parts[1])
    length_suffix = parts[2]
    length = _MRCR_LENGTHS[length_suffix]

    enc = tiktoken.get_encoding("o200k_base")

    def n_tokens(example):
        return sum(len(enc.encode(m["content"])) for m in example["prompt"]) + len(enc.encode(example["answer"]))

    data = load_dataset("openai/mrcr")["train"]

    # Filter by needle count and token length range
    data = data.filter(lambda x: x["n_needles"] == n_needles)
    data = data.map(lambda example: {"prompt": json.loads(example["prompt"])})
    data = data.filter(lambda x: length / 2 < n_tokens(x) <= length)

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    def _grade(response, answer, random_string_to_prepend):
        if not response.startswith(random_string_to_prepend):
            return 0.0
        response = response.removeprefix(random_string_to_prepend)
        answer = answer.removeprefix(random_string_to_prepend)
        return float(SequenceMatcher(None, response, answer).ratio())

    def post_process(output, example):
        prediction = output["output"]
        score = _grade(prediction, example["answer"], example["random_string_to_prepend"])
        return {"score": score, "primary_metric": score}, {"parsed_output": prediction}

    return {
        "data": data,
        "system_template": "",
        "is_chat": True,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register MRCR variants (mrcr_4 and mrcr_8, 8k through 1m)
# ---------------------------------------------------------------------------

for _n_needles in ["4", "8"]:
    for _length_name, _length_val in _MRCR_LENGTHS.items():
        register_task(
            f"mrcr_{_n_needles}_{_length_name}",
            loader=load_mrcr,
            input_max_length=_length_val,
            generation_max_length=1000,
            use_chat_template=True,
            shots=0,
            primary_metric="score",
        )
