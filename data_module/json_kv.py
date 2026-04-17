from datasets import load_dataset, load_from_disk

from utils import calculate_metrics, parse_output
from data_module.base import register_task, LENGTH_MAP

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_json_kv(dataset: str, task_config, data_args):
    """Load and preprocess a JSON key-value retrieval dataset.

    The task presents a large JSON object and asks the model to extract the
    value for a specified key.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)
    shots = task_config.shots

    user_template = "{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n{demos}Key: {question}"
    system_template = "Corresponding value:"
    prompt_template = user_template + "\n" + system_template

    path = task_config.test_file
    if path.endswith(".json"):
        data = load_dataset("json", data_files=path, field="data")["train"]
    elif path.endswith(".jsonl"):
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = load_from_disk(path)
        return {"data": data, "prompt_template": prompt_template, "user_template": user_template, "system_template": system_template}

    demo_template = "Key: {key}\nCorresponding value:{value}"
    data = data.map(lambda x: {
        "demos": "\n\n".join([demo_template.format(key=key, value=" "+value) for key, value in x["demos"][:shots]]) + ("\n\n" if shots > 0 else ""),
        "k": x["num_kvs"],
    })

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(max_test_samples, len(data))))

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        mets = calculate_metrics(prediction, answer)
        mets["primary_metric"] = mets["substring_exact_match"]
        return mets, {}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register json_kv variants keyed by the test file (k value)
# ---------------------------------------------------------------------------

_JSON_KV_VARIANTS = {
    # llama2 tokenizer variants (4k through 1m). this was used in the original HELMET paper
    "json_kv_4k_llama2":   ("data/json_kv/test_k50_dep6.jsonl",   "4k"),
    "json_kv_8k_llama2":   ("data/json_kv/test_k105_dep6.jsonl",  "8k"),
    "json_kv_16k_llama2":  ("data/json_kv/test_k220_dep6.jsonl",  "16k"),
    "json_kv_32k_llama2":  ("data/json_kv/test_k440_dep6.jsonl",  "32k"),
    "json_kv_64k_llama2":  ("data/json_kv/test_k900_dep6.jsonl",  "64k"),
    "json_kv_128k_llama2": ("data/json_kv/test_k1800_dep6.jsonl", "128k"),
    "json_kv_256k_llama2": ("data/json_kv/test_k3600_dep6.jsonl", "256k"),
    "json_kv_512k_llama2": ("data/json_kv/test_k7200_dep6.jsonl", "512k"),
    "json_kv_1m_llama2":   ("data/json_kv/test_k14400_dep6.jsonl","1m"),

    # llama3 tokenizer variants (32k through 2m)
    "json_kv_32k":  ("data/json_kv/test_k630_dep6.jsonl",   "32k"),
    "json_kv_64k":  ("data/json_kv/test_k1260_dep6.jsonl",  "64k"),
    "json_kv_128k": ("data/json_kv/test_k2550_dep6.jsonl",  "128k"),
    "json_kv_256k": ("data/json_kv/test_k5125_dep6.jsonl",  "256k"),
    "json_kv_512k": ("data/json_kv/test_k10250_dep6.jsonl", "512k"),
    "json_kv_1m":   ("data/json_kv/test_k20500_dep6.jsonl", "1m"),
    "json_kv_2m":   ("data/json_kv/test_k41000_dep6.jsonl", "2m"),
}

for _name, (_path, _length) in _JSON_KV_VARIANTS.items():
    register_task(
        _name,
        loader=load_json_kv,
        input_max_length=LENGTH_MAP[_length],
        test_file=_path,
        generation_max_length=100,
        stop_new_line=False,
        shots=2,
        primary_metric="substring_exact_match",
    )
