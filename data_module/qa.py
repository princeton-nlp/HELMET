import math
import random
import hashlib

from datasets import load_dataset

from data_module.base import register_task, LENGTH_MAP

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _drop_duplicates(data, key):
    """Keep only the first occurrence of each key value."""
    seen = set()
    indices = []
    for i, d in enumerate(data):
        if d[key] not in seen:
            seen.add(d[key])
            indices.append(i)
    return data.select(indices)


def load_qa(dataset: str, task_config, data_args):
    """Load and preprocess a QA dataset (NQ, TriviaQA, HotpotQA, PopQA).

    Each sample has retrieved passages in 'ctxs' and ground-truth answers in 'answers'.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)
    shots = task_config.shots

    user_template = "Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    data = load_dataset("json", data_files=task_config.test_file)["train"]
    demo_data = load_dataset("json", data_files=task_config.demo_file)["train"]

    # popqa: filter by popularity and exclude test questions from demos
    is_popqa = "popqa" in dataset
    if is_popqa:
        data = data.filter(lambda x: math.log10(x['s_pop']) < 3)
        demo_data = demo_data.filter(lambda x: math.log10(x['s_pop']) < 3)

    key = "id" if "id" in data.column_names else "question"

    if max_test_samples is not None:
        keys = set(data[key])
        keys = random.Random(seed).sample(sorted(keys), min(max_test_samples, len(keys)))
        data = data.filter(lambda x: x[key] in keys)

    # for popqa, demo and test share the same split, so pre-filter demos
    if is_popqa:
        test_keys = set(data[key])
        demo_data = demo_data.filter(lambda x: x[key] not in test_keys)

    passage_template = "Document (Title: {title}): {text}"
    demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"

    def update(sample):
        demo_text = ""
        if shots > 0:
            h = int(hashlib.sha256(str(sample[key]).encode("utf-8")).hexdigest(), 16) % 2**31
            demos = demo_data.shuffle(seed=h)
            demos = _drop_duplicates(demos, key).select(range(shots))
            demo_text = "\n\n".join([
                demo_template.format(
                    **d,
                    documents="\n\n".join([passage_template.format(**c) for c in d["ctxs"]]),
                    answer=d["answers"][0],
                ) for d in demos
            ]) + "\n\n"

        passage_text = ""
        if len(sample['ctxs']) > 0:
            passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])

        return {"demos": demo_text, "context": passage_text, "answer": sample["answers"]}

    data = data.map(update)

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


# ---------------------------------------------------------------------------
# Register QA variants (4k through 128k, llama2 tokenizer lengths)
# ---------------------------------------------------------------------------

_LENGTHS = ["4k", "8k", "16k", "32k", "64k", "128k"]
_K_VALUES = ["20", "50", "105", "220", "440", "1000"]

_QA_DATASETS = {
    "kilt_nq": {
        "test_template": "data/kilt/nq-dev-multikilt_1000_k{k}_dep6.jsonl",
        "demo_file": "data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
    },
    "kilt_triviaqa": {
        "test_template": "data/kilt/triviaqa-dev-multikilt_1000_k{k}_dep6.jsonl",
        "demo_file": "data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
    },
    "kilt_hotpotqa": {
        "test_template": "data/kilt/hotpotqa-dev-multikilt_1000_k{k}_dep3.jsonl",
        "demo_file": "data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
    },
    "kilt_popqa": {
        "test_template": "data/kilt/popqa_test_1000_k{k}_dep6.jsonl",
        "demo_file": "data/kilt/popqa_test_1000_k3_dep6.jsonl",
    },
}

for _base, _config in _QA_DATASETS.items():
    for _length, _k in zip(_LENGTHS, _K_VALUES):
        register_task(
            f"{_base}_{_length}",
            loader=load_qa,
            input_max_length=LENGTH_MAP[_length],
            test_file=_config["test_template"].format(k=_k),
            demo_file=_config["demo_file"],
            generation_max_length=20,
            stop_new_line=True,
            shots=2,
            primary_metric="substring_exact_match",
        )
