import re

from datasets import load_dataset

from data_module.base import register_task, LENGTH_MAP

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parse_list(response: str):
    """Parse a list from the model response, looking for 'Final Answer: [...]' or bare '[...]'."""
    list_part = re.search(r"final answer: ?\[([^\]]*)\]?", response, re.IGNORECASE)
    if not list_part:
        list_part = re.search(r"\[([^\]]*)\]?", response)
    if list_part:
        result_list = list_part.group(1).split(",")
        return [item.strip() for item in result_list if item.strip()]
    return []


def load_graphwalk(dataset: str, task_config, data_args):
    """Load and preprocess a GraphWalk dataset (BFS or parent traversal)."""
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)

    data = load_dataset("json", data_files=task_config.test_file)["train"]

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(max_test_samples, len(data))))

    user_template = "{context}\n\n{question}"
    system_template = "Final Answer:"
    prompt_template = user_template + "\n\n" + system_template

    def post_process(output, example):
        prediction = _parse_list(output["output"])
        answer = example["answer"]

        if len(answer) == 0:
            f1 = 1.0 if len(prediction) == 0 else 0.0
        else:
            n_overlap = len(set(prediction) & set(answer))
            recall = n_overlap / len(answer)
            precision = n_overlap / len(prediction) if len(prediction) > 0 else 0.0
            f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0.0

        return {"f1": f1, "primary_metric": f1}, {"parsed_output": prediction}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register GraphWalk variants (4k through 1m)
# ---------------------------------------------------------------------------

_LONG_LENGTHS = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1m", "2m"]
_K_VALUES = ["250", "500", "1000", "2100", "4400", "8800", "17600", "35200", "70400", "140800"]

# graphwalk_bfs, default depth = 10
for _length, _k in zip(_LONG_LENGTHS, _K_VALUES):
    register_task(
        f"graphwalk_bfs_{_length}",
        loader=load_graphwalk,
        input_max_length=LENGTH_MAP[_length],
        test_file=f"data/graphwalk/bfs_k{_k}.jsonl",
        generation_max_length=1000,
        shots=0,
        primary_metric="f1",
    )

# graphwalk_bfs_dep8, shorter depth = shorter generation length = faster
for _length, _k in zip(_LONG_LENGTHS, _K_VALUES):
    register_task(
        f"graphwalk_bfs_dep8_{_length}",
        loader=load_graphwalk,
        input_max_length=LENGTH_MAP[_length],
        test_file=f"data/graphwalk/bfs_k{_k}_dep8.jsonl",
        generation_max_length=500,
        shots=0,
        primary_metric="f1",
    )

# graphwalk_bfs_dep5
for _length, _k in zip(_LONG_LENGTHS, _K_VALUES):
    register_task(
        f"graphwalk_bfs_dep5_{_length}",
        loader=load_graphwalk,
        input_max_length=LENGTH_MAP[_length],
        test_file=f"data/graphwalk/bfs_k{_k}_dep5.jsonl",
        generation_max_length=200,
        shots=0,
        primary_metric="f1",
    )

# graphwalk_parent
for _length, _k in zip(_LONG_LENGTHS, _K_VALUES):
    register_task(
        f"graphwalk_parent_{_length}",
        loader=load_graphwalk,
        input_max_length=LENGTH_MAP[_length],
        test_file=f"data/graphwalk/parent_k{_k}.jsonl",
        generation_max_length=100,
        shots=0,
        primary_metric="f1",
    )
