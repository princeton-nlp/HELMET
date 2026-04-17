from dataclasses import dataclass, field
from typing import Callable, Any

from torch.utils.data import Dataset

from utils import calculate_metrics, parse_output

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


LENGTH_MAP = {
    "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
    "64k": 65536, "128k": 131072, "256k": 262144, "512k": 524288,
    "1m": 1048576, "2m": 2097152,
}


@dataclass
class TaskConfig:
    """Static configuration for a registered task."""
    loader: Callable
    input_max_length: int = 131072
    test_file: str | None = None
    demo_file: str | None = None
    generation_max_length: int = 100
    use_chat_template: bool = False
    stop_new_line: bool = False
    shots: int = 0
    primary_metric: str = "substring_exact_match"


# Global registry: exact dataset name -> TaskConfig
TASK_REGISTRY: dict[str, TaskConfig] = {}


def register_task(name: str, **kwargs):
    """Register a task configuration by exact name."""
    TASK_REGISTRY[name] = TaskConfig(**kwargs)


def default_post_process(output, example, primary_metric="substring_exact_match"):
    """
    Default post-processing: parse the output and compute EM/F1/ROUGE metrics.

    Args:
        output: dict with at least:
            - "output" (str): the raw generated text from the model
        example: dict with at least:
            - "answer" (str | list[str]): the ground-truth answer(s)
        primary_metric: which metric key to copy into "primary_metric"

    Returns:
        metrics: dict[str, float] — metric name to value (e.g. {"exact_match": 1.0, "f1": 0.85, ...})
        extras: dict[str, Any] — additional info to store alongside the result (e.g. {"parsed_output": ...})
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer)
    parsed_pred = parse_output(prediction)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    mets["primary_metric"] = mets[primary_metric]
    return mets, {"parsed_output": parsed_pred}


def load_data(data_args, dataset: str):
    """
    Load data using the task registry.

    Args:
        data_args: object with attributes max_test_samples, shots, seed
                   (compatible with DatasetOptions from arguments.py)
        dataset: exact dataset name registered in TASK_REGISTRY

    Returns:
        dict with keys:
            - "data" (datasets.Dataset): the test samples, each row is a dict
            - "prompt_template" (str): full prompt = user_template + system_template, this is useful for base models without chat templates
            - "user_template" (str): the user/instruction portion, with {context}, {question}, etc. placeholders
            - "system_template" (str): the assistant prefix to prepend to output (e.g. "Answer:")
            - "post_process" (Callable): function(output, example) -> (metrics dict, extras dict)
            - "primary_metric" (str): the metric name to use for reporting (e.g. "exact_match")
        Optional keys set by some loaders:
            - "is_chat" (bool): if True, inputs are already in chat message format
    """
    if dataset not in TASK_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(TASK_REGISTRY.keys())}")

    task_config = TASK_REGISTRY[dataset]
    data = task_config.loader(dataset=dataset, task_config=task_config, data_args=data_args)
    if "post_process" not in data:
        pm = task_config.primary_metric
        data["post_process"] = lambda output, example, _pm=pm: default_post_process(output, example, primary_metric=_pm)
    data.setdefault("primary_metric", task_config.primary_metric)
    return data


class TestItemDataset(Dataset):
    """
    data is a dictionary that should contain the "data" field, which is a list of samples
    llm is of type LLM from model_utils
    tokenizer is any callable tokenizer with decode method, but not necessary
    """
    def __init__(self, data: dict[str, Any], llm, tokenizer=None):
        self.data = data
        self.llm = llm
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        inputs = self.llm.prepare_inputs(self.data["data"][idx], self.data)
        original_text = None
        if "original_text" in inputs:
            original_text = inputs["original_text"]
        elif "input_ids" in inputs:
            original_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return inputs, original_text
