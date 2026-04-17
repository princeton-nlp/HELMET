import math
import random
import hashlib
from dataclasses import dataclass

import datasets
from datasets import load_dataset

from utils import parse_output, normalize_answer
from data_module.base import register_task, LENGTH_MAP

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ICLDatasetConfig:
    """HuggingFace dataset metadata for an ICL task."""
    hf_name: str
    text_field: str
    label_field: str
    num_labels: int
    train_split: str = "train"
    test_split: str = "test"
    hf_subset: str | None = None


ICL_DATASETS: dict[str, ICLDatasetConfig] = {
    "icl_trec_coarse": ICLDatasetConfig(
        hf_name="CogComp/trec",
        text_field="text",
        label_field="coarse_label",
        num_labels=6,
    ),
    "icl_trec_fine": ICLDatasetConfig(
        hf_name="CogComp/trec",
        text_field="text",
        label_field="fine_label",
        num_labels=50,
    ),
    "icl_banking77": ICLDatasetConfig(
        hf_name="PolyAI/banking77",
        text_field="text",
        label_field="label",
        num_labels=77,
    ),
    "icl_clinic150": ICLDatasetConfig(
        hf_name="clinc/clinc_oos",
        hf_subset="plus",
        text_field="text",
        label_field="intent",
        num_labels=151,
        test_split="validation",
    ),
    "icl_nlu": ICLDatasetConfig(
        hf_name="xingkunliuxtracta/nlu_evaluation_data",
        text_field="text",
        label_field="label",
        num_labels=68,
        test_split=None,  # uses train_test_split
    ),
}


def _get_base_icl_name(dataset: str) -> str:
    """Extract the base ICL name from a full variant name.

    e.g. 'icl_trec_coarse_128k' -> 'icl_trec_coarse'
    """
    # match longest prefix first to avoid 'icl_trec_coarse' matching before 'icl_trec_coarse_fine'
    matches = [name for name in ICL_DATASETS if dataset.startswith(name)]
    if not matches:
        raise ValueError(f"Unknown ICL dataset: {dataset}")
    return max(matches, key=len)


def _load_hf_data(icl_config: ICLDatasetConfig, seed: int = 42):
    """Load train/test splits from HuggingFace for an ICL dataset."""
    hf_args = [icl_config.hf_name]
    if icl_config.hf_subset is not None:
        hf_args.append(icl_config.hf_subset)

    if icl_config.test_split is None:
        # dataset needs manual splitting (e.g., NLU)
        all_data = load_dataset(*hf_args, trust_remote_code=True)["train"]
        splits = all_data.train_test_split(test_size=0.1, seed=seed)
        train_data = splits["train"]
        test_data = splits["test"]
    else:
        all_data = load_dataset(*hf_args, trust_remote_code=True)
        train_data = all_data[icl_config.train_split]
        test_data = all_data[icl_config.test_split]

    id2label = train_data.features[icl_config.label_field].names

    return train_data, test_data, id2label


def _balance_labels(data, label_field: str, shots: int, seed: int):
    """Sample a label-balanced subset of data."""
    rand = random.Random(seed)

    label_mapping = {x[label_field]: [] for x in data}
    for x in data:
        label_mapping[x[label_field]].append(x)

    num_rounds = math.ceil(shots / len(label_mapping))
    new_data = [[] for _ in range(num_rounds)]
    for _, samples in label_mapping.items():
        indices = rand.sample(range(len(samples)), num_rounds % len(samples))
        while len(indices) < num_rounds:
            indices += rand.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))

        for i, idx in enumerate(indices):
            new_data[i].append(samples[idx])

    for i in range(len(new_data)):
        rand.shuffle(new_data[i])
    new_data = [item for sublist in new_data for item in sublist][:shots]
    return new_data


def load_icl(dataset: str, task_config, data_args):
    """Load and preprocess an ICL (in-context learning) dataset.

    Dataset names follow the pattern icl_{base}_{length}, e.g. icl_trec_coarse_128k.
    Shot count is stored in task_config.shots; balanced sampling is always used.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)

    base_name = _get_base_icl_name(dataset)
    icl_config = ICL_DATASETS[base_name]
    shot = task_config.shots

    text_field = icl_config.text_field
    label_field = icl_config.label_field
    num_labels = icl_config.num_labels

    train_data, test_data, _id2label = _load_hf_data(icl_config, seed=seed)

    # subsample and balance test data
    if max_test_samples is not None and len(test_data) > max_test_samples:
        test_data = test_data.shuffle(seed=seed)
        test_data = _balance_labels(test_data, label_field, max_test_samples, seed)
        test_data = datasets.Dataset.from_list(test_data)

    item_template = "{text}\nlabel: {label}"
    user_template = "Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}"
    system_template = "label:"
    prompt_template = user_template + "\n" + system_template

    def preprocess(sample):
        local_seed = (int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16) + seed) % 2**31

        demos = _balance_labels(train_data, label_field, shot, local_seed)

        label_map = list(range(num_labels))
        random.seed(local_seed)
        random.shuffle(label_map)

        context = "\n\n".join([
            item_template.format(text=selected_item[text_field], label=str(label_map[int(selected_item[label_field])]))
            for selected_item in demos
        ])
        return {"context": context, "question": sample[text_field], "answer": str(label_map[int(sample[label_field])])}

    final_data = test_data.map(preprocess, num_proc=40)

    def post_process(output, example):
        prediction = parse_output(output["output"], system_template)
        answer = example["answer"]
        em = normalize_answer(prediction) == normalize_answer(answer) if prediction else False
        return {"exact_match": em, "primary_metric": em}, {"parsed_output": prediction}

    return {
        "data": final_data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register the specific ICL variants that are used in evaluation configs
# ---------------------------------------------------------------------------

# length -> shots mapping for each ICL dataset (llama2 tokenizer lengths)
# lengths: 4k, 8k, 16k, 32k, 64k, 128k
_LENGTHS = ["4k", "8k", "16k", "32k", "64k", "128k"]
_ICL_SHOTS = {
    "icl_trec_coarse": [200, 400, 800, 1600, 3300, 6600],
    "icl_trec_fine":   [200, 400, 800, 1600, 3200, 6400],
    "icl_banking77":   [180, 360, 720, 1450, 2900, 5900],
    "icl_clinic150":   [220, 440, 880, 1750, 3525, 7050],
    "icl_nlu":         [250, 510, 1020, 2040, 4080, 8296],
}

for _base, _shots_list in _ICL_SHOTS.items():
    for _length, _shots in zip(_LENGTHS, _shots_list):
        register_task(
            f"{_base}_{_length}",
            loader=load_icl,
            input_max_length=LENGTH_MAP[_length],
            generation_max_length=20,
            stop_new_line=True,
            shots=_shots,
            primary_metric="exact_match",
        )
