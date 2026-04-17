from datasets import load_dataset

from data_module.base import register_task

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_ruler_templates(dataset: str):
    """Return (user_template, system_template) based on the RULER subtask type.

    Template strings may contain data-level placeholders like {type_needle_v},
    {query}, {num_v}, {example} that get filled from each example's fields.
    """
    # https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
    if "niah_mv" in dataset or "niah_mq" in dataset:
        user = "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system = "The special magic {type_needle_v} for {query} mentioned in the provided text are"
    elif "niah" in dataset:
        user = "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system = "The special magic {type_needle_v} for {query} mentioned in the provided text is"
    elif "vt" in dataset:
        user = "{example}Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above."
        system = "Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are:"
    elif "cwe" in dataset:
        user = "{example}Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?"
        system = "Answer: The top 10 words that appear most often in the list are:"
    elif "fwe" in dataset:
        user = "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words.\n{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?"
        system = "Answer: According to the coded text above, the three most frequently appeared words are:"
    elif "qa" in dataset:
        user = "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {question}"
        system = "Answer:"
    else:
        raise ValueError(f"Unknown RULER subtask in dataset name: {dataset}")
    return user, system


def load_ruler(dataset: str, task_config, data_args):
    """Load and preprocess a RULER synthetic evaluation dataset."""
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)

    data = load_dataset("json", data_files=task_config.test_file)["train"]

    user_template, system_template = _get_ruler_templates(dataset)
    prompt_template = user_template + "\n" + system_template

    def process_example(example):
        return {
            "question": example["query"] if "query" in example else example.get("question", ""),
            "example": example["example"] + "\n\n" if example.get("example") else "",
            "answer": example["answer"] if "answer" in example else example["outputs"],
        }
    data = data.map(process_example)

    if max_test_samples is not None:
        data = data.shuffle(seed).select(range(min(len(data), max_test_samples)))

    result = {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }

    # QA subtasks use default_post_process (substring_exact_match);
    # all other subtasks use ruler_recall
    if "qa" not in dataset:
        def post_process(output, example):
            prediction = output["output"]
            answer = example["answer"]
            recall = sum(a.lower() in prediction.lower() for a in answer) / len(answer)
            return {"ruler_recall": recall, "primary_metric": recall}, {"parsed_output": prediction}
        result["post_process"] = post_process

    return result


# ---------------------------------------------------------------------------
# Register RULER variants
# ---------------------------------------------------------------------------

# subtask_name -> (data_subdir, generation_max_length, primary_metric)
_RULER_SUBTASKS = {
    "ruler_niah_s_1":  ("niah_single_1",    50, "ruler_recall"),
    "ruler_niah_s_2":  ("niah_single_2",    50, "ruler_recall"),
    "ruler_niah_s_3":  ("niah_single_3",    50, "ruler_recall"),
    "ruler_niah_mk_1": ("niah_multikey_1",  50, "ruler_recall"),
    "ruler_niah_mk_2": ("niah_multikey_2",  50, "ruler_recall"),
    "ruler_niah_mk_3": ("niah_multikey_3", 100, "ruler_recall"),
    "ruler_niah_mq":   ("niah_multiquery", 100, "ruler_recall"),
    "ruler_niah_mv":   ("niah_multivalue",  50, "ruler_recall"),
    "ruler_cwe":       ("cwe",             100, "ruler_recall"),
    "ruler_fwe":       ("fwe",              50, "ruler_recall"),
    "ruler_vt":        ("vt",               50, "ruler_recall"),
    "ruler_qa_1":      ("qa_1",             50, "substring_exact_match"),
    "ruler_qa_2":      ("qa_2",             50, "substring_exact_match"),
}

_LONG_LENGTHS = {
    "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
    "64k": 65536, "128k": 131072, "256k": 262144, "512k": 524288, "1m": 1048576, "2m": 2097152,
}

# llama2 tokenizer variants (original HELMET)
for _name, (_subdir, _gen_len, _metric) in _RULER_SUBTASKS.items():
    for _length_name, _length_val in _LONG_LENGTHS.items():
        register_task(
            f"{_name}_{_length_name}_llama2",
            loader=load_ruler,
            input_max_length=_length_val,
            test_file=f"data/ruler/{_subdir}/validation_{_length_val}.jsonl",
            generation_max_length=_gen_len,
            shots=0,
            primary_metric=_metric,
        )

# llama3 tokenizer variants (default for HELMET v2)
for _name, (_subdir, _gen_len, _metric) in _RULER_SUBTASKS.items():
    for _length_name, _length_val in _LONG_LENGTHS.items():
        register_task(
            f"{_name}_{_length_name}",
            loader=load_ruler,
            input_max_length=_length_val,
            test_file=f"data/ruler_llama3/{_subdir}/validation_{_length_val}.jsonl",
            generation_max_length=_gen_len,
            shots=0,
            primary_metric=_metric,
        )
