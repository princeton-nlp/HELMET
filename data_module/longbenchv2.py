from datasets import load_dataset
from transformers import AutoTokenizer

from utils import calculate_metrics, parse_output
from data_module.base import register_task

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRUNCATE_TOKENIZER = AutoTokenizer.from_pretrained("/scratch/gpfs/PLI/models/Llama-3.1-8B")

_TRUNCATION_POSTFIX = " ... [the rest of the text is omitted]"

# Maps length suffix to (token count, LongBench-v2 length split)
_LENGTH_VARIANTS = {
    "64k":  (65536,  "short"),
    "256k": (262144, "medium"),
    "1m":   (1048576, "long"),
}


def load_longbenchv2(dataset: str, task_config, data_args):
    """Load and preprocess the LongBench-v2 multiple choice dataset.

    Samples are filtered by length category (short/medium/long) and presented
    as 4-choice questions (A/B/C/D).
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)

    # Parse length from dataset name (e.g. longbenchv2_256k)
    length_suffix = dataset.rsplit("_", 1)[-1]
    truncation_length, length_split = _LENGTH_VARIANTS[length_suffix]

    data = load_dataset("THUDM/LongBench-v2", split="train")
    data = data.filter(lambda x: x["length"] == length_split)

    user_template = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""

    system_template = "The correct answer is"
    prompt_template = user_template + "\n" + system_template

    def preprocess_example(example):
        example["answer"] = [example["answer"], example["answer"] + ". " + example[f"choice_{example['answer']}"]]
        return example

    data = data.map(preprocess_example)

    # Truncate context
    tokenizer = TRUNCATE_TOKENIZER
    sep_len = len(tokenizer(_TRUNCATION_POSTFIX)["input_ids"])

    def truncate(sample):
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > truncation_length:
            sample["context"] = sample["context"][:tokens["offset_mapping"][truncation_length - sep_len][1]] + _TRUNCATION_POSTFIX
        return sample

    data = data.map(truncate, num_proc=16)

    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        mets = calculate_metrics(prediction, answer)

        parsed_pred = parse_output(prediction, prefix=system_template)
        if parsed_pred is not None:
            new_mets = calculate_metrics(parsed_pred, answer)
            new_mets.pop("substring_exact_match")
            mets = {k: max(v, new_mets[k]) for k, v in mets.items()}

        mets["primary_metric"] = mets["exact_match"]
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register LongBench-v2 variants
# ---------------------------------------------------------------------------

for _length_name, (_token_count, _split) in _LENGTH_VARIANTS.items():
    register_task(
        f"longbenchv2_{_length_name}",
        loader=load_longbenchv2,
        input_max_length=_token_count,
        generation_max_length=128,
        use_chat_template=True,
        shots=0,
        primary_metric="exact_match",
    )
