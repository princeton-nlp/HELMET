from pydantic import BaseModel, Field

from datasets import load_dataset
from transformers import AutoTokenizer

from utils import calculate_metrics, parse_output
from data_module.base import register_task

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maps length suffix to token count (used to compute truncation length)
_LENGTH_VALUES = {
    "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
    "64k": 65536, "128k": 131072,
}

# Minimum document length in llama3 tokens to be included (used llama2 in the original HELMET)
_MIN_DOC_LENGTH = 131072
TRUNCATE_TOKENIZER = AutoTokenizer.from_pretrained("/scratch/gpfs/PLI/models/Llama-3.1-8B")

# Buffer for prompt instructions (template overhead)
_PROMPT_OVERHEAD = 200

_TRUNCATION_POSTFIX = " ... [the rest of the text is omitted]"

# prompts inspired by https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.
Although you are not given the context, you will be given a set of correct answers that achieves full scores on all metrics, and you need to assess the provided answers using the correct answers.

Below is your grading rubric:

Fluency:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Correctness:
- Score 0 (Incorrect): The answer does not agree with the provided correct answers at all.
- Score 1 (partly correct): Partly agree with one of the provided correct answers (for example, the question asks for a date and a person; the answer gets the date right but the person wrong).
- Score 2 (correct but not fully relevant): Fully agrees with one of the provided correct answers but mentions other completely irrelevant information. Note that extra details provided in the answer, even if not mentioned in the correct answers, should NOT be seen as irrelevant as long as they are relevant to the question to a reasonable extend.
- Score 3 (correct and relevant): Fully agrees with one of the provided correct answers and only provides information relevant to the question. Note that if the answer is longer than the correct answer, as long as everything in the answer is relevant to the question, it should still be given score 3. For example, if the correct answer is "the North Pole" and the answer is "They are headed for the North Pole", it should still be given a score of 3.

Now, read the following question, answer, and correct answers. First think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"fluency": 0, "correctness": 1}}.

Question: {question}
Correct answers: {correct_answers}
Answer: {parsed_output}"""


class JudgeScoresResponse(BaseModel):
    fluency: int = Field(ge=0, le=1, description="The fluency score of the answer")
    correctness: int = Field(ge=0, le=3, description="The correctness score of the answer")


_judge_client = None

def _call_judge(prompt):
    """Call GPT-4o to judge a single response, returning a JudgeScoresResponse."""
    global _judge_client
    if _judge_client is None:
        import openai
        _judge_client = openai.OpenAI()
    response = _judge_client.responses.parse(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": prompt}],
        response_format=JudgeScoresResponse,
        temperature=0.3,
    )
    return response.output_parsed


def load_narrativeqa(dataset: str, task_config, data_args):
    """Load and preprocess the NarrativeQA dataset.

    Documents are filtered for length (>131072 llama3 tokens) and then truncated
    to fit the target context window: input_length - 200 (prompt) - generation_max_length.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)
    shots = task_config.shots

    # Parse target input length from dataset name (e.g. narrativeqa_128k -> 131072)
    length_suffix = dataset.rsplit("_", 1)[-1]
    input_length = _LENGTH_VALUES[length_suffix]
    truncation_length = input_length - _PROMPT_OVERHEAD - task_config.generation_max_length

    user_template = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    all_data = load_dataset("narrativeqa")
    data = all_data["test"].shuffle(seed=seed)

    # Filter for long documents using llama3 tokenizer as reference
    tokenizer = TRUNCATE_TOKENIZER
    data = data.map(lambda x: {"_doc_len": len(tokenizer(x["document"]["text"])["input_ids"])})
    data = data.filter(lambda x: x["_doc_len"] > _MIN_DOC_LENGTH)
    data = data.remove_columns("_doc_len")

    # Extract fields
    data = data.map(lambda example: {
        "context": example["document"]["text"],
        "question": example["question"]["text"],
        "answer": [ex["text"] for ex in example["answers"]],
        "demo": "" if shots == 0 else (
            "For example:\n\n"
            + "\n\n".join([
                f"Question: {ex['question']['text']}\nAnswer: {ex['answers'][0]['text']}"
                for ex in all_data["train"].shuffle().select(range(shots))
            ])
            + "\n\nNow, use the following story to answer the question:\n\n"
        ),
    }, remove_columns=["document", "answers"])

    # Truncate context to target length
    sep_len = len(tokenizer(_TRUNCATION_POSTFIX)["input_ids"])

    def truncate(sample):
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > truncation_length:
            sample["context"] = sample["context"][:tokens["offset_mapping"][truncation_length - sep_len][1]] + _TRUNCATION_POSTFIX
        return sample

    data = data.map(truncate, num_proc=16)

    if max_test_samples is not None:
        data = data.select(range(min(max_test_samples, len(data))))

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]

        mets = {}
        judge_input = JUDGE_PROMPT.format(
            question=example["question"],
            correct_answers=answer,
            parsed_output=prediction,
        )

        scores = _call_judge(judge_input)
        # normalize to 0-1
        mets["judge_score"] = scores.fluency * scores.correctness / 3.0
        mets["primary_metric"] = mets["judge_score"]

        return mets, {"judge_scores": scores.model_dump()}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register NarrativeQA variants (4k through 128k)
# ---------------------------------------------------------------------------

for _length_name, _length_val in _LENGTH_VALUES.items():
    register_task(
        f"narrativeqa_{_length_name}",
        loader=load_narrativeqa,
        input_max_length=_length_val,
        generation_max_length=100,
        use_chat_template=True,
        shots=2,
        primary_metric="judge_score",
    )
