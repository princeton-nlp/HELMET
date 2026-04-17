import random
import hashlib

from datasets import load_dataset, load_from_disk

from utils import parse_rankings, calculate_retrieval_metrics
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


def load_msmarco_rerank(dataset: str, task_config, data_args):
    """Load and preprocess the MS MARCO passage reranking dataset.

    Each sample contains a query and a list of passages (ctxs) with relevance
    labels. The model must rank them from most to least relevant.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)
    shots = task_config.shots
    random.seed(seed)

    user_template = "You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}"
    system_template = "Ranking:"
    prompt_template = user_template + "\n" + system_template

    data = load_dataset("json", data_files=task_config.test_file)["train"]
    demos = load_dataset("json", data_files=task_config.demo_file)["train"]

    if max_test_samples is not None:
        key = "qid" if "qid" in data.column_names else "query"
        keys = set(data[key])
        keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
        data = data.filter(lambda x: x[key] in keys)

    # k values for retrieval metric computation
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(data[0]["ctxs"])]

    # Pre-filter demos to exclude test queries if demo set is large enough
    demo_filtered = False
    if len(demos) > 2 * len(data):
        qids = set(data["qid"])
        demos = demos.filter(lambda x: x["qid"] not in qids)
        demo_filtered = True

    def update(sample):
        passage_template = (
            "[ID: {id}] Document (Title: {title}): {text}"
            if "title" in sample["ctxs"][0]
            else "[ID: {id}] Document: {text}"
        )
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample["ctxs"]])

        demo_text = ""
        if shots > 0:
            demo_pool = demos
            if not demo_filtered:
                demo_pool = demos.filter(lambda x: x["qid"] != sample["qid"])
            h = abs(int(hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest(), 16) % 2**31)
            demo_pool = demo_pool.shuffle(seed=h)
            demo_pool = _drop_duplicates(demo_pool, "qid").select(range(shots))

            for d in demo_pool:
                ids = sorted(d["ctxs"], key=lambda x: x["label"], reverse=True)
                ranking = " > ".join([x["id"] for x in ids])
                demo_text += "\n\n".join([passage_template.format(**c) for c in d["ctxs"]]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}\n\n"

        gold_ranking = " > ".join([
            x["id"] for x in sorted(sample["ctxs"], key=lambda x: x["label"], reverse=True)
        ])
        qrel = [[c["id"], str(c["label"])] for c in sample["ctxs"]]

        return {
            "context": passage_text,
            "question": sample["query"],
            "demos": demo_text,
            "answer": gold_ranking,
            "qrel": qrel,
        }

    data = data.map(update, remove_columns=["query", "ctxs"])

    def post_process(output, example):
        parsed_pred = parse_rankings(output["output"])
        qrels = {example["qid"]: {c[0]: int(c[1]) for c in example["qrel"]}}
        mets = calculate_retrieval_metrics(
            results={example["qid"]: parsed_pred}, qrels=qrels, k_values=k_values,
        )
        mets["num_preds"] = len(parsed_pred)
        mets["primary_metric"] = mets["NDCG@10"]
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register msmarco_rerank_psg variants (4k through 128k)
# ---------------------------------------------------------------------------

_LENGTHS = ["4k", "8k", "16k", "32k", "64k", "128k"]
_K_VALUES = ["14", "50", "130", "285", "600", "1000"]

for _length, _k in zip(_LENGTHS, _K_VALUES):
    register_task(
        f"msmarco_rerank_psg_{_length}",
        loader=load_msmarco_rerank,
        input_max_length=LENGTH_MAP[_length],
        test_file=f"data/msmarco/test_reranking_data_k{_k}_dep3.jsonl",
        demo_file="data/msmarco/test_reranking_data_k10_dep3.jsonl",
        generation_max_length=200,
        shots=2,
        primary_metric="NDCG@10",
    )
