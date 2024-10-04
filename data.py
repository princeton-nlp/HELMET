import json
import os
import sys
import copy
import math
import random
import numpy as np

from collections import defaultdict
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import re
from utils import calculate_metrics, parse_output, parse_rankings, calculate_retrieval_metrics

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_contexts(data):
    # filter the contexts and only keep the ones that contain the answer
    new_data = []
    for d in data:
        d = copy.deepcopy(d)
        d["ctxs"] = [ctx for ctx in d["ctxs"] if ctx["has_answer"]]
        if len(d["ctxs"]) > 0:
            d["gold_doc"] = d["ctxs"][0]["text"]
            d["gold_title"] = d["ctxs"][0]["title"]
            new_data.append(d)
    return new_data


def drop_duplicates(data, key="id"):
    indices_to_keep = []
    keys = set()
    for i, d in enumerate(data):
        if d[key] in keys:
            continue
        indices_to_keep.append(i)
        keys.add(d[key])
    data = data.select(indices_to_keep)
    return data


def load_qa(dataset, path, demo_path, max_test_samples=None, popularity_threshold=None, shots=0):
    """
    Load the data for QA tasks
    """
    if "nq_bad" in dataset:
        user_template = "Use the given documents to write a concise and short answer to the question. Only use the information presented in the documents, and output 'unanswerable' if the question is not valid or cannot be answered with the given document. Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
    else:
        user_template = "Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    if path.endswith(".json"):
        data = load_dataset("json", data_files=path, field="data")["train"]
    elif path.endswith(".jsonl"):
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = load_from_disk(path)
        return {"data": data, "prompt_template": prompt_template, "user_template": user_template, "system_template": system_template}
    
    if demo_path.endswith(".json"):
        if "nq_bad" in dataset:
            with open(demo_path) as f:
                demo_data = json.load(f)
        else:
            demo_data = load_dataset("json", data_files=demo_path, field="data")["train"]
    else:
        demo_data = load_dataset("json", data_files=demo_path)["train"]

    # popularity filtering for popqa
    if "popqa" in dataset and popularity_threshold is not None:
        data = data.filter(lambda x: math.log10(x['s_pop']) < popularity_threshold)
        demo_data = demo_data.filter(lambda x: math.log10(x['s_pop']) < popularity_threshold)

    key = "id" if "id" in data.column_names else "question"
    if max_test_samples is not None:
        # some datasets do not have id (e.g., nq), so we assume unique questions
        keys = set(data[key])
        keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
        data = data.filter(lambda x: x[key] in keys)

    # demo_template = "Document (Title: {gold_title}): {gold_doc}\n\nQuestion: {question}\nAnswer: {answer}"
    demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
    passage_template = "Document (Title: {title}): {text}"
    def update(sample):
        demos = demo_data
        demo_text = ""
        if shots > 0:
            if 'popqa' in dataset:
                # popqa only has one split
                demos = demo_data.filter(lambda x: x[key] != sample[key])

            # unanswerable is the only one case (for now) where we care about balancing the labels
            if "nq_bad" in dataset:
                # this is in the format of {"normal": [], "unanswerable": []}, and we just sample from them accordingly
                count = (shots + 1) // 2
                idx1, idx2 = random.sample(range(len(demos["normal"])), count), random.sample(range(len(demos["unanswerable"])), count)
                dlist = [[demos["normal"][i1], demos["unanswerable"][i2]] for i1, i2 in zip(idx1, idx2)] 
                for d in dlist:
                    random.shuffle(d)
                demos = [d for sublist in dlist for d in sublist]
                
            else:
                # seed ensures that we get the same demos for the same question
                demos = demos.shuffle(seed=hash(sample[key]) % ((sys.maxsize + 1) * 2))
                demos = drop_duplicates(demos, key).select(range(shots))
            demo_text = "\n\n".join([demo_template.format(**d, documents="\n\n".join([passage_template.format(**c) for c in d["ctxs"]]), answer=d["answers"][0]) for d in demos]) + "\n\n"
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


def load_json_kv(path, shots, max_test_samples=None, seed=42):
    # prompt from https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/prompts/kv_retrieval.prompt
    user_template = "{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n{demos}Key: {question}"
    system_template = "Corresponding value:"
    prompt_template = user_template + "\n" + system_template

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
        # we don't really need to parse because we ues substring em, but could be nice to see how precise the model is
        parsed_pred = parse_output(prediction, "corresponding value:")
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data, 
        "prompt_template": prompt_template, 
        "user_template": user_template, 
        "system_template": system_template,
        "post_process": post_process,
    }


def truncate_llama2(dataset, data, postfix_text=" ... [the rest of the text is omitted]"):
    # use the llama 2 tokenizer to truncate to max_length, which only applies to the main document (context) and exclude the instructions and the demos
    # this is to make sure that every model see the same amount of information
    max_length = int(dataset.split("_")[-1]) 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    separator_length = len(tokenizer(postfix_text)["input_ids"])
    
    def truncate(sample):
        # tokens = tokenizer(sample["context"], max_length=max_length, truncation=True, return_offsets_mapping=True)
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > max_length:
            # we need to truncate
            sample["context"] = sample["context"][:tokens["offset_mapping"][max_length-separator_length][1]] + postfix_text
        return sample
    return data.map(truncate, num_proc=16)


def load_narrativeqa(dataset, path=None, shots=0, max_samples=None, seed=42):
    user_template = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    if path is not None and path != "":
        data = load_from_disk(path)
    else:
        all_data = load_dataset("narrativeqa")
        data = all_data["test"].shuffle(seed=seed)
        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))
        data = data.map(lambda example: {
            "context": example["document"]["text"],
            "question": example["question"]["text"],
            "answer": [ex["text"] for ex in example["answers"]],
            "demo": "" if shots == 0 else "For example:\n\n" + "\n\n".join([f"Question: {ex['question']['text']}\nAnswer: {ex['answers'][0]['text']}" for ex in all_data["train"].shuffle().select(range(shots))]) + "\n\nNow, use the following story to answer the question:\n\n"
        }, remove_columns=["document", "answers"])
        data = truncate_llama2(dataset, data)

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def drop_duplicates_in_input(untokenized_dataset):
    # https://github.com/tau-nlp/scrolls/blob/bfc0da0747976418cd0c4b8837db023ea567ba84/evaluator/dataset_evaluator.py#L107
    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def load_qasper(dataset, path=None, shots=0, max_samples=None, seed=42):
    user_template = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n\n{demo}{context}\n\nQuestion: {question}'
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template
    if path is not None and path != "":
        data = load_from_disk(path)
    else:
        # instead of using allenai/qasper, we use tau/scrolls, because it's nicely preprocessed
        # but the instructions are from zeroscrolls
        all_data = load_dataset("tau/scrolls", "qasper")
        data = drop_duplicates_in_input(all_data["validation"]).shuffle(seed=seed)
        train_data = drop_duplicates_in_input(all_data["train"])
        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))

        data = data.map(lambda example: {
            "context": example["input"][example["input"].index("\n\n")+2:].strip(),
            "question": example["input"][:example["input"].index("\n\n")].strip(),
            "answer": example["outputs"],
            # "demo": "" if shots == 0 else "\n\n".join(["[Text omitted]\n\nQuestion: {}\nAnswer: {}".format(ex['input'][:ex['input'].index('\n\n')].strip(), ex['outputs'][0]) for ex in train_data.shuffle().select(range(shots))]) + "\n\n"
            "demo": "" if shots == 0 else "For example:\n\n" + "\n\n".join(["Question: {}\nAnswer: {}".format(ex['input'][:ex['input'].index('\n\n')].strip(), ex['outputs'][0]) for ex in train_data.shuffle().select(range(shots))]) + "\n\nNow, use the following article to answer the question:\n\n"
        }, remove_columns=["outputs"])
        data = truncate_llama2(dataset, data)
        
    return {"data": data, "prompt_template": prompt_template, "user_template": user_template, "system_template": system_template}


def load_qmsum(dataset, path=None, shots=0, max_samples=None, seed=42):
    # https://github.com/Yale-LILY/QMSum/blob/main/data_process.ipynb
    def clean_data(text):
        text = text.replace('{ vocalsound } ', '')
        text = text.replace('{ disfmarker } ', '')
        text = text.replace('a_m_i_', 'ami')
        text = text.replace('l_c_d_', 'lcd')
        text = text.replace('p_m_s', 'pms')
        text = text.replace('t_v_', 'tv')
        text = text.replace('{ pause } ', '')
        text = text.replace('{ nonvocalsound } ', '')
        text = text.replace('{ gap } ', '')
        return text
    
    all_data = load_dataset("json", data_files={"train": "/scratch/gpfs/hyen/QMSum/data/ALL/jsonl/train.jsonl", "validation": "/scratch/gpfs/hyen/QMSum/data/ALL/jsonl/val.jsonl", "test": "/scratch/gpfs/hyen/QMSum/data/ALL/jsonl/test.jsonl"})
    all_data = all_data.map(lambda x: {
        "context": clean_data("\n".join([f"{s['speaker']}: {s['content']}" for s in x["meeting_transcripts"]]))
    })
    all_data = truncate_llama2(dataset, all_data)

    demos = {"specific_query_list": [], "general_query_list": []}
    for d in all_data["train"]:
        for t in ["specific_query_list", "general_query_list"]:
            demos[t] += d[t]

    test_data = []
    for d in all_data["validation"]:
        for t in ["specific_query_list", "general_query_list"]:
            for q in d[t]:
                demo_text = ""
                if shots > 0:
                    demo = random.sample(demos[t], shots)
                    demo_text = "For example:\n\n" + "\n\n".join(["Question: {}\nAnswer: {}".format(ex['query'], ex['answer']) for ex in demo]) + "\n\nNow, use the following transcript to answer the question:\n\n"

                test_data.append({
                    "context": d["context"],
                    "question": q["query"],
                    "answer": q["answer"],
                    "demo": demo_text,
                    "relevant_span": [clean_data("\n".join([f"{s['speaker']}: {s['content']}" for s in d["meeting_transcripts"][int(x[0]):int(x[1])+1]])) for x in q["relevant_text_span"]] if "relevant_text_span" in q else None,
                    "type": t,
                })
    
    user_template = "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\n{demo}{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    if "specific" in dataset:
        test_data = [x for x in test_data if x["type"] == "specific_query_list"]
    elif "general" in dataset:
        test_data = [x for x in test_data if x["type"] == "general_query_list"]

    if max_samples is not None and len(test_data) > max_samples:
        test_data = random.sample(test_data, max_samples)

    return {
        "data": test_data, 
        "prompt_template": prompt_template, 
        "user_template": user_template, 
        "system_template": system_template
    }


def load_multi_lexsum(dataset, path=None, shots=0, max_samples=None, seed=42):
    all_data = load_dataset("allenai/multi_lexsum", name="v20230518")
    all_data = all_data.filter(lambda x: x["summary/short"] is not None)

    user_template = "You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n{context}\n\nNow please summarize the case."
    system_template = "Summary:"
    prompt_template = user_template + "\n\n" + system_template
    train_data = all_data["train"]

    all_data = all_data.map(lambda x: {
        "context": '\n\n'.join(x["sources"]),
        "demo": "" if shots == 0 else "Example summaries:\n\n" + "\n\n".join(["Summary: {}".format(ex["summary/short"]) for ex in train_data.shuffle().select(range(shots))]) + "\n\nNow, write a summary of the following legal documents.\n",
        "answer": x["summary/short"],
        "question": "",
    })
    all_data = truncate_llama2(dataset, all_data)
    test_data = all_data["validation"]

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        mets = calculate_metrics(prediction, answer)
        # we don't really need to parse because we ues substring em, but could be nice to see how precise the model is
        parsed_pred = parse_output(prediction, system_template)
        if parsed_pred is not None:
            new_mets = calculate_metrics(parsed_pred, answer)
            mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
        return mets, {"parsed_output": parsed_pred}

    if max_samples is not None and len(test_data) > max_samples:
        test_data = test_data.shuffle(seed=seed).select(range(max_samples))
    
    return {
        "data": test_data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def load_flenqa(path, dataset="flenqa", flenqa_ctx_size=512):
    user_template = "{question}\n\n{context}\n\n{instruction}\n\n{question}"
    system_template = "Answer:" if "cot" not in dataset else "Reasoning:"
    prompt_template = user_template + "\n" + system_template

    all_data = load_dataset("alonj/flenqa")
    def process_example(example):
        vanilla = "Answer the question with only 'True.' or 'False.'"
        cot = "Think step by step. First, recall the relevant information from the text, and then reason about the text to answer the question. Write your answer in the following format:\nReasoning: [Reasoning]\nAnswer: [True/False]."
        if example["dataset"] == "PIR":
            question = f"Question: {example['assertion/question']}"
            context = f"{example['mixin']}"
            instruction = "Answer the question as it appears exactly based on the given text." + (f" {cot}" if "cot" in dataset else f" {vanilla}")

        elif example["dataset"] == "Simplified RuleTaker":
            question = f"Statement: {example['assertion/question']}"
            context = f"Rules: {example['rule']}\nFacts: {example['mixin']}"
            instruction = "Answer whether the statement can be derived from the given rule and facts." + (f" {cot}" if "cot" in dataset else f" {vanilla}")

        elif example["dataset"] == "MonoRel":
            question = f"Question: {example['assertion/question']}"
            context = f"{example['mixin']}"
            instruction = "Answer the question as it appears exactly based on the given text." + (f" {cot}" if "cot" in dataset else f" {vanilla}")

        else:
            raise ValueError(f"Unknown dataset {example['dataset']}")

        return {
            "context": context,
            "question": question,
            "answer": example["label"],
            "instruction": instruction,
        }

    # we only want to parse one word
    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        patterns = [re.compile(r"(?:answer:)(?:\s+)(\w*)", flags=re.IGNORECASE), re.compile(r"(?:^)(?:\s+)(\w*)")]
        for pat in patterns:
            matches = pat.search(prediction)
            if matches is not None:
                prediction = matches[1].strip() # 0 index includes the non-capturing group
                break
        mets = calculate_metrics(prediction, answer)
        return mets, {"parsed_output": prediction}

    test_data = all_data["eval"].map(process_example)
    test_data = test_data.filter(lambda x: x['ctx_size'] == flenqa_ctx_size)
    return {
        "data": test_data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def load_msmarco_rerank(path, demo_path=None, max_test_samples=None, shots=0):
    user_template = "You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}"
    system_template = "Ranking:"
    prompt_template = user_template + "\n" + system_template

    if path.endswith(".jsonl"):
        # we have preprocessed it into a jsonl file
        data = load_dataset("json", data_files=path)["train"]
    else:
        data = load_from_disk(path)
    
    demos = load_dataset("json", data_files=demo_path)["train"]

    def get_qrels(data):
        # for evaluation, to be passed into trec_eval
        qrels = {}
        for d in data:
            qrels[d["qid"]] = {c["id"]: c["label"] for c in d["ctxs"]}
        return qrels

    if max_test_samples is not None:
        key = "qid" if "qid" in data.column_names else "query"
        keys = set(data[key])
        keys = random.sample(sorted(keys), min(max_test_samples, len(keys)))
        data = data.filter(lambda x: x[key] in keys)
    
    # the k values are used to calculate metrics later
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(data[0]["ctxs"])]
    qrels = get_qrels(data)

    # could also do this question by question, but not necessary if we are sampling
    demo_filtered = False
    if len(demos) > 2*len(data):
        qids = set(data["qid"])
        demos = demos.filter(lambda x: x["qid"] not in qids)
        demo_filtered = True

    def update(sample, demos):
        passage_text = ""

        passage_template = "[ID: {id}] Document (Title: {title}): {text}"  if "title" in sample["ctxs"][0] else "[ID: {id}] Document: {text}"
        passage_text = "\n\n".join([passage_template.format(**c) for c in sample['ctxs']])
        gold_ranking = " > ".join([x['id'] for x in sorted(sample["ctxs"], key=lambda x: x["label"], reverse=True)])
        demo_text = ""

        if shots > 0:
            # need to make sure we don't pick the same question as the demos
            if not demo_filtered:
                demos = demos.filter(lambda x: x["qid"] != sample["qid"])
            demo = demos.shuffle(seed=hash(sample["qid"]) % ((sys.maxsize + 1) * 2))
            demo = drop_duplicates(demo, 'qid').select(range(shots))
            
            demo_ids = set()
            for d in demo:
                if d["qid"] in demo_ids or len(demo_ids) >= shots:
                    continue
                demo_ids.add(d["qid"])
                # sort ids by label
                ids = sorted(d["ctxs"], key=lambda x: x["label"], reverse=True)
                ranking = " > ".join([x['id'] for x in ids])
                demo_text += "\n\n".join([passage_template.format(**c) for c in d['ctxs']]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"

        return {"context": passage_text, "question": sample["query"], "demos": demo_text, "answer": gold_ranking}

    data = data.map(lambda x: update(x, demos), remove_columns=["query", "ctxs"])

    def post_process(output, example):
        parsed_pred = parse_rankings(output["output"])
        o = {"parsed_output": parsed_pred}
        # qrels = {k: v for k, v in example["qrel"].items() if v is not None}
        mets = calculate_retrieval_metrics({example['qid']: parsed_pred}, qrels, k_values)
        mets = {**mets, "num_preds": len(parsed_pred)}
        return mets, o

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "qrels": qrels,
        "k_values": k_values,
        "post_process": post_process,
    }


def load_icl(dataset, max_test_sample=None, seed=42):
    shot = int(dataset.split("shot")[0].split("_")[-1])

    if "trec_fine" in dataset.lower():
        train_data = load_dataset("CogComp/trec", trust_remote_code=True)["train"]
        test_data = load_dataset("CogComp/trec", trust_remote_code=True)["test"]
        text_field = "text"
        label_field = "fine_label"
        num_labels = 50
    elif "trec_coarse" in dataset.lower():
        train_data = load_dataset("CogComp/trec", trust_remote_code=True)["train"]
        test_data = load_dataset("CogComp/trec", trust_remote_code=True)["test"]
        text_field = "text"
        label_field = "coarse_label"
        num_labels = 6
    elif "sst2" in dataset.lower():
        train_data = load_dataset("nyu-mll/glue", "sst2")["train"]
        test_data = load_dataset("nyu-mll/glue", "sst2")["validation"]
        text_field = "sentence"
        label_field = "label"
        num_labels = 2
    elif "banking77" in dataset.lower():
        train_data = load_dataset("PolyAI/banking77")["train"]
        test_data = load_dataset("PolyAI/banking77")["test"]
        id2label = train_data.features["label"].names
        id2label = {i: id2label[i] for i in range(len(id2label))}
        text_field = "text"
        label_field = "label"
        num_labels = 77
    elif "clinic150" in dataset.lower():
        train_data = load_dataset("clinc_oos", "plus")["train"]
        test_data = load_dataset("clinc_oos", "plus")["validation"]
        text_field = "text"
        label_field = "intent"
        num_labels = 151
    elif "nlu" in dataset.lower():
        data = load_dataset("xingkunliuxtracta/nlu_evaluation_data", trust_remote_code=True)["train"]
        data = data.train_test_split(test_size=0.1, seed=seed)
        train_data = data["train"]
        test_data = data["test"]
        text_field = "text"
        label_field = "label"
        num_labels = 68
    elif "dialogre" in dataset.lower():
        label2id = {'per:alternate_names': 0, 'per:alumni': 1, 'per:positive_impression': 2, 'unanswerable': 3, 'per:place_of_residence': 4, 'per:employee_or_member_of': 5, 'per:girl/boyfriend': 6, 'per:title': 7, 'gpe:residents_of_place': 8, 'org:employees_or_members': 9, 'per:children': 10, 'per:parents': 11, 'per:siblings': 12, 'per:spouse': 13, 'per:friends': 14, 'per:negative_impression': 15, 'per:client': 16, 'per:pet': 17, 'per:place_of_work': 18, 'per:boss': 19, 'per:subordinate': 20, 'per:acquaintance': 21, 'per:roommate': 22, 'per:dates': 23, 'per:other_family': 24, 'per:age': 25, 'per:visited_place': 26, 'gpe:visitors_of_place': 27, 'per:origin': 28, 'per:neighbor': 29, 'per:works': 30, 'per:schools_attended': 31, 'org:students': 32, 'per:major': 33, 'per:date_of_birth': 34, 'per:place_of_birth': 35, 'gpe:births_in_place': 36}
        num_labels = len(label2id)
        train_data = load_dataset("dataset-org/dialog_re", trust_remote_code=True)['train']
        test_data = load_dataset("dataset-org/dialog_re", trust_remote_code=True)['validation']

        def dialogre_convert(dataset):
            new_dataset = []
            for conv in dataset:
                conv_history = "dialog:\n" + "\n".join(conv["dialog"]) + "\n"
                for i in range(len(conv["relation_data"]['r'])):
                    text = conv_history + f"entity pair: [{conv['relation_data']['x'][i]}], [{conv['relation_data']['y'][i]}]"
                    new_dataset.append({"text": text, "label": [label2id[x] for x in conv["relation_data"]["r"][i]]})
            return new_dataset
        
        train_data = dialogre_convert(train_data)
        test_data = dialogre_convert(test_data)
        text_field = "text"
        label_field = "label"
    
    def balance_labels(data, shots):
        label_mapping = {x[label_field]: [] for x in data}
        for x in data:
            label_mapping[x[label_field]].append(x)
        
        # rearrange the data such that every label has the same number of samples
        # they are also in consecutive sets with random order in each set
        num_rounds = math.ceil(shots / len(label_mapping))
        new_data = [[] for _ in range(num_rounds)]
        for _, samples in label_mapping.items():
            indices = random.sample(range(len(samples)), num_rounds % len(samples))
            while len(indices) < num_rounds:
                # sample with replacement if necessary, shouldn't happen unless we have very many shots 
                indices += random.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))
            
            for i, idx in enumerate(indices):
                new_data[i].append(samples[idx])

        for i in range(len(new_data)):
            random.shuffle(new_data[i])
        new_data = [item for sublist in new_data for item in sublist][:shots]
        return new_data
        
    if max_test_sample is not None and len(test_data) > max_test_sample:
        test_data = test_data.shuffle(seed=seed).select(range(max_test_sample))

    item_template = "{text}\nlabel: {label}"
    user_template = "Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}"
    system_template = "label:"
    prompt_template = user_template + "\n" + system_template

    def preprocess(sample):
        if "balance" in dataset:
            demos = balance_labels(train_data, shot)
        else:
            demos = []
            while len(demos) < shot:
                demos += list(np.random.choice(train_data, min(len(train_data), shot - len(demos)), replace=False))
        if "natural_label" in dataset:
            label_mapping = [id2label[i] for i in range(num_labels)]
        else:
            label_mapping = list(range(num_labels))
            random.shuffle(label_mapping)

        if "dialogre" in dataset.lower():
            context = "\n\n".join([
                item_template.format(text=selected_item[text_field], label=",".join([str(label_mapping[int(x)]) for x in selected_item[label_field]]))
                for selected_item in demos]
            )
            return {"context": context, "question": sample[text_field], "answer": [str(label_mapping[int(x)]) for x in sample[label_field]]}
        else:
            context = "\n\n".join([
                item_template.format(text=selected_item[text_field], label=str(label_mapping[int(selected_item[label_field])]))
                for selected_item in demos]
            )
            return {"context": context, "question": sample[text_field], "answer": str(label_mapping[int(sample[label_field])])}
    
    final_data = test_data.map(preprocess, num_proc=40)

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]
        prediction = parse_output(prediction, system_template)
        mets = calculate_metrics(prediction, answer)
        return mets, {"parsed_output": prediction}

    def post_process_dialogre(output, example):
        prediction = output["output"]
        answers = example["answer"] # a list
        prediction = parse_output(prediction, system_template) # a string, multiple answers separated by ,
        preds = list(set([p.strip() for p in prediction.split(",")]))

        # TODO: technically unanswerable needs to be processed separately....
        correct = sum([p in answers for p in preds]) # longicl eval allows label to be substring of p
        precision = correct / len(preds) if len(preds) > 0 else 0
        recall = correct / len(answers) if len(answers) > 0 else 0 # although labels should not be empty
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        print("answers", answers)
        print("preds", preds)
        print("p, r, f1", precision, recall, f1)
        print("-----")

        return {"dialogre_precision": precision, "dialogre_recall": recall, "dialogre_f1": f1}, {"parsed_output": preds, "labels": answers}

    return {
        "data": final_data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process_dialogre if "dialogre" in dataset.lower() else post_process,
    }


def load_longicl(dataset):
    keys = {"1r": "1 Round Prompt", "2r": "2 Round Prompt", "3r": "3 Round Prompt", "4r": "4 Round Prompt", "5r": "5 Round Prompt"}
    field = [v for k, v in keys.items() if k in dataset.lower()][0]

    def replace_label_name(s, label2id):
        for old_label, new_label in label2id.items():
            s = s.replace(old_label, str(new_label))
        return s

    if "banking77" in dataset.lower():
        # in total 77 classes
        label2id = {'unable_to_verify_identity': 0, 'transfer_not_received_by_recipient': 1, 'get_disposable_virtual_card': 2, 'pending_top_up': 3, 'lost_or_stolen_card': 4, 'card_payment_wrong_exchange_rate': 5, 'exchange_rate': 6, 'cash_withdrawal_charge': 7, 'request_refund': 8, 'receiving_money': 9, 'automatic_top_up': 10, 'wrong_exchange_rate_for_cash_withdrawal': 11, 'terminate_account': 12, 'disposable_card_limits': 13, 'getting_virtual_card': 14, 'card_about_to_expire': 15, 'balance_not_updated_after_bank_transfer': 16, 'verify_source_of_funds': 17, 'declined_card_payment': 18, 'exchange_via_app': 19, 'card_arrival': 20, 'apple_pay_or_google_pay': 21, 'card_acceptance': 22, 'card_payment_fee_charged': 23, 'card_not_working': 24, 'cash_withdrawal_not_recognised': 25, 'virtual_card_not_working': 26, 'get_physical_card': 27, 'cancel_transfer': 28, 'top_up_reverted': 29, 'activate_my_card': 30, 'declined_cash_withdrawal': 31, 'card_delivery_estimate': 32, 'change_pin': 33, 'card_payment_not_recognised': 34, 'balance_not_updated_after_cheque_or_cash_deposit': 35, 'direct_debit_payment_not_recognised': 36, 'pin_blocked': 37, 'top_up_limits': 38, 'edit_personal_details': 39, 'transaction_charged_twice': 40, 'supported_cards_and_currencies': 41, 'pending_transfer': 42, 'atm_support': 43, 'why_verify_identity': 44, 'age_limit': 45, 'order_physical_card': 46, 'declined_transfer': 47, 'exchange_charge': 48, 'country_support': 49, 'Refund_not_showing_up': 50, 'verify_my_identity': 51, 'pending_card_payment': 52, 'card_swallowed': 53, 'reverted_card_payment?': 54, 'fiat_currency_support': 55, 'verify_top_up': 56, 'top_up_by_cash_or_cheque': 57, 'getting_spare_card': 58, 'passcode_forgotten': 59, 'topping_up_by_card': 60, 'contactless_not_working': 61, 'card_linking': 62, 'beneficiary_not_allowed': 63, 'top_up_failed': 64, 'extra_charge_on_statement': 65, 'wrong_amount_of_cash_received': 66, 'transfer_timing': 67, 'transfer_into_account': 68, 'top_up_by_card_charge': 69, 'visa_or_mastercard': 70, 'compromised_card': 71, 'failed_transfer': 72, 'transfer_fee_charged': 73, 'lost_or_stolen_phone': 74, 'top_up_by_bank_transfer_charge': 75, 'pending_cash_withdrawal': 76}
        user_template = "{context}\n{question}"
        system_template = "intent category:"
        prompt_template = user_template + "\n" + system_template
        def process_example(example):
            context = example[field]
            query = context.split("\n")[-2]
            context = "\n".join(context.split("\n")[:-2])
            if "numlabel" in dataset:
                return {"context": replace_label_name(context, label2id), "question": query, "answer": replace_label_name(example["label"], label2id)}
            else:
                return {"context": context, "question": query, "answer": example["label"]}
        data = load_dataset("tiger-lab/longiclbench", split="BANKING77")
        data = data.map(process_example, remove_columns=data.column_names)

        def post_process(output, example):
            prediction = output["output"]
            answer = example["answer"]
            prediction = parse_output(prediction, "intent category:")
            mets = calculate_metrics(prediction, answer)
            return mets, {"parsed_output": prediction}
    elif "tacred" in dataset.lower():
        # in total 41 classes
        label2id = {'per:title': 0, 'per:stateorprovince_of_birth': 1, 'per:stateorprovince_of_death': 2, 'org:top_members/employees': 3, 'org:subsidiaries': 4, 'per:city_of_birth': 5, 'org:city_of_headquarters': 6, 'org:founded_by': 7, 'per:religion': 8, 'per:siblings': 9, 'per:date_of_birth': 10, 'org:shareholders': 11, 'per:countries_of_residence': 12, 'org:political/religious_affiliation': 13, 'per:origin': 14, 'org:parents': 15, 'per:other_family': 16, 'per:parents': 17, 'org:member_of': 18, 'org:stateorprovince_of_headquarters': 19, 'per:age': 20, 'per:cause_of_death': 21, 'per:cities_of_residence': 22, 'org:founded': 23, 'per:children': 24, 'org:dissolved': 25, 'per:stateorprovinces_of_residence': 26, 'per:country_of_birth': 27, 'per:spouse': 28, 'per:alternate_names': 29, 'per:city_of_death': 30, 'per:country_of_death': 31, 'org:number_of_employees/members': 32, 'per:schools_attended': 33, 'per:date_of_death': 34, 'per:charges': 35, 'org:website': 36, 'org:country_of_headquarters': 37, 'org:members': 38, 'org:alternate_names': 39, 'per:employee_of': 40}
        user_template = "{context}\n{question}"
        system_template = "the relation between the two entities is:"
        prompt_template = user_template + "\n" + system_template
        def process_example(example):
            context = example[field]
            query = context.split("\n")[-3] + "\n" + context.split("\n")[-2] 
            context = "\n".join(context.split("\n")[:-3])
            if "numlabel" in dataset:
                return {"context": replace_label_name(context, label2id), "question": query, "answer": replace_label_name(example["label"], label2id)}
            else:
                return {"context": context, "question": query, "answer": example["label"]}
        data = load_dataset("tiger-lab/longiclbench", split="TacRED")
        data = data.map(process_example, remove_columns=data.column_names)

        def post_process(output, example):
            prediction = output["output"]
            answer = example["answer"]
            prediction = parse_output(prediction, system_template)
            mets = calculate_metrics(prediction, answer)
            return mets, {"parsed_output": prediction} 

    elif "dialogre" in dataset.lower():
        # a major difference between how we calculate the scores vs. the original longiclbench paper is that
        # they calculate the precision and recall over all predictions and labels
        # https://github.com/TIGER-AI-Lab/LongICLBench/blob/main/dialogueRE_evaluate.py
        # but we calculate the precision and recall for example data sample and then average over all examples
        # this should not affect the findings much but the absolute numbers will be different

        # in total 34 classes
        label2id = {'per:major': 0, 'per:girl/boyfriend': 1, 'org:students': 2, 'per:pet': 3, 'per:place_of_work': 4, 'per:place_of_residence': 5, 'per:roommate': 6, 'per:boss': 7, 'per:client': 8, 'per:subordinate': 9, 'per:date_of_birth': 10, 'per:visited_place': 11, 'per:other_family': 12, 'gpe:visitors_of_place': 13, 'per:negative_impression': 14, 'per:dates': 15, 'per:works': 16, 'per:title': 17, 'per:positive_impression': 18, 'per:schools_attended': 19, 'per:friends': 20, 'org:employees_or_members': 21, 'per:alumni': 22, 'per:spouse': 23, 'per:origin': 24, 'per:siblings': 25, 'per:children': 26, 'per:employee_or_member_of': 27, 'per:alternate_names': 28, 'per:acquaintance': 29, 'gpe:residents_of_place': 30, 'per:neighbor': 31, 'per:parents': 32, 'per:age': 33}

        user_template = "{context}{question}"
        system_template = "The {n} respective relations between each entity pair are:"
        prompt_template = user_template + "\n" + system_template

        def post_process(output, example):
            prediction = output["output"]
            answer = example["answer"]

            labels = answer.split(",")
            labels = [l.strip() for l in labels]
            # we can try to parse the prediction first
            prediction = parse_output(prediction, "respective relations between each entity pair are:")
            preds = prediction.split(",")
            preds = [p.strip() for p in preds]
            for i, p in enumerate(preds):
                # try to parse in the format of "type:relation" and disregard the starting and ending strings
                matches = re.findall(r"(\w+:\w+)", p)
                if len(matches) > 0:
                    preds[i] = matches[0]
            # zip makes sure that we only compare the same number of predictions and labels
            correct = sum([l in p for p, l in zip(preds, labels)]) # longicl eval allows label to be substring of p
            num_preds = len(preds)
            num_labels = len(labels)
            precision = correct / num_preds if num_preds > 0 else 0
            recall = correct / num_labels if num_labels > 0 else 0 # although labels should not be empty
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            return {"precision": precision, "recall": recall, "f1": f1, "num_preds": num_preds, "num_labels": num_labels}, {"parsed_output": preds, "labels": labels}

        def process_example(example):
            context = example[field]
            idx = context.rfind("Dialogue: \n")
            query = context[idx:].strip()
            context = context[:idx]
            query = query[:query.rfind("\n")]
            if "numlabel" in dataset:
                return {"context": replace_label_name(context, label2id), "question": query, "answer": replace_label_name(example["label"], label2id), "n": len(example["label"].split(","))}
            else:
                return {"context": context, "question": query, "answer": example["label"], "n": len(example["label"].split(","))}
        data = load_dataset("tiger-lab/longiclbench", split="DialogRE")
        data = data.map(process_example, remove_columns=data.column_names)

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


def load_ruler(dataset, path, max_test_samples=None, seed=42):
    data = load_dataset("json", data_files=path)["train"]
    user_template = "{context}\n\n{question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    # https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
    if "mv_niah" in dataset or "mq_niah" in dataset:
        user_template = "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system_template = "The special magic {type_needle_v} for {query} mentioned in the provided text are"
    elif "niah" in dataset:
        user_template = "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?"
        system_template = "The special magic {type_needle_v} for {query} mentioned in the provided text is"
    elif "vt" in dataset:
        user_template = "{example}Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above."
        system_template = "Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are:"
    elif "cwe" in dataset:
        user_template = "{example}Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?"
        system_template = "Answer: The top 10 words that appear most often in the list are:"
    elif "fwe" in dataset:
        user_template = "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words.\n{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?"
        system_template = "Answer: According to the coded text above, the three most frequently appeared words are:"
    elif "qa" in dataset:
        # note that for qa, instead of calculating the recall, we simply check for substring exact match
        user_template = "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {question}"
        system_template = "Answer:"
    else:
        raise NotImplementedError(f"Unknown ruler dataset {dataset}")
    prompt_template = user_template + "\n" + system_template

    def process_example(example):
        return {
            "question": example["query"] if "query" in example else example["question"] if "question" in example else "", 
            "example": example["example"] + "\n\n" if "example" in example and example["example"] != "" else "",
            "answer": example["answer"] if "answer" in example else example['outputs'],
        }
    data = data.map(process_example)

    def post_process(output, example):
        # we don't do any parsing since we are only checking for substring exact match
        prediction = output["output"]
        answer = example["answer"]
        recall = sum([a.lower() in prediction.lower() for a in answer]) / len(answer)
        mets = {"ruler_recall": recall}
        return mets, {"parsed_output": prediction}
    
    if max_test_samples is not None:
        data = data.shuffle(seed).select(range(min(len(data), max_test_samples)))

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process if "qa" not in dataset else default_post_process,
    }


def load_alce(dataset, path, demo_path, shots=0):
    # demo path is the prompt file
    with open(demo_path, "r") as f:
        demos = json.load(f)
    instruction = demos["instruction"]
    demo_prompt = demos["demo_prompt"]
    doc_prompt = demos["doc_prompt"]
    # there are 5 docs for each demo, and we use all of them
    
    user_template = "{demo_text}\n\n\n{instruction}\n\nQuestion: {question}\n\n{context}"
    system_template = "Answer:"
    prompt_template = user_template + "\n\n" + system_template

    data = load_dataset("json", data_files=path)["train"]

    num_docs = int(dataset.split("_")[-1])

    def preprocess_example(example):
        context = "\n\n".join([doc_prompt.format(**d, ID=idx+1) for idx, d in enumerate(example["docs"][:num_docs])])
        demo_text = "\n\n\n".join([
            demo_prompt.format(**demo, instruction=instruction, context = "\n\n".join([doc_prompt.format(**d, ID=idx+1) for idx, d in enumerate(demo["docs"])]))
            for demo in random.sample(demos["demos"], shots)
        ])
        return {"context": context, "demo_text": demo_text, "instruction": instruction}
    data = data.map(preprocess_example)
    
    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def load_infbench(dataset, shots=0, max_test_samples=None, seed=42):
    from datasets import load_dataset, Value, Sequence, Features
    ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"), "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
    data = load_dataset("xinrongzhang2022/infinitebench", features=ft)
   
    # https://github.com/OpenBMB/InfiniteBench/blob/main/src/prompt.py 
    # slightly modified to be consistent with other datasets, shouldn't affect performance
    post_process = default_post_process
    if "qa_eng" in dataset:
        user_template = "You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n{demo}{context}\n\nQuestion: {question}"
        system_template = "Answer:"
        data = data["longbook_qa_eng"]
    elif "choice_eng" in dataset:
        user_template = "You are given a story and a question with multiple choices. Choose the best answer from the options provided. Only one of the following options is correct, output the answer using one single letter (A, B, C, or D). Don't say anything else.\n\n{demo}{context}\n\nQuestion: {question}\nOptions:\n{options}"
        system_template = "Answer:"
        data = data["longbook_choice_eng"]
        def pp(output, example):
            prediction = output["output"]
            answer = example["answer"]
            mets = calculate_metrics(prediction, answer)
            mets.pop("substring_exact_match")

            parsed_pred = parse_output(prediction)
            if parsed_pred is not None:
                new_mets = calculate_metrics(parsed_pred, answer)
                new_mets.pop("substring_exact_match")
                mets = {k: max(v, new_mets[k]) for k, v in mets.items()}

            # we only allow for substring exact match for the second answer (A. option)
            # to make it easier to collect the results, we merge exact_match and substring_exact_match here
            mets["substring_exact_match"] = False
            if answer[1].lower() in prediction.lower():
                # we shouldn't need to do other normalization
                mets["substring_exact_match"] = True
                mets["exact_match"] = True
            return mets, {"parsed_output": parsed_pred}

        post_process = pp
        
    elif "sum_eng" in dataset:
        user_template = "You are given a book and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary.\n\n{demo}{context}\n\nNow summarize the book."
        system_template = "Summary:"
        data = data["longbook_sum_eng"]
    prompt_template = user_template + "\n\n" + system_template

    def process_example(example):
        update = {"question": example["input"], "demo": ""}
        if "choice" in dataset:
            options = "A. {}\nB. {}\nC. {}\nD. {}".format(*example["options"])
            answer = example["options"].index(example["answer"][0])
            answer = chr(ord("A") + answer)
            update["options"] = options
            update["answer"] = [answer, f"{answer}. {example['answer'][0]}"]
        return update
    
    data = truncate_llama2(dataset, data)
    all_data = data.map(process_example)

    data = all_data
    if max_test_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    def add_demos(example):
        demos = all_data.filter(lambda x: x["id"] != example["id"]).shuffle(seed=seed).select(range(shots))
        if "qa_eng" in dataset:
            temp = "[story text]\nQuestion: {question}\nAnswer: {answer[0]}"
            demo = "\n\n".join([temp.format(**x) for x in demos])
        elif "choice_eng" in dataset:
            temp = "[story text]\nQuestion: {question}\nOptions:\n{options}\nAnswer: {answer[0]}"
            demo = "\n\n".join([temp.format(**x) for x in demos])
        elif "sum_eng" in dataset:
            demo = "\n\n".join([f"[story text]\nSummary: {x['answer'][0].strip()}" for x in demos])
        return {"demo": f"For example:\n\n{demo}\n\nNow, read the following story:\n\n"}
    if shots > 0:
        data = data.map(add_demos)

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }

def shuffle_labels(data, method="shuffle"):
    """
    For classification tasks with fixed number of labels, we can shuffle the labels to make the task harder.
    The model needs to rely on the demo more than using the clue from the label names.
    We support different ways of doing this.
     1. shuffle -- the label names don't change but we shuffle them (a bijection mapping from old to new and different label)
     2. numbers -- change labels to 0 to n-1 
     3. uuid -- change labels to random uuids
    """
    # 1. create the mapping from original label to the new label
    label_set = list(set(data["data"]["answer"]))
    if method == "shuffle":
        # random shuffle and then create a mapping, this gives us a random bijection mapping
        random.shuffle(label_set)
        mapping = {label_set[i]: label_set[(i+1) % len(label_set)] for i in range(len(label_set))}
    elif method == "numbers":
        mapping = {label: i for i, label in enumerate(label_set)}
    elif method == "uuid":
        import uuid
        mapping = {label: str(uuid.uuid4()) for label in label_set}
    else:
        raise NotImplementedError(f"Unknown method {method}")

    logger.info(f"Mapping: {mapping}")
    # 2. replace the original label with the new label in the text
    # we do the replace with system_template prepend to avoid replacing the label strings that are also substrings of the test text
    pattern = re.compile("|".join(mapping.keys()))
    def replace(sample):
        context_mapping = {data["system_template"].format(sample) + " " + k: data["system_template"].format(sample) + " " + v for k, v in mapping.items()}
        context_pattern = re.compile("|".join(context_mapping.keys()))
        return {
            # "context": context_pattern.sub(lambda x: context_mapping[re.escape(x.group(0))], sample["context"]),
            "context": pattern.sub(lambda x: mapping[re.escape(x.group(0))], sample["context"]),
            "answer": mapping[sample["answer"]],
            "original_answer": sample["answer"],
        }
    data["data"] = data["data"].map(replace)


def default_post_process(output, example):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    return mets, {"parsed_output": parsed_pred}


def load_data(args, dataset, path=None, demo_path=None):
    if "popqa" in dataset:
        popularity_threshold = float(dataset.split("_")[-1])
        data = load_qa(dataset, path, demo_path, max_test_samples=args.max_test_samples, popularity_threshold=popularity_threshold, shots=args.shots)
    elif any([x in dataset for x in ["nq", "hotpotqa", "triviaqa"]]):
        data = load_qa(dataset, path, demo_path, max_test_samples=args.max_test_samples, shots=args.shots)
    elif dataset == "json_kv":
        data = load_json_kv(path, args.shots, args.max_test_samples, args.seed)
    elif "narrativeqa" in dataset:
        data = load_narrativeqa(dataset, path, args.shots, args.max_test_samples, args.seed)
    elif "qasper" in dataset:
        data = load_qasper(dataset, path, args.shots, args.max_test_samples, args.seed)
    elif "flenqa" in dataset:
        data = load_flenqa(path, dataset, args.flenqa_ctx_size)
        if args.max_test_samples is not None:
            data["data"] = data["data"].shuffle(seed=args.seed).select(range(min(args.max_test_samples, len(data["data"]))))
    elif "msmarco" in dataset:
        data = load_msmarco_rerank(path, demo_path, args.max_test_samples, args.shots)
    elif "longicl" in dataset:
        data = load_longicl(dataset)
        if args.max_test_samples is not None:
            data["data"] = data["data"].shuffle(seed=args.seed).select(range(min(args.max_test_samples, len(data["data"]))))
    elif "alce" in dataset:
        data = load_alce(dataset, path, demo_path, args.shots)
        if args.max_test_samples is not None:
            data["data"] = data["data"].shuffle(seed=args.seed).select(range(min(args.max_test_samples, len(data["data"]))))
    elif "icl" in dataset:
        data = load_icl(dataset, max_test_sample=args.max_test_samples, seed=args.seed)
    elif "qmsum" in dataset:
        data = load_qmsum(dataset, path, args.shots, args.max_test_samples, seed=args.seed)
    elif "multi_lexsum" in dataset:
        data = load_multi_lexsum(dataset, path, args.shots, args.max_test_samples, seed=args.seed)
    elif "ruler" in dataset:
        if args.shots != 0:
            logger.info("RULER does not support ICL demos, not using any shots")
        data = load_ruler(dataset, path, args.max_test_samples, seed=args.seed)
    elif "infbench" in dataset:
        data = load_infbench(dataset, args.shots, args.max_test_samples, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    if "post_process" not in data:
        data["post_process"] = default_post_process
    
    return data


class TestItemDataset(Dataset):
    def __init__(self, data, llm, tokenizer):
        self.data = data
        self.llm = llm
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        inputs = self.llm.prepare_inputs(self.data["data"][idx], self.data)
        original_text = None
        if "input_ids" in inputs:
            original_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return inputs, original_text
