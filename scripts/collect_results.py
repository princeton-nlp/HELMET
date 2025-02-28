import os
import json
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from tqdm import tqdm

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    
    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",
    
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    
    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],
    
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    
    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt-4-f1"],
    
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}
custom_avgs = {
    "Recall": ["json_kv substring_exact_match", "ruler_niah_mk_2 ruler_recall", "ruler_niah_mk_3 ruler_recall", "ruler_niah_mv ruler_recall"],
    "RAG": ['nq substring_exact_match', 'hotpotqa substring_exact_match', 'popqa substring_exact_match', 'triviaqa substring_exact_match',],
    "ICL": ['trec_coarse exact_match', 'trec_fine exact_match', 'banking77 exact_match', 'clinic150 exact_match', 'nlu exact_match'],
    "Cite": ['alce_asqa str_em', 'alce_asqa citation_rec', 'alce_asqa citation_prec', 'alce_qampari qampari_rec_top5', 'alce_qampari citation_rec', 'alce_qampari citation_prec', ],
    "Re-rank": ['msmarco_rerank_psg NDCG@10', ],
    "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    # "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall', 'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall', 'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall', 'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match', 'ruler_qa_2 substring_exact_match'],
    "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
}

@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3
        
    category: str = "synthetic"
    
    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_path(self):
        tag = self.tag
        path = os.path.join(self.output_dir, "{args.dataset}_{tag}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(args=self, tag=tag))

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")
        
        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None
    
    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)
        
        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}
        
        s = {m : v * (100 if m == "gpt-4-f1" else 1) * (100/3 if m == "gpt-4-score" else 1) for m, v in s.items()}
        print("found scores:", s)
        return s
        
    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []        
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d['ctxs'])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)
        
        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")

if __name__ == "__main__":
    # comment out the models you don't want to include, or add the new ones 
    models_configs = [
        {"model": "gpt-4-0125-preview", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-mini-2024-07-18", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-2024-05-13", "use_chat_template": True, "training_length": 128000},
        {"model": "gpt-4o-2024-08-06", "use_chat_template": True, "training_length": 128000},
        {"model": "claude-3-5-sonnet-20240620", "use_chat_template": True, "training_length": 200000},
        {"model": "gemini-1.5-flash-001", "use_chat_template": True, "training_length": 1048576},
        {"model": "gemini-1.5-pro-001", "use_chat_template": True, "training_length": 2097152},

        # llama 2 based models
        {"model": "Llama-2-7B-32K", "use_chat_template": False, "training_length": 32768},
        {"model": "Llama-2-7B-32K-Instruct", "training_length": 32768},
        {"model": "llama-2-7b-80k", "use_chat_template": False, "training_length": 80000},
        {"model": "Yarn-Llama-2-7b-64k", "use_chat_template": False, "training_length": 65536},
        {"model": "Yarn-Llama-2-7b-128k", "use_chat_template": False, "training_length": 131072},
        
        # llama 3 models
        {"model": "Meta-Llama-3-8B", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Instruct", "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Theta16M", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-8B-Instruct-Theta16M", "training_length": 8192},
        {"model": "Meta-Llama-3-70B-Theta16M", "use_chat_template": False, "training_length": 8192},
        {"model": "Meta-Llama-3-70B-Instruct-Theta16M", "training_length": 8192},
        
        {"model": "Llama-3.1-8B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        {"model": "Llama-3.1-70B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.1-70B-Instruct", "training_length": 131072},
        {"model": "Llama-3.3-70B-Instruct", "training_length": 131072},
        
        {"model": "Llama-3.2-1B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.2-1B-Instruct", "training_length": 131072},
        {"model": "Llama-3.2-3B", "use_chat_template": False, "training_length": 131072},
        {"model": "Llama-3.2-3B-Instruct", "training_length": 131072},
        
        # mistral models
        {"model": "Mistral-7B-v0.1", "use_chat_template": False, "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.1", "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.2", "training_length": 32768},
        {"model": "Mistral-7B-v0.3", "use_chat_template": False, "training_length": 32768},
        {"model": "Mistral-7B-Instruct-v0.3", "training_length": 32768},
        {"model": "Ministral-8B-Instruct-2410", "training_length": 131072},
        
        {"model": "Mistral-Nemo-Base-2407", "use_chat_template": False, "training_length": 128000},
        {"model": "Mistral-Nemo-Instruct-2407", "training_length": 128000},
        {"model": "MegaBeam-Mistral-7B-512k", "training_length": 524288},
        
        # yi models
        {"model": "Yi-6B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-9B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-34B-200K", "use_chat_template": False, "training_length": 200000},
        {"model": "Yi-1.5-9B-32K", "use_chat_template": False, "training_length": 32768},
        
        # phi models
        {"model": "Phi-3-mini-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-small-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-medium-128k-instruct", "training_length": 131072},
        {"model": "Phi-3.5-mini-instruct", "training_length": 131072},
        
        # qwen models
        {"model": "Qwen2-7B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2-7B-Instruct", "training_length": 32768},
        {"model": "Qwen2-57B-A14B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2-57B-A14B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-1.5B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2.5-1.5B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-3B", "use_chat_template": False, "training_length": 32768},
        {"model": "Qwen2.5-3B-Instruct", "training_length": 32768},
        {"model": "Qwen2.5-7B", "use_chat_template": False, "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct", "training_length": 131072},
        {"model": "Qwen2.5-72B-Instruct", "training_length": 131072},
        
        # prolong
        {"model": "Llama-3-8B-ProLong-512k-Instruct", "training_length": 524288},
        
        # gemma 2 models
        {"model": "gemma-2-9b", "use_chat_template": False, "training_length": 8192},
        {"model": "gemma-2-9b-it", "training_length": 8192},
        {"model": "gemma-2-9b-it-Theta320K", "training_length": 8192},

        {"model": "gemma-2-27b", "use_chat_template": False, "training_length": 8192},
        {"model": "gemma-2-27b-it", "training_length": 8192},
        {"model": "gemma-2-27b-it-Theta320K", "training_length": 8192},
        
        # others
        {"model": "c4ai-command-r-v01", "training_length": 131072},
        {"model": "Jamba-v0.1", "use_chat_template": False, "training_length": 262144},
        {"model": "AI21-Jamba-1.5-Mini", "training_length": 262144},
    ]

    
    models_configs = [
            {"model": "Llama-3.1-8B", "use_chat_template": False, "training_length": 131072},
            {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
            {"model": "DeepSeek-R1-Distill-Llama-8B", "training_length": 131072, "do_sample": True, "temperature": 0.6},
            {"model": "Qwen2-7B", "use_chat_template": False, "training_length": 32768},
            {"model": "Qwen2-7B-Instruct", "training_length": 32768},
            {"model": "DeepSeek-R1-Distill-Qwen-7B", "training_length": 131072, "do_sample": True, "temperature": 0.6},
    ]

    # set your configs here, only include the ones that you ran
    config_files = [
        "configs/recall.yaml", "configs/recall_short.yaml", 
        "configs/rag.yaml", "configs/rag_short.yaml", 
        "configs/longqa.yaml", "configs/longqa_short.yaml", 
        "configs/summ.yaml", "configs/summ_short.yaml", 
        "configs/rerank.yaml", "configs/rerank_short.yaml", 
        "configs/icl.yaml", "configs/icl_short.yaml", 
        "configs/cite.yaml", "configs/cite_short.yaml", 
        "configs/ruler.yaml", "configs/ruler_short.yaml", 
    ]

    dataset_configs = []
    for file in config_files:
        c = yaml.safe_load(open(file))
        
        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
        for d, t, l, g in zip(c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','), c['generation_max_length'].split(',')):
            dataset_configs.append({"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l), "generation_max_length": int(g), "max_test_samples": c['max_test_samples'], 'use_chat_template': c['use_chat_template'], 'shots': c['shots']})
    print(dataset_configs)    

    failed_paths = []
    df = []
    for model in tqdm(models_configs):
        args = arguments()
        args.tag = "v1" # SET YOUR TAG HERE
        args.output_dir = f"output/{model['model']}"
    
        for dataset in dataset_configs:
            args.update(dataset)
            args.update(model)

            metric = args.get_averaged_metric()
            dsimple, mnames = args.get_metric_name()

            if metric is None:
                failed_paths.append(args.get_path())
                continue
                
            for k, m in metric.items():
                df.append({**asdict(args), **model,
                    "metric name": k, "metric": m, 
                    "dataset_simple": dsimple + " " + k, "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                })

    all_df = pd.DataFrame(df)
    lf_df = all_df.pivot_table(index=["input_max_length", "model", ], columns="dataset_simple", values="metric", sort=False)
    lf_df = lf_df.reset_index()

    for k, v in custom_avgs.items():
        lf_df[k] = lf_df[v].mean(axis=1)

    print(lf_df.to_csv(index=False))

    print("Warning, failed to get the following paths, make sure that these are correct or the printed results will not be accurate:", failed_paths)
    # import pdb; pdb.set_trace()