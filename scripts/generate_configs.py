import yaml

# cannot be shared ones: use_chat_template, shots, and stop_new_line

lengths_mapping = {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}
master_mapping = {
    # ruler tasks, shots: 0, use_chat_template: False, and stop_new_line: False
    "ruler_niah_s_1": { # NIAH Repeat
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/niah_single_1/validation_{v}.jsonl"
        } for k, v in {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}.items()
    },
    "ruler_niah_s_2": { # NIAH 
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/niah_single_2/validation_{v}.jsonl"
        } for k, v in {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}.items()
    },
    "ruler_niah_s_3": { # NIAH UUID
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/niah_single_3/validation_{v}.jsonl"
        } for k, v in {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}.items()
    },
    "ruler_niah_mk_1": { # NIAH MK Essay
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/niah_multikey_1/validation_{v}.jsonl"
        } for k, v in {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}.items()
    },
    "ruler_niah_mk_2": { # NIAH MK Needle
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/niah_multikey_2/validation_{v}.jsonl"
        } for k, v in  lengths_mapping.items()
    },
    "ruler_niah_mk_3": { # NIAH MK UUID
        k: {
            "input_length": v, "generation_max_length": 100, "test_files": f"data/ruler/niah_multikey_3/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_niah_mq": { # NIAH MQ
        k: {
            "input_length": v, "generation_max_length": 100, "test_files": f"data/ruler/niah_multiquery/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_niah_mv": { # NIAH MV
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/niah_multivalue/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_cwe": { # RULER CWE
        k: {
            "input_length": v, "generation_max_length": 100, "test_files": f"data/ruler/cwe/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_fwe": { # RULER FWE
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/fwe/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_vt": { # RULER VT
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/vt/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_niah_qa_1": { # SQuAD
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/qa_1/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "ruler_niah_qa_2": { # HotpotQA
        k: {
            "input_length": v, "generation_max_length": 50, "test_files": f"data/ruler/qa_2/validation_{v}.jsonl"
        } for k, v in lengths_mapping.items()
    },

    "json_kv": {
        k: {
            "input_length": v, "generation_max_length": 100, "test_files": f"data/json_kv/test_k" + ["50", "105", "220", "440", "900", "1800"][i] + "_dep6.jsonl", "demo_files": ""
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },

    # generation with citations -- alce
    "alce_asqa": { # ASQA
        k: {
            "input_length": v, "generation_max_length": 300, "test_files": f"data/alce/asqa_eval_gtr_top2000.json", "demo_files": f"prompts/asqa_revised.json", "name_postfix": ["_8", "_30", "_75", "_165", "_345", "_700"][i]
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "alce_qampari": { # QAMPARI
        k: {
            "input_length": v, "generation_max_length": 300, "test_files": f"data/alce/qampari_eval_gtr_top2000.json", "demo_files": f"prompts/qampari_revised.json", "name_postfix": ["_8", "_30", "_75", "_165", "_345", "_700"][i]
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },

    # RAG tasks, using KILT's datasets and retrieval corpus
    "kilt_nq": {
        k: {
            "input_length": v, "generation_max_length": 20, 
            "test_files": "data/kilt/nq-dev-multikilt_1000_k" + ["20", "50", "105", "220", "440", "1000"][i] + "_dep6.jsonl", 
            "demo_files": "data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    }, 
    "kilt_triviaqa": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "data/kilt/triviaqa-dev-multikilt_1000_k" + ["20", "50", "105", "220", "440", "1000"][i] + "_dep6.jsonl",
            "demo_files": "data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "kilt_hotpotqa": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "data/kilt/hotpotqa-dev-multikilt_1000_k" + ["20", "50", "105", "220", "440", "1000"][i] + "_dep3.jsonl",
            "demo_files": "data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "kilt_popqa": {
        k: {
            "input_length": v, "generation_max_length": 20, "name_postfix": "_3",
            "test_files": "data/kilt/popqa_test_1000_k" + ["20", "50", "105", "220", "440", "1000"][i] + "_dep6.jsonl", 
            "demo_files": "data/kilt/popqa_test_1000_k3_dep6.jsonl"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },

    # for longqa, we truncate by the length - 200 - the generation length
    "narrativeqa": {
        k: {
            "input_length": v, "generation_max_length": 100, "test_files": "", "demo_files": "", "name_postfix": f"_{v - 200 - 100}"
        } for k, v in lengths_mapping.items()
    },
    "infbench_qa_eng": {
        k: {
            "input_length": v, "generation_max_length": 10, "test_files": "", "demo_files": "", "name_postfix": f"_{v - 200 - 10}"
        } for k, v in lengths_mapping.items()
    },
    "infbench_choice_eng": {
        k: {
            "input_length": v, "generation_max_length": 10, "test_files": "", "demo_files": "", "name_postfix": f"_{v - 200 - 10}"
        } for k, v in lengths_mapping.items()
    },

    "infbench_sum_eng": {
        k: {
            "input_length": v, "generation_max_length": 1200, "test_files": "", "demo_files": "", "name_postfix": f"_{v - 200 - 1200}"
        } for k, v in lengths_mapping.items()
    },
    # for multi lexsum, we truncate by the length - 300 (prompt and buffer) - 400 (generation)
    "multi_lexsum": {
        k: {
            "input_length": v, "generation_max_length": 400, "test_files": "", "demo_files": "", "name_postfix": f"_{v - 300 - 400}"
        } for k, v in lengths_mapping.items()
    },

    "msmarco_rerank_psg": {
        k: {
            "input_length": v, "generation_max_length": 200, 
            "test_files": "data/msmarco/test_reranking_data_k" + ["14", "50", "130", "285", "600", "1000"][i] + "_dep3.jsonl",
            "demo_files": "data/msmarco/test_reranking_data_k10_dep3.jsonl"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },

    "icl_trec_coarse": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "", "demo_files": "", "name_postfix": "_" + ["200", "400", "800", "1600", "3300", "6600"][i] + "shot_balance"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "icl_trec_fine": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "", "demo_files": "", "name_postfix": "_" + ["200", "400", "800", "1600", "3200", "6400"][i] + "shot_balance"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "icl_banking77": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "", "demo_files": "", "name_postfix": "_" + ["180", "360", "720", "1450", "2900", "5900"][i] + "shot_balance"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "icl_clinic150": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "", "demo_files": "", "name_postfix": "_" + ["220", "440", "880", "1750", "3525", "7050"][i] + "shot_balance"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
    "icl_nlu": {
        k: {
            "input_length": v, "generation_max_length": 20,
            "test_files": "", "demo_files": "", "name_postfix": "_" + ["250", "510", "1020", "2040", "4080", "8296"][i] + "shot_balance"
        } for i, (k, v) in enumerate(lengths_mapping.items())
    },
}

def process_configs(config_name, datasets, input_lengths, **kwargs):
    configs = []
    for i, d in enumerate(datasets):
        con = master_mapping[d]
        print(d)
        for l in input_lengths:
            c = con[l]
            print(c)
            configs.append({
                "input_max_length": c['input_length'],
                "datasets": d + c.get("name_postfix", ""),
                "generation_max_length": c['generation_max_length'],
                "test_files": c.get("test_files", ""),
                "demo_files": c.get("demo_files", ""),
            })
        out_config = {k: ",".join([str(c[k]) for c in configs]) for k in configs[0]}
        # llama 3 by default but you can change it to anything else
        out_config.update({
            **kwargs,
            "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "output_dir": "output/Llama-3.1-8B-Instruct",
            "model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "output_dir": "output/Llama-3.2-1B-Instruct",
        })
    with open(config_name, "w") as f:
        yaml.dump(out_config, f, sort_keys=False)

def helmet_configs(input_lengths = ["128k"], fname_postfix = ""):
    synthetic = ["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv", "json_kv"]
    # ruler actually doesn't support demos so it defaults to 0, json kv uses 2
    process_configs(
        f"configs/recall{fname_postfix}.yaml", synthetic, input_lengths, 
        use_chat_template=False, max_test_samples=100, shots=2, stop_new_line=False
    ) 

    rag = ['kilt_nq', 'kilt_triviaqa', 'kilt_hotpotqa', 'kilt_popqa']
    process_configs(
        f"configs/rag{fname_postfix}.yaml", rag, input_lengths,
        use_chat_template=False, max_test_samples=100, shots=2, stop_new_line=True # could be false but set to true so it runs faster
    )

    longqa = ['narrativeqa', 'infbench_qa_eng', 'infbench_choice_eng']
    process_configs(
        f"configs/longqa{fname_postfix}.yaml", longqa, input_lengths,
        use_chat_template=True, max_test_samples=100, shots=2, stop_new_line=False
    )

    summ = ['infbench_sum_eng', 'multi_lexsum']
    process_configs(
        f"configs/summ{fname_postfix}.yaml", summ, input_lengths,
        use_chat_template=True, max_test_samples=100, shots=2, stop_new_line=False
    )

    icl = ['icl_trec_coarse', 'icl_trec_fine', 'icl_banking77', 'icl_clinic150', 'icl_nlu']
    process_configs(
        f"configs/icl{fname_postfix}.yaml", icl, input_lengths,
        use_chat_template=False, max_test_samples=100, shots=0, stop_new_line=True
    )

    rerank = ["msmarco_rerank_psg"]
    process_configs(
        f"configs/rerank{fname_postfix}.yaml", rerank, input_lengths,
        use_chat_template=False, max_test_samples=100, shots=2, stop_new_line=True
    )

    cite = ["alce_asqa", "alce_qampari"]
    process_configs(
        f"configs/cite{fname_postfix}.yaml", cite, input_lengths,
        use_chat_template=True, max_test_samples=100, shots=2, stop_new_line=False
    )
    

def niah_configs():
    input_lengths = [8192, 16384, 32768, 65536, 131072]
    dataset=["ruler_niah_s_2"]
    gen_lengths = [50]
    for i, l in enumerate(input_lengths):
        config = {
            "input_max_length": l,
            "datasets": dataset[0],
            "generation_max_length": gen_lengths[0],
            "test_files": f'data/ruler/{dataset[0].replace("ruler_", "").replace("_s_", "_single_")}/validation_{l}.jsonl',
            "demo_files": "",
        }
    with open(f"configs/niah.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
    

def ruler_all_configs():
    input_lengths = [4096, 8192, 16384, 32768]
    input_lengths = [65536, 131072]

    dataset=["ruler_niah_s_1", "ruler_niah_s_2", "ruler_niah_s_3", "ruler_niah_mk_1", "ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mq", "ruler_niah_mv", "ruler_cwe", "ruler_fwe", "ruler_vt", "ruler_qa_1", "ruler_qa_2"]
    gen_lengths = [50, 50, 50, 50, 50, 100, 100, 50, 100, 50, 50, 50, 50]

    assert len(dataset) == len(gen_lengths)
    
    configs = []
    for i, d in enumerate(dataset):
        for l in input_lengths:
            configs.append({
                "input_max_length": l,
                "datasets": d,
                "generation_max_length": gen_lengths[i],
                "test_files": f'data/ruler/{d.replace("ruler_", "").replace("_s_", "_single_").replace("mq", "multiquery").replace("mk", "multikey").replace("mv", "multivalue")}/validation_{l}.jsonl',
                "demo_files": "",
            })

    # with open(f"configs/ruler_all{'' if max(input_lengths) <= 2**15 else '_long'}.yaml", "w") as f:
    with open(f"configs/niah{'' if max(input_lengths) <= 2**15 else '_long'}.yaml", "w") as f:
        config = {
            k: ",".join([str(c[k]) for c in configs]) for k in configs[0]
        }
        config.update({
            "use_chat_template": False,
            "max_test_samples": 100,
            "shots": 0,
            "stop_new_line": False,
            "model_name_or_path": "/scratch/gpfs/hyen/models/Meta-Llama-3.1-8B",
            "output_dir": "output/Meta-Llama-3.1-8B",
        })
        
        print(config)
        yaml.dump(config, f, sort_keys=False)


if __name__ == "__main__":
    helmet_configs()
    helmet_configs(input_lengths=["8k", "16k", "32k", "64k"], fname_postfix="_short")
    niah_configs()
    ruler_all_configs()
