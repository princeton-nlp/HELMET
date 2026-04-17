import yaml


def expand(bases, lengths):
    """Expand base task names with length suffixes to produce registered names."""
    return [f"{b}_{l}" for b in bases for l in lengths]


def write_config(path, datasets, max_test_samples):
    config = {
        "dataset_options": {
            "datasets": datasets,
            "max_test_samples": max_test_samples,
        },
        "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
    }
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


_PPL_MAP = {
    "32k": "ppl_longmino_32768_8192",
    "64k": "ppl_longmino_65536_8192",
    "128k": "ppl_longmino_131072_8192",
    "256k": "ppl_longmino_262144_8192",
    "512k": "ppl_longmino_524288_8192",
    "1m": "ppl_longmino_1048576_8192",
}

_RULER_BASES = [
    "ruler_niah_s_1", "ruler_niah_s_2", "ruler_niah_s_3",
    "ruler_niah_mk_1", "ruler_niah_mk_2", "ruler_niah_mk_3",
    "ruler_niah_mq", "ruler_niah_mv",
    "ruler_cwe", "ruler_fwe", "ruler_vt",
    "ruler_qa_1", "ruler_qa_2",
]


def helmet_configs(lengths=["128k"], fname_postfix=""):
    write_config(
        f"configs/recall{fname_postfix}.yaml",
        expand(["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv", "json_kv"], lengths),
        100,
    )
    write_config(
        f"configs/rag{fname_postfix}.yaml",
        expand(["kilt_nq", "kilt_triviaqa", "kilt_hotpotqa", "kilt_popqa"], lengths),
        100,
    )
    write_config(
        f"configs/longqa{fname_postfix}.yaml",
        expand(["narrativeqa", "infbench_qa", "infbench_choice"], lengths),
        100,
    )
    write_config(
        f"configs/summ{fname_postfix}.yaml",
        expand(["infbench_sum", "multi_lexsum"], lengths),
        100,
    )
    write_config(
        f"configs/icl{fname_postfix}.yaml",
        expand(["icl_trec_coarse", "icl_trec_fine", "icl_banking77", "icl_clinic150", "icl_nlu"], lengths),
        500,
    )
    write_config(
        f"configs/rerank{fname_postfix}.yaml",
        expand(["msmarco_rerank_psg"], lengths),
        100,
    )
    write_config(
        f"configs/cite{fname_postfix}.yaml",
        expand(["alce_asqa", "alce_qampari"], lengths),
        100,
    )
    write_config(
        f"configs/alce_nocite{fname_postfix}.yaml",
        expand(["alce_asqa_nocite"], lengths),
        100,
    )
    write_config(
        f"configs/ruler{fname_postfix}.yaml",
        expand(_RULER_BASES, lengths),
        100,
    )


def additional_configs(lengths=["128k"], fname_postfix=""):
    write_config(
        f"configs/mrcr{fname_postfix}.yaml",
        expand(["mrcr_8"], lengths),
        100,
    )
    write_config(
        f"configs/recall_add{fname_postfix}.yaml",
        expand(["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv", "json_kv", "ruler_qa_1", "ruler_qa_2"], lengths),
        100,
    )


def longbenchv2_configs(lengths=["256k"], fname_postfix=""):
    write_config(
        f"configs/longbenchv2{fname_postfix}.yaml",
        expand(["longbenchv2"], lengths),
        None,
    )


def separate_configs(lengths=["128k"], fname_postfix=""):
    for name in ["kilt_nq", "kilt_triviaqa", "kilt_hotpotqa", "kilt_popqa"]:
        write_config(f"configs/rag/{name}{fname_postfix}.yaml", expand([name], lengths), 100)

    for name in ["icl_trec_coarse", "icl_trec_fine", "icl_banking77", "icl_clinic150", "icl_nlu"]:
        write_config(f"configs/icl/{name}{fname_postfix}.yaml", expand([name], lengths), 500)


def dev_configs():
    write_config(
        "configs/dev_32k_v2.yaml",
        expand(["json_kv", "ruler_niah_mk_2", "ruler_niah_mv", "kilt_nq", "kilt_popqa",
                "msmarco_rerank_psg", "mrcr_8", "icl_clinic150", "icl_nlu"], ["32k"]),
        100,
    )
    write_config(
        "configs/dev_128k_v2.yaml",
        expand(["json_kv", "ruler_niah_mk_2", "ruler_niah_mv", "kilt_nq", "kilt_popqa",
                "msmarco_rerank_psg", "mrcr_8", "icl_clinic150", "icl_nlu"], ["128k"]),
        100,
    )
    write_config(
        "configs/dev_64k_v2.yaml",
        expand(["kilt_nq", "kilt_popqa", "msmarco_rerank_psg", "icl_clinic150", "icl_nlu",
                "longbenchv2", "infbench_qa", "infbench_choice"], ["64k"]),
        100,
    )
    write_config(
        "configs/dev_256k_v2.yaml",
        expand(["json_kv", "ruler_niah_mk_2", "ruler_niah_mv", "mrcr_8", "longbenchv2"], ["256k"]),
        100,
    )


def synthetic_configs():
    bases = [
        "json_kv", "ruler_niah_mk_2", "ruler_niah_mv", "ruler_niah_mq",
        "ruler_fwe", "mrcr_4", "mrcr_8", "graphwalk_bfs_dep8", "graphwalk_parent",
    ]
    for length in ["32k", "64k", "128k", "256k", "512k", "1m"]:
        write_config(f"configs/dev_syn_{length}_v3.yaml", expand(bases, [length]), 100)


def ppl_configs():
    write_config("configs/ppl_longmino_short.yaml", [_PPL_MAP["32k"], _PPL_MAP["64k"]], 300)
    write_config("configs/ppl_longmino_64k.yaml", [_PPL_MAP["64k"]], 300)


if __name__ == "__main__":
    helmet_configs()
    helmet_configs(lengths=["8k", "16k", "32k", "64k"], fname_postfix="_short")
    separate_configs()
    separate_configs(lengths=["8k", "16k", "32k", "64k"], fname_postfix="_short")

    additional_configs(lengths=["512k", "1m"], fname_postfix="_long")
    additional_configs(lengths=["128k", "256k"])
    additional_configs(lengths=["32k", "64k"], fname_postfix="_short")

    longbenchv2_configs(lengths=["1m"], fname_postfix="_long")
    longbenchv2_configs(lengths=["256k"])
    longbenchv2_configs(lengths=["64k"], fname_postfix="_short")

    dev_configs()
    synthetic_configs()
    ppl_configs()

    write_config("configs/json_kv_256k.yaml", expand(["json_kv"], ["256k"]), 100)
