from datasets import Dataset as HFDataset

try:
    from .longproc.longproc.longproc_data import load_longproc_data
except ImportError:
    pass


def load_longproc_data_for_helmet(dataset: str, path="longproc_addon/longproc/data", max_test_samples=None, seed=42):
    # packed data: list of "input_prompt", "reference_output", "item"
    packed_data, eval_func = load_longproc_data(dataset, path)

    packed_data = HFDataset.from_list(packed_data)
    if max_test_samples is not None:
        packed_data = packed_data.shuffle(seed=seed).select(range(min(max_test_samples, len(packed_data))))

    def helmet_eval_wrapper(output: dict, example: dict):
        predict = output["output"]
        return eval_func(predict, example)

    return {
        "data": packed_data,
        "prompt_template": "{input_prompt}",
        "user_template": "{input_prompt}",
        "post_process": helmet_eval_wrapper,
    }


def _test_load_all():
    def test_loading(dataset):
        data, eval_func = load_longproc_data(dataset, "longproc_addon/longproc/data")
        print(f"Dataset: {dataset}")
        print(f"N samples: {len(data)}")
        print(f"Eval func: {eval_func}")
        print(f"Max input chars: {max([len(d['input_prompt']) for d in data])}")
        print(f"Max output chars: {max([len(d['reference_output']) for d in data])}")

    [test_loading(d) for d in ["path_traversal_0.5k", "path_traversal_2k", "path_traversal_8k"]]

    [test_loading(d) for d in ["html_to_tsv_0.5k", "html_to_tsv_2k", "html_to_tsv_8k"]]

    [test_loading(d) for d in ["pseudo_to_code_0.5k", "pseudo_to_code_2k",]]

    [test_loading(d) for d in ["travel_planning_2k", "travel_planning_8k"]]

    [test_loading(d) for d in ["tom_tracking_0.5k", "tom_tracking_2k", "tom_tracking_8k"]]

    [test_loading(d) for d in ["countdown_0.5k", "countdown_2k", "countdown_8k"]]


if __name__ == "__main__":
    _test_load_all()

