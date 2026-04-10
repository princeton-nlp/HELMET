import os

from collections import defaultdict
import random
import json
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from arguments import parse_arguments, args_to_dict
from model_utils import load_LLM, HFModel, VLLMModel

from data import load_data


class PPLDataset(Dataset):
    """Dataset that pre-converts token id lists to CPU tensors."""
    def __init__(self, all_input_ids, all_labels, all_strides):
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids]
        self.labels = [torch.tensor(lbl, dtype=torch.long) for lbl in all_labels]
        self.strides = all_strides

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.strides[idx]

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.no_grad()
def compute_ppl_hf(model, all_input_ids, all_labels, all_strides, batch_size=1):
    """Compute per-sample perplexity using HF model forward pass with batched DataLoader."""
    dataset = PPLDataset(all_input_ids, all_labels, all_strides)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    results = []
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    device = model.model.device
    stride = all_strides[0]

    for input_ids, labels, _ in tqdm(dataloader, desc="Computing perplexity (HF)"):
        input_ids = input_ids.to(device)  # (B, seq_len)
        labels = labels.to(device)        # (B, seq_len)
        B = input_ids.size(0)

        outputs = model.model(input_ids=input_ids)
        logits = outputs.logits[:, -stride:]  # (B, stride, vocab_size)
        labels = labels[:, -stride:]          # (B, stride)

        # (B * stride, vocab_size) vs (B * stride,) -> (B * stride,)
        token_losses = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        # per-sample mean: (B, stride) -> (B,)
        mask = (labels != -100).float()
        avg_nlls = (token_losses.view(B, stride) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        ppls = torch.exp(avg_nlls)

        for i in range(B):
            results.append({
                "perplexity": ppls[i].item(),
                "avg_nll": avg_nlls[i].item(),
                "input_len": input_ids.size(1),
            })

        del outputs, logits
        torch.cuda.empty_cache()
    return results


def compute_ppl_vllm(model, all_input_ids, all_labels, all_strides):
    """Compute per-sample perplexity using vLLM prompt_logprobs."""
    from vllm import SamplingParams, TokensPrompt

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=0,  # return logprob of each prompt token
    )

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in all_input_ids]

    outputs = model.model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    results = []
    for output, labels, stride in zip(outputs, all_labels, all_strides):
        # prompt_logprobs[i] gives logprob of prompt_token_ids[i] given tokens 0..i-1
        # This corresponds to predicting labels[i-1] (since labels[j] = input_ids[j+1])
        # So to get the logprob for labels[j], we look at prompt_logprobs[j+1]
        prompt_logprobs = output.prompt_logprobs
        # only evaluate the last stride tokens of labels
        start_j = len(labels) - stride
        log_probs = []
        for j in range(start_j, len(labels)):
            if labels[j] == -100:
                continue
            # logprob for labels[j] is at prompt_logprobs[j+1]
            if j + 1 >= len(prompt_logprobs) or prompt_logprobs[j + 1] is None:
                continue
            token_id = output.prompt_token_ids[j + 1]  # == labels[j]
            lp = prompt_logprobs[j + 1][token_id].logprob
            log_probs.append(lp)

        if len(log_probs) > 0:
            avg_nll = -np.mean(log_probs)
            ppl = float(np.exp(avg_nll))
        else:
            avg_nll = float('inf')
            ppl = float('inf')

        results.append({
            "perplexity": ppl,
            "avg_nll": avg_nll,
            "input_len": len(output.prompt_token_ids),
        })

    return results


def run_test(args, model, dataset, test_file, demo_file):
    logger.info(f"running perplexity evaluation on {dataset} with test {test_file} and demo {demo_file}")
    tag = args.tag

    output_path = os.path.join(args.output_dir, f"{dataset}_{tag}_{args.seed}.json")
    if os.path.exists(output_path) and not args.overwrite and not args.debug:
        logger.info(f"{output_path} already exists, skipping...")
        averaged_metrics = json.load(open(output_path))["averaged_metrics"]
        return output_path, averaged_metrics

    random.seed(args.seed)
    data = load_data(args, dataset, test_file, demo_file)
    logger.info(f"loaded {len(data['data'])} samples from {dataset}")

    # data['data'] already has input_ids, labels, and stride from load_ppl
    all_input_ids = [sample['input_ids'] for sample in data['data']]
    all_labels = [sample['labels'] for sample in data['data']]
    all_strides = [sample['stride'] for sample in data['data']]

    if args.count_tokens:
        lengths = [len(ids) for ids in all_input_ids]
        logger.info(f"----{dataset}----\nAverage input length: {np.mean(lengths):.02f}, std: {np.std(lengths):.02f}, max: {max(lengths)}, min: {min(lengths)}\n----returning----")
        return output_path, None

    logger.info("Computing perplexity...")
    start_time = time.time()

    if isinstance(model, HFModel):
        ppl_results = compute_ppl_hf(model, all_input_ids, all_labels, all_strides)
    elif isinstance(model, VLLMModel):
        ppl_results = compute_ppl_vllm(model, all_input_ids, all_labels, all_strides)
    else:
        raise ValueError(f"Perplexity evaluation only supports HFModel and VLLMModel, got {type(model).__name__}")

    end_time = time.time()

    metrics = defaultdict(list)
    results = []
    for idx, ppl_result in enumerate(ppl_results):
        sample = data["data"][idx]

        metrics["perplexity"].append(ppl_result["perplexity"])
        metrics["avg_nll"].append(ppl_result["avg_nll"])
        metrics["input_len"].append(ppl_result["input_len"])

        result = {
            "domain": sample.get("domain", ""),
            "created": str(sample.get("created", "")),
            **ppl_result,
        }
        results.append(result)

        if idx < 5 or args.debug:
            logger.info(f"Example {idx+1}: input_len={ppl_result['input_len']}, ppl={ppl_result['perplexity']:.4f}, avg_nll={ppl_result['avg_nll']:.4f}")

        if args.debug:
            import pdb; pdb.set_trace()

    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    logger.info(f"Total time: {end_time - start_time:.02f} s")
    logger.info(f"Throughput: {len(results) / (end_time - start_time):.02f} samples/s")

    if len(results) == 0:
        logger.error("No results to evaluate, something went wrong, returning...")
        return output_path, None

    averaged_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}

    logger.info("Averaged metrics:")
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.04f}")

    output = {
        "args": args_to_dict(args),
        "data": results,
        "metrics": {k: [float(x) for x in v] for k, v in metrics.items()},
        "averaged_metrics": averaged_metrics,
        "throughput": len(results) / (end_time - start_time),
    }
    if not args.no_cuda:
        output["memory_usage"] = mem_usage

    if args.output_dir is not None:
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=4)
        except Exception as e:
            logger.error(f"Error writing to {output_path}, {e}")
            os.remove(output_path)
            raise e
        with open(output_path + ".score", "w") as f:
            json.dump(output["averaged_metrics"], f, indent=4)
        logger.info(f"done, results are written to {output_path}")

    return output_path, averaged_metrics


def main():
    args = parse_arguments()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = args.dataset_options.datasets
    test_files = args.dataset_options.test_files
    demo_files = args.dataset_options.demo_files
    max_lengths = args.dataset_options.input_max_length
    gen_lengths = args.dataset_options.generation_max_length
    use_chat_template = args.dataset_options.use_chat_template
    args.max_test_samples = args.dataset_options.max_test_samples
    args.shots = args.dataset_options.shots
    args.stop_new_line = args.dataset_options.stop_new_line
    args.tokenizer = args.model_name_or_path

    args.input_max_length = max(max_lengths)
    args.generation_max_length = max(gen_lengths)
    args.use_chat_template = any(use_chat_template)
    model = load_LLM(args)

    if not isinstance(model, (HFModel, VLLMModel)):
        raise ValueError(f"Perplexity evaluation only supports HFModel and VLLMModel, got {type(model).__name__}")

    success = True
    all_metrics = {}

    for dataset, test_file, demo_file, max_length, gen_length, uct in zip(datasets, test_files, demo_files, max_lengths, gen_lengths, use_chat_template):
        args.datasets = dataset
        args.test_files = test_file
        args.demo_files = demo_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length
        args.use_chat_template = uct

        model.max_length = max_length
        model.generation_max_length = gen_length
        model.use_chat_template = uct

        try:
            output_path, metrics = run_test(args, model, dataset, test_file, demo_file)
            all_metrics[dataset] = metrics

        except Exception as e:
            logger.exception(e)
            logger.error(f"Error in {dataset}, continuing...")
            success = False
            if args.debug:
                raise e

    if success:
        logger.info("All evaluations completed successfully!!")
        if args.config_path is not None:
            out_file = os.path.basename(args.config_path[0]).replace('.yaml', '')
            with open(os.path.join(args.output_dir, f"{out_file}_{args.tag}_{args.seed}_ppl_metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=4)
    else:
        logger.info("Some evaluations failed, exiting...")
        exit(1)

if __name__ == "__main__":
    main()
