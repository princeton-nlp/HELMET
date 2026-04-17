import ast
import os

from dataclasses import dataclass, is_dataclass, asdict
from typing import List
from simple_parsing import ArgumentParser

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def args_to_dict(args):
    args_dict = args.__dict__.copy()
    args_dict.pop("config_path", None)
    for k, v in args_dict.items():
        if is_dataclass(v):
            args_dict[k] = asdict(v)
    return args_dict


@dataclass
class DatasetOptions:
    datasets: List[str] | None = None
    max_test_samples: int | None = None


def parse_arguments():
    parser = ArgumentParser(add_config_path_arg=True)

    parser.add_argument("--tag", type=str, default="eval", help="tag to add to the output file")
    parser.add_arguments(DatasetOptions, dest="dataset_options")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm engine")
    parser.add_argument("--use_sglang", action="store_true", help="whether to use sglang engine")
    parser.add_argument("--use_vllm_serving", action="store_true", help="whether to use vllm serving engine")
    parser.add_argument("--use_tgi_serving", action="store_true", help="whether to use tgi serving engine")
    parser.add_argument("--endpoint_url", type=str, default="http://localhost:8080/v1/", help="endpoint url for tgi or vllm serving engine")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="api key for model endpoint")

    # data settings
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to overwrite the saved file")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use sampling (false is greedy), overwrites temperature")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--system_message", type=str, default=None, help="system message to add to the beginning of context")

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bf16 and use fp32")
    parser.add_argument("--no_torch_compile", action="store_true", help="disable torchcompile")
    parser.add_argument("--rope_theta", type=int, default=None, help="override rope theta")
    parser.add_argument("--rope_scaling", type=int, default=None, help="override rope scaling factor")
    parser.add_argument("--thinking", type=int, default=None, help="for reasoning models (e.g., Deepseek-r1), when this is set, we allow the model to generate additional tokens and exclude all texts between <think>*</think> from the output for evaluation")

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument("--count_tokens", action="store_true", help="instead of running generation, just count the number of tokens (only for HF models not API)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    return args
