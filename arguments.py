import argparse
import yaml
import ast
import os

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--tag", type=str, default="eval", help="tag to add to the output file")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm engine")
    parser.add_argument("--use_sglang", action="store_true", help="whether to use sglang engine")
    parser.add_argument("--use_tgi_or_vllm_serving", action="store_true", help="whether to use tgi or vllm serving engine")
    parser.add_argument("--endpoint_url", type=str,default="http://localhost:8080/v1/", help="endpoint url for tgi or vllm serving engine")

    # data settings
    parser.add_argument("--datasets", type=str, default=None, help="comma separated list of dataset names")
    parser.add_argument("--demo_files", type=str, default=None, help="comma separated list of demo files")
    parser.add_argument("--test_files", type=str, default=None, help="comma separated list of test files")
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to the saved file")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")

    # dataset specific settings
    parser.add_argument("--popularity_threshold", type=int, default=3, help="popularity threshold for popqa, in log scale")

    # evaluation settings
    parser.add_argument("--batch_mode", type=str, default="batch_api", help="batch mode for OpenAIModel and AnthropicModel, can select batch_api or multi_thread")
    parser.add_argument("--shots", type=int, default=2, help="total number of ICL demos")
    parser.add_argument("--input_max_length", type=str, default='8192', help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets")

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use sampling (false is greedy), overwrites temperature")
    parser.add_argument("--generation_max_length", type=str, default='10', help="max number of tokens to generate, can be separated by comma to match the specified datasets")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False, help="whether to stop generation at newline")
    parser.add_argument("--system_message", type=str, default=None, help="system message to add to the beginning of context")

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bf16 and use fp32")
    parser.add_argument("--no_torch_compile", action="store_true", help="disable torchcompile")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use chat template")
    parser.add_argument("--rope_theta", type=int, default=None, help="override rope theta")
    parser.add_argument("--thinking", action="store_true", help="for reasoning models (e.g., Deepseek-r1), when this is set, we allow the model to generate an additional 32k tokens and exclude all texts between <think>*</think> from the output for evaluation")

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument("--count_tokens", action="store_true", help="instead of running generation, just count the number of tokens (only for HF models not API)")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    if not args.do_sample and args.temperature != 0.0:
        args.temperature = 0.0
        logger.info("overwriting temperature to 0.0 since do_sample is False")

    return args
