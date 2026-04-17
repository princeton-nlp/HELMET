import torch

from model_utils.base import LLM, APIModel, format_chat, call_api
from model_utils.local import LocalModel, tokenize
from model_utils.hf import HFModel
from model_utils.vllm_model import VLLMModel
from model_utils.sglang_model import SGLangModel
from model_utils.openai_model import OpenAIModel, TgiVllmModel
from model_utils.anthropic_model import AnthropicModel
from model_utils.gemini_model import GeminiModel
from model_utils.together_model import TogetherModel

import logging
logger = logging.getLogger(__name__)


def load_LLM(args):
    kwargs = {}
    if "gpt" in args.model_name_or_path and "oss" not in args.model_name_or_path:
        model_cls = OpenAIModel
        kwargs['seed'] = args.seed
    elif "claude" in args.model_name_or_path:
        model_cls = AnthropicModel
    elif "gemini" in args.model_name_or_path:
        model_cls = GeminiModel
    elif "togetherapi" in args.model_name_or_path:
        model_cls = TogetherModel
    elif args.use_vllm:
        model_cls = VLLMModel
        kwargs['seed'] = args.seed
    elif args.use_tgi_serving or args.use_vllm_serving:
        model_cls = TgiVllmModel
        kwargs['seed'] = args.seed
        kwargs["endpoint_url"] = args.endpoint_url
        kwargs["api_key"] = args.api_key
    elif args.use_sglang:
        model_cls = SGLangModel
        kwargs['seed'] = args.seed
    else:
        model_cls = HFModel
        kwargs['seed'] = args.seed
        if args.no_torch_compile:
            kwargs["torch_compile"] = False
        if args.no_bf16:
            kwargs["torch_dtype"] = torch.float32
        if args.rope_theta is not None:
            kwargs["rope_theta"] = args.rope_theta

    # Normalize sampling parameters
    temperature = args.temperature
    top_p = args.top_p
    do_sample = args.do_sample
    if not do_sample or temperature == 0.0:
        if temperature != 0.0 or top_p != 1.0:
            logger.info(f"Greedy decoding: overriding temperature={temperature} -> 0.0, top_p={top_p} -> 1.0")
        temperature = 0.0
        top_p = 1.0
        do_sample = False

    logger.info(f"Loading model {args.model_name_or_path} with {model_cls.__name__}")
    model = model_cls(
        args.model_name_or_path,
        temperature=temperature,
        top_p=top_p,
        max_length=args.input_max_length,
        generation_max_length=args.generation_max_length,
        do_sample=do_sample,
        stop_new_line=args.stop_new_line,
        use_chat_template=args.use_chat_template,
        system_message=args.system_message,
        **kwargs,
    )

    return model
