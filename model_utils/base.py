import time
import functools
from typing import Optional, List, Dict, Any, Callable

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def format_chat(
    message: str,
    system_message: Optional[str] = None,
) -> List[Dict[str, str]]:
    if system_message is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    return [{"role": "user", "content": message}]


def merge_user_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge consecutive user messages into one (needed for models like Gemma 3)."""
    merged = []
    for message in messages:
        if message['role'] == 'user' and merged and merged[-1]['role'] == 'user':
            merged[-1]['content'] += "\n" + message['content']
        else:
            merged.append(message)
    return merged


def call_api(func: Callable, limit: int = 5, pause: int = 10):
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            msg = str(e).lower()
            if "rate limit" in msg or "rate_limit" in msg or "quota" in msg or "429" in msg:
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                output = None
                break
    return output


class LLM:
    """Base class for all generative models."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_length: int = 32768,
        generation_max_length: int = 2048,
        do_sample: bool = True,
        stop_new_line: bool = False,
        use_chat_template: bool = False,
        system_message: Optional[str] = "You are a helpful assistant.",
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.system_message = system_message
        self.stops = ["\n", "\n\n"] if stop_new_line else None

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def generate(self, inputs: Optional[Any] = None, prompt: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def generate_batch(self, inputs: Optional[List[Any]] = None, prompt: Optional[List[str]] = None, **kwargs) -> List[Optional[Dict[str, Any]]]:
        outputs = []
        if inputs is None:
            for p in tqdm(prompt):
                outputs.append(self.generate(prompt=p, **kwargs))
        else:
            for i in tqdm(inputs):
                outputs.append(self.generate(inputs=i, **kwargs))
        return outputs


class APIModel(LLM):
    """Base for API-based models that use thread_map for batch generation."""

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = [None for _ in prompt]
        else:
            prompt = [None for _ in inputs]

        if len(kwargs) > 0:
            logger.warning("kwargs are not supported for threaded batch generation")
        return thread_map(self.generate, inputs, prompt, max_workers=32)
