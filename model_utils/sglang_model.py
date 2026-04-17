import time
from typing import Optional, List

from model_utils.base import format_chat
from model_utils.local import LocalModel

import logging
logger = logging.getLogger(__name__)


class SGLangModel(LocalModel):
    def __init__(self, model_name, seed=42, **kwargs):
        super().__init__(model_name, **kwargs)

        import sglang as sgl
        self.model = sgl.Engine(
            model_path=model_name,
            dtype="bfloat16",
            context_length=self.max_length,
            random_seed=seed,
            show_time_cost=True,
            decode_log_interval=1000,
            log_level="info",
        )
        self.tokenizer = self.model.tokenizer_manager.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _sampling_params(self):
        return {
            "temperature": self.temperature if self.do_sample else 0.0,
            "top_p": self.top_p,
            "max_new_tokens": self.generation_max_length,
            "stop": self.stops,
        }

    def _save_prompt(self, token_ids):
        if len(token_ids) > 1000:
            return self.tokenizer.decode(token_ids[:500]) + " <skip> " + self.tokenizer.decode(token_ids[-500:])
        return self.tokenizer.decode(token_ids)

    def generate(self, inputs=None, prompt: str = None, **kwargs):
        if inputs is None:
            assert prompt is not None
            if self.use_chat_template and isinstance(prompt, str):
                chat = format_chat(prompt, system_message=self.system_message)
                inputs = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, max_length=self.max_length - self.generation_max_length, truncation=True, padding=True)
                inputs = {'input_ids': inputs}
            else:
                inputs = self.tokenizer([prompt], max_length=self.max_length - self.generation_max_length, truncation=True)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            sampling_params=self._sampling_params(),
            **kwargs,
        )[0]
        return {
            "output": outputs["text"],
            "input_len": outputs["meta_info"]["prompt_tokens"],
            "output_len": outputs["meta_info"]["completion_tokens"],
            "input_text": self._save_prompt(inputs["input_ids"][0] if isinstance(inputs["input_ids"][0], list) else inputs["input_ids"]),
        }

    def generate_batch(self, inputs: Optional[List] = None, prompt: Optional[List[str]] = None, **kwargs):
        if inputs is None:
            assert prompt is not None
            if self.use_chat_template:
                chat = [format_chat(p, system_message=self.system_message) for p in prompt]
                inputs = [{'input_ids': self.tokenizer.apply_chat_template(c, tokenize=True, add_generation_prompt=True, max_length=self.max_length - self.generation_max_length, truncation=True, padding=True)} for c in chat]
            else:
                inputs = [self.tokenizer(p, truncation=True, max_length=self.max_length - self.generation_max_length) for p in prompt]

        start_time = time.time()
        outputs = self.model.generate(
            input_ids=[i['input_ids'] for i in inputs],
            sampling_params=self._sampling_params(),
            **kwargs,
        )
        logger.info(f"Finished batch generation for {len(inputs)} samples in {time.time() - start_time:.1f}s")

        return [
            {
                "output": output["text"],
                "input_len": output["meta_info"]["prompt_tokens"],
                "output_len": output["meta_info"]["completion_tokens"],
                "input_text": self._save_prompt(ins['input_ids']),
            } for ins, output in zip(inputs, outputs)
        ]
