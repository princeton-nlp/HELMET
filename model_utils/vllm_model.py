import time
from typing import Optional, List

import torch

from model_utils.base import format_chat
from model_utils.local import LocalModel

import logging
logger = logging.getLogger(__name__)


class VLLMModel(LocalModel):
    def __init__(self, model_name, seed=42, **kwargs):
        super().__init__(model_name, **kwargs)

        from vllm import LLM
        model_kwargs = {}
        if kwargs.get("rope_scaling") is not None:
            model_kwargs["rope_scaling"] = {"type": "dynamic", "factor": kwargs["rope_scaling"], "original_max_position_embeddings": 32768}
        if kwargs.get("rope_theta") is not None:
            model_kwargs["rope_theta"] = kwargs["rope_theta"]

        self.model = LLM(
            model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=True,
            seed=seed,
            max_model_len=self.max_length,
            enable_chunked_prefill=True,
            **model_kwargs,
        )
        self.tokenizer = self.model.get_tokenizer()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            max_tokens=self.generation_max_length,
            stop=self.stops,
        )

    def _save_prompt(self, token_ids):
        if len(token_ids) > 1000:
            return self.tokenizer.decode(token_ids[:500]) + " <skip> " + self.tokenizer.decode(token_ids[-500:])
        return self.tokenizer.decode(token_ids)

    def generate(self, inputs=None, prompt: str = None, **kwargs):
        from vllm import TokensPrompt
        if inputs is None:
            assert prompt is not None
            if self.use_chat_template and isinstance(prompt, str):
                chat = format_chat(prompt, system_message=self.system_message)
                inputs = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", max_length=self.max_length - self.generation_max_length, truncation=True, padding=True)
                inputs = {'input_ids': inputs}
            else:
                inputs = self.tokenizer([prompt], return_tensors="pt", max_length=self.max_length - self.generation_max_length, truncation=True, padding=True)

        if inputs['input_ids'].shape[1] > self.max_length - self.generation_max_length:
            return None

        outputs = self.model.generate(
            prompts=TokensPrompt(prompt_token_ids=inputs["input_ids"][0].tolist()),
            sampling_params=self._sampling_params(),
            **kwargs,
        )[0]
        return {
            "output": outputs.outputs[0].text,
            "input_len": len(outputs.prompt_token_ids),
            "output_len": len(outputs.outputs[0].token_ids),
            "input_text": self._save_prompt(inputs["input_ids"][0].tolist()),
        }

    def generate_batch(self, inputs: Optional[List] = None, prompt: Optional[List[str]] = None, **kwargs):
        from vllm import TokensPrompt
        if inputs is None:
            start_time = time.time()
            assert prompt is not None
            if self.use_chat_template:
                chat = [format_chat(p, system_message=self.system_message) for p in prompt]
                inputs = [{'input_ids': self.tokenizer.apply_chat_template(c, tokenize=True, add_generation_prompt=True, max_length=self.max_length - self.generation_max_length, truncation=True, padding=True, return_tensors="pt")} for c in chat]
            else:
                inputs = [self.tokenizer(p, truncation=True, max_length=self.max_length - self.generation_max_length, return_tensors='pt') for p in prompt]
            logger.info(f"Finished preparing inputs for {len(inputs)} samples in {time.time() - start_time:.1f}s")

        idxs = []
        prompts = []
        final_outputs = {}
        for idx, i in enumerate(inputs):
            if i['input_ids'].size(1) > self.max_length - self.generation_max_length:
                final_outputs[idx] = None
            else:
                idxs.append(idx)
                prompts.append(TokensPrompt(prompt_token_ids=i['input_ids'][0].tolist()))

        start_time = time.time()
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=self._sampling_params(),
            **kwargs,
        )
        logger.info(f"Finished batch generation for {len(idxs)} samples in {time.time() - start_time:.1f}s")

        for idx, output in zip(idxs, outputs):
            final_outputs[idx] = {
                "output": output.outputs[0].text,
                "input_len": len(output.prompt_token_ids),
                "output_len": len(output.outputs[0].token_ids),
                "input_text": self._save_prompt(output.prompt_token_ids),
            }
        return [final_outputs[idx] for idx in range(len(inputs))]
