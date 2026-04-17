from typing import Optional, Dict, Any

from transformers import PreTrainedTokenizer

from model_utils.base import LLM, format_chat, merge_user_messages

import logging
logger = logging.getLogger(__name__)


def tokenize(
    sample: Dict[str, Any],
    data: Dict[str, Any],
    tokenizer,
    max_length: int,
    generation_max_length: int,
    is_chat: bool = False,
    use_chat_template: bool = False,
    continue_final_message: bool = False,
    system_message: Optional[str] = "You are a helpful assistant.",
):
    if is_chat:
        try:
            ids = tokenizer.apply_chat_template(sample['prompt'], return_tensors="pt", add_generation_prompt=True)
        except Exception:
            prompt = sample['prompt']
            for p in prompt:
                if p['role'] == 'system':
                    p['role'] = 'user'
            try:
                ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
            except Exception:
                prompt = merge_user_messages(prompt)
                ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        return {"input_ids": ids, "original_text": sample['prompt']}

    if continue_final_message:
        assert use_chat_template

    def format_input(sample):
        if use_chat_template:
            chat = format_chat(
                data["user_template"].format(**sample),
                system_message=system_message,
            )
            if continue_final_message:
                chat.append({"role": "assistant", "content": data['system_template'].format(**sample)})
            try:
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=not continue_final_message, continue_final_message=continue_final_message)
            except Exception:
                chat = format_chat(data["user_template"].format(**sample), system_message=None)
                if continue_final_message:
                    chat.append({"role": "assistant", "content": data['system_template'].format(**sample)})
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=not continue_final_message, continue_final_message=continue_final_message)
            tokenized_input = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = data["prompt_template"].format(**sample)
            tokenized_input = tokenizer([prompt], return_tensors="pt")
        return tokenized_input

    if "Phi3SmallTokenizer" in str(type(tokenizer)) and max_length == 131072:
        buffer = 64
    else:
        buffer = 4

    tokenized_input = format_input(sample)
    if tokenized_input.input_ids.size(1) > max_length - generation_max_length - buffer:
        truncate_length = tokenized_input.input_ids.size(1) - (max_length - generation_max_length - buffer)

        if isinstance(tokenizer, PreTrainedTokenizer) and not tokenizer.is_fast:
            context_tokens = tokenizer(sample["context"])
            new_context = tokenizer.decode(context_tokens["input_ids"][:-truncate_length])
        else:
            context_tokens = tokenizer([sample["context"]], return_offsets_mapping=True)
            new_context = sample["context"][:context_tokens["offset_mapping"][0][-truncate_length][0]]

        sample["context"] = new_context
        tokenized_input = format_input(sample)
    return tokenized_input


class LocalModel(LLM):
    """Base for models with a local tokenizer (HF, vLLM, SGLang)."""

    tokenizer = None

    def prepare_inputs(self, test_item, data):
        return tokenize(
            test_item,
            data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            generation_max_length=self.generation_max_length,
            is_chat=data.get("is_chat", False),
            use_chat_template=self.use_chat_template,
            system_message=self.system_message,
        )
