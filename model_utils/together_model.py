import functools

from model_utils.base import APIModel, format_chat, call_api

import logging
logger = logging.getLogger(__name__)


class TogetherModel(APIModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

        from transformers import AutoTokenizer
        from together import Together
        self.model = Together()
        self.model_name = model_name.replace("togetherapi/", "")
        name_mapping = {
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "deepseek-ai/DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(name_mapping[self.model_name])

    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        tokens = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        input_len = len(tokens)

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            context_tokens = self.tokenizer(test_item["context"], return_offsets_mapping=True)
            new_context = test_item["context"][:context_tokens["offset_mapping"][-truncate_length][0]]

            test_item["context"] = new_context
            prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        return prompt

    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = format_chat(prompt, system_message=self.system_message)

        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None
