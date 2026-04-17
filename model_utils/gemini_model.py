import os
import functools

from model_utils.base import APIModel, call_api

import logging
logger = logging.getLogger(__name__)


class GeminiModel(APIModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        import vertexai
        vertexai.init()
        from vertexai.preview.tokenization import get_tokenizer_for_model
        self.model = genai.GenerativeModel(model_name)
        self.tokenizer = get_tokenizer_for_model(model_name)
        self.model_name = model_name
        if self.system_message is not None:
            logger.warning("system_message is not supported for GeminiModel")

    def prepare_inputs(self, test_item, data):
        prompt = data["prompt_template"].format(**test_item)
        buffer = 100
        inputs = self.tokenizer.compute_tokens(prompt).token_info_list()[0].tokens
        input_len = len(inputs)

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            new_context = self.tokenizer._sentencepiece_adapter._tokenizer.decode(
                self.tokenizer.compute_tokens(test_item["context"]).token_info_list()[0].token_ids[:-truncate_length]
            )
            test_item['context'] = new_context
            prompt = data["prompt_template"].format(**test_item)

        return prompt

    def generate(self, inputs=None, prompt=None, **kwargs):
        import google.generativeai as genai
        if inputs is None:
            inputs = prompt

        generation_config = genai.GenerationConfig(temperature=self.temperature, top_p=self.top_p, max_output_tokens=self.generation_max_length)
        func = functools.partial(
            self.model.generate_content,
            contents=inputs,
            generation_config=generation_config,
        )
        output = call_api(func, pause=15)
        if output is not None:
            try:
                output.text
            except Exception as e:
                logger.error(f"Error in output: {output}; {e}")
                return None

            return {
                "output": output.text,
                "input_len": output.usage_metadata.prompt_token_count,
                "output_len": output.usage_metadata.candidates_token_count,
                "input_text": inputs,
            }
        return None
