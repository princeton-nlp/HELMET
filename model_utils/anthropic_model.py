import time
import functools

from tqdm.contrib.concurrent import thread_map

from model_utils.base import APIModel, format_chat, call_api

import logging
logger = logging.getLogger(__name__)


class AnthropicModel(APIModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

        from anthropic import Anthropic, AnthropicVertex
        if "vertex" in model_name:
            self.model = AnthropicVertex()
            model_name = model_name[model_name.index("/") + 1:]
        else:
            self.model = Anthropic()

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file("claude.tokenizer.json")
        self.model_name = model_name
        if self.system_message is None:
            self.system_message = ""

    def prepare_inputs(self, test_item, data):
        if data.get("is_chat", False):
            return test_item['prompt']

        buffer = 100
        prompt = format_chat(data["user_template"].format(**test_item), system_message=None)
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            tokens = self.tokenizer.encode(test_item["context"])
            new_context = test_item["context"][:tokens.offsets[-truncate_length - 1][1]]
            test_item["context"] = new_context
            prompt = format_chat(data["user_template"].format(**test_item), system_message=None)
        return prompt

    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = format_chat(prompt, system_message=None)

        func = functools.partial(
            self.model.messages.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop_sequences=self.stops,
            system=self.system_message,
            **kwargs,
        )
        output = call_api(func, pause=20)

        if output is not None:
            return {
                "output": output.content[0].text,
                "input_len": output.usage.input_tokens,
                "output_len": output.usage.output_tokens,
                "input_text": inputs,
            }
        return None

    def batch_api(self, inputs, **kwargs):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        requests = []
        for idx, p in enumerate(inputs):
            requests.append(Request(
                custom_id=f"{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    messages=p,
                    max_tokens=self.generation_max_length,
                    temperature=self.temperature if self.do_sample else 0.0,
                    top_p=self.top_p,
                    stop_sequences=self.stops,
                    system=self.system_message,
                    **kwargs,
                )
            ))
        batch_job = self.model.messages.batches.create(requests=requests)

        while batch_job.processing_status not in ['succeeded', 'ended']:
            if batch_job.processing_status in ['errored', 'cancelled', 'expired']:
                raise Exception(f"Batch job {batch_job.id} failed: {batch_job.processing_status}")
            time.sleep(5)
            batch_job = self.model.messages.batches.retrieve(batch_job.id)
            logger.info(batch_job)

        outputs = [None for _ in inputs]
        for result in self.model.messages.batches.results(batch_job.id):
            if result.result.type == "succeeded":
                outputs[int(result.custom_id)] = {
                    "output": result.result.message.content[0].text,
                    "input_len": result.result.message.usage.input_tokens,
                    "output_len": result.result.message.usage.output_tokens,
                    "input_text": inputs[int(result.custom_id)],
                }

        return outputs

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        batch_file = kwargs.pop("batch_file", None)

        if batch_file:
            if inputs is None:
                inputs = [format_chat(p, system_message=None) for p in prompt]

            try:
                outputs = self.batch_api(inputs, **kwargs)
            except Exception as e:
                batch_size = 100
                logger.info(f"Error in batch generation: {e} with size {len(inputs)}, re-running with batch size {batch_size}")
                outputs = []
                for i in range(0, len(inputs), batch_size):
                    outputs.extend(self.batch_api(inputs[i:i + batch_size], **kwargs))
            return outputs

        return super().generate_batch(inputs=inputs, prompt=prompt, **kwargs)
