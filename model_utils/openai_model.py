import time
import json
import functools

from tqdm.contrib.concurrent import thread_map

from model_utils.base import LLM, APIModel, format_chat, call_api

import logging
logger = logging.getLogger(__name__)


class OpenAIModel(APIModel):
    def __init__(self, model_name, seed=42, **kwargs):
        super().__init__(model_name, **kwargs)

        import openai
        import tiktoken
        if "azure" in model_name:
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/") + 1:]
        else:
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.seed = seed
        self.API_MAX_LENGTH = 128000

    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if self.max_length > self.API_MAX_LENGTH:
            logger.warning(f"max_length {self.max_length} > {self.API_MAX_LENGTH}, clamping")
            self.max_length = self.API_MAX_LENGTH

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            new_context = self.tokenizer.decode(self.tokenizer.encode(test_item["context"])[:-truncate_length])
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
            seed=self.seed,
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
                "system_fingerprint": output.system_fingerprint,
            }
        return None

    def batch_api(self, inputs, batch_file, **kwargs):
        with open(batch_file, "w") as f:
            for idx, p in enumerate(inputs):
                f.write(json.dumps({
                    "custom_id": f"{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": p,
                        "max_tokens": self.generation_max_length,
                        "temperature": self.temperature if self.do_sample else 0.0,
                        "top_p": self.top_p,
                        "stop": self.stops,
                        "seed": self.seed,
                        **kwargs,
                    }
                }) + "\n")
        upload_file = self.model.files.create(file=open(batch_file, "rb"), purpose="batch")
        batch_job = self.model.batches.create(input_file_id=upload_file.id, endpoint="/v1/chat/completions", completion_window='24h')
        logger.info(f"Starting batch job: {batch_job.id}")

        while batch_job.status != "completed":
            if batch_job.status in ['failed', 'expired', 'cancelled']:
                raise Exception(f"Batch job {batch_job.id} failed: {batch_job.status}")
            time.sleep(5)
            batch_job = self.model.batches.retrieve(batch_job.id)
            logger.info(batch_job)

        result_file_id = batch_job.output_file_id
        result = self.model.files.content(result_file_id).content
        outputs = [None for _ in inputs]
        with open(batch_file + ".result", "wb") as f:
            f.write(result)

        for line in result.decode("utf-8").strip().split("\n"):
            output = json.loads(line)
            task_id = int(output["custom_id"])
            res = output["response"]['body']
            if res["choices"][0]["message"]["content"] is not None:
                outputs[task_id] = {
                    "output": res["choices"][0]["message"]["content"],
                    "input_len": res["usage"]["prompt_tokens"],
                    "output_len": res["usage"]["completion_tokens"],
                    "input_text": inputs[task_id],
                    "system_fingerprint": res["system_fingerprint"],
                }

        return outputs

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        batch_file = kwargs.pop("batch_file", None)
        if batch_file:
            logger.info(f"Using {batch_file} for batch generation")
            if inputs is None:
                inputs = [format_chat(p, system_message=self.system_message) for p in prompt]

            try:
                outputs = self.batch_api(inputs, batch_file, **kwargs)
            except Exception as e:
                batch_size = 100
                logger.info(f"Error in batch generation: {e} with size {len(inputs)}, re-running with batch size {batch_size}")
                outputs = []
                for i in range(0, len(inputs), batch_size):
                    outputs.extend(self.batch_api(inputs[i:i + batch_size], batch_file, **kwargs))
            return outputs

        return super().generate_batch(inputs=inputs, prompt=prompt, **kwargs)


class TgiVllmModel(OpenAIModel):
    def __init__(self, model_name, seed=42, **kwargs):
        # Skip OpenAIModel.__init__ — we set up the client differently
        LLM.__init__(self, model_name, **kwargs)

        from openai import OpenAI
        from transformers import AutoTokenizer

        endpoint_url = kwargs["endpoint_url"]
        logger.info(f"Endpoint URL: {endpoint_url}")

        self.model = OpenAI(
            base_url=endpoint_url,
            api_key=kwargs["api_key"],
        )
        if "tgi" in model_name:
            model_name = model_name[model_name.index(":") + 1:]
            self.model_name = "tgi"
        else:
            self.model_name = model_name
        logger.info(f"Model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seed = seed
        self.API_MAX_LENGTH = float('inf')

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        # TGI/vLLM serving uses threaded generation, no batch API
        return APIModel.generate_batch(self, inputs=inputs, prompt=prompt, **kwargs)
