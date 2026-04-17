import torch
from transformers import set_seed

from model_utils.base import format_chat
from model_utils.local import LocalModel

import logging
logger = logging.getLogger(__name__)


class HFModel(LocalModel):
    def __init__(self, model_name, seed=42, **kwargs):
        super().__init__(model_name, **kwargs)
        set_seed(seed)

        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        model_kwargs = {}
        from pkg_resources import parse_version
        if parse_version(transformers.__version__) <= parse_version("4.34.1"):
            model_kwargs["use_flash_attention_2"] = True
        else:
            model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")

        FLASH_ATTN_NOT_SUPPORTED = ["recurrentgemma", "yarn"]
        if any([x in model_name.lower() for x in FLASH_ATTN_NOT_SUPPORTED]):
            model_kwargs = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if kwargs.get("rope_theta") is not None:
            logger.info(f"Override rope theta to {kwargs['rope_theta']}")
            config.rope_theta = kwargs["rope_theta"]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            dtype=kwargs.get("dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs,
        )
        if kwargs.get("torch_compile", True):
            self.model = torch.compile(self.model)

        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if self.stops:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(s) for s in stop] + stop_token_ids))
            if "llama" in model_name.lower():
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device
        self.disable_prefill = False

        if "gemma" in model_name.lower():
            self.disable_prefill = True
            logger.warning("gemma models cannot prefill with past kvs due to cache implementation")

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            assert prompt is not None
            if self.use_chat_template and isinstance(prompt, str):
                chat = format_chat(prompt, system_message=self.system_message)
                inputs = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", max_length=self.max_length - self.generation_max_length, truncation=True, padding=True)
            else:
                inputs = self.tokenizer([prompt], return_tensors="pt", max_length=self.max_length - self.generation_max_length, truncation=True, padding=True)

        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        if hasattr(self.model, "model") and not self.disable_prefill:
            from transformers import BatchEncoding
            extra = {}
            if "jamba" in str(type(self.model)).lower():
                from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache
                cache = HybridMambaAttentionDynamicCache(self.model.config, inputs.input_ids.shape[0], self.model.dtype, device=self.model.device)
                extra = {"past_key_values": cache}

            prefill = self.model.model(input_ids=inputs.input_ids[..., :-1], attention_mask=inputs.attention_mask[..., :-1], **extra)
            past_key_values = prefill.past_key_values
            if past_key_values is None:
                self.disable_prefill = True
                logger.warning("past key values is None, disabling prefill...")
            else:
                inputs = BatchEncoding({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "past_key_values": past_key_values})

        gen_kwargs = {
            "max_new_tokens": self.generation_max_length,
            "do_sample": self.do_sample,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        outputs = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)

        save_prompt = self.tokenizer.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.tokenizer.decode(inputs["input_ids"][0][-500:])
        output_len = outputs['sequences'].size(1) - input_len
        del inputs, outputs

        return {
            "output": text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }
