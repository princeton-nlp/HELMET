# HELMET

How to Evaluate Long-context Language Models Effectively and Thoroughly

---

[Paper](https://arxiv.org/abs/2410.02694)
HELMET is a comprehensive benchmark for long-context language models covering a seven diverse categories of tasks.
The datasets are application-centric and are designed to evaluate models at different lengths and levels of complexity.
Please check out the paper for more details, and this repo will detail how to run the evaluation.

## Quick Links

- [Setup](#setup)
- [Data](#data)
- [Running evaluation](#running-evaluation)
- [Adding new tasks](#adding-new-tasks)
- [Adding new models](#adding-new-models)
- [Contacts](#contacts)
- [Citation](#citation)

## Release Progress

- [x] HELMET Code
- [ ] HELMET data
- [ ] Retrievel setup
- [ ] VLLM Support


## Setup

Please install the necesary packages with
```bash
pip install -r requirements.txt
```

Additionally, if you wish to use the API models, you will need to install the package corresponding to the API you wish to use
```bash
pip install openai # OpenAI API
pip install anthropic # Anthropic API
pip install google-generativeai # Google GenerativeAI API
pip install together # Together API
```

## Data

Data will be uploaded soon :)
In the mean time, please contact me to get access

## Running evaluation

To run the evaluation, simply use one of the config files in the `configs` directory, you may also overwrite any arguments in the config file or add new arguments simply through command line:
```bash
python eval.py --config configs/cite.yaml --model_name_or_path {local model path or huggingface model name} --output_dir {output directory}
```


## Adding new tasks

To add a new task/dataset, you just need to modify the `data.py` file:

Create a function that specify how to load the data:
1. Specify the string templates for the task through `user_template`, `system_template`, and `prompt_template` (which is usually just the concatenation of the two)
2. Process each sample to fit the specified templates (the tokenization code will call `user_template.format(**test_sample)` and same for `system_template`). Importantly, each sample should have a `context` field, which will be truncated automatically if the input is too long (e.g., for QA, this is the retrieved passasges; for NarrativeQA, this is the book/script). You should use the `question` and `answer` field to make evaluation/printing easier.
3. Optionally, add a `post_process` function to process the model output (e.g., for MS MARCO, we use a ranking parse function; for RULER, we calculate the recall). There is also a `default_post_process` function that parses and calculate simple metrics like EM and F1 that you may use. This function should take in the model output and the test sample and return a tuple of `(metrics, changed_output)`, the `metrics` (e.g., EM, ROUGE) are aggregated across all samples, and the `changed_output` are added to the test_sample and saved to the output file.
4. The function should return `{'data': [list of data samples], 'prompt_template': prompt_template, 'user_template': user_template, 'system_template': system_template, 'post_process': [optional custom function]}`.

Finally, simply add a new case to the `load_data` function that calls the function that you just wrote to load your data.
You can refer to the existing tasks for examples (e.g., `load_json_kv`, `load_narrativeqa`, and `load_msmarco_rerank`).

## Adding new models

The existing code supports using HuggingFace-supported models and API models (OpenAI, Anthropic, Google, and Together). To add a new model or use a different framework (e.g., VLLM), you can modify the `model_utils.py` file.
Specifically, you need to create a new class that implements `prepare_inputs` (how the inputs are processed) and `generate` functions. Then, you can add a new case to `load_LLM`.
Please refer to the existing classes for examples.

## Contacts

If you have any questions, please email me at `hyen@cs.princeton.edu`.
If you encounter any problems, you also also open an issue here. Please try to specify the problem with details so we can help you better and quicker!

## Citation

If you find our work useful, please cite us:
```
@misc{yen2024helmetevaluatelongcontextlanguage,
      title={HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly}, 
      author={Howard Yen and Tianyu Gao and Minmin Hou and Ke Ding and Daniel Fleischer and Peter Izasak and Moshe Wasserblat and Danqi Chen},
      year={2024},
      eprint={2410.02694},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02694}, 
}
```