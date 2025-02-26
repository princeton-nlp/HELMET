# Changelog
All notable changes will be documented in this file.

## 2025-02-25

In this version, we make some significant improvements to reduce the cost of running the experiments.

- Add support for batch evaluation
    - For OpenAI and Anthropic, we use their batch API to reduce the cost of API calls by 50% ([OpenAI documentation](https://cookbook.openai.com/examples/batch_processing), [Anthropic documentation](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)). The model-based evaluation script have also been updated to reduce cost.
    - For other API providers, we use a simple multi-threading approach to parallelize the API calls
    - For open-source models, we use batching from the VLLM library for more speed-up.
- Changes to the datasets pre-processing — the paper will be updated in a future version.
    - ICL datasets now evaluate 500 samples instead of 100, use a different set of demonstrations for each test instance, and we balance the number of test labels—this is to make the evaluation more consistent and robust. 
    - RAG, Re-ranking, and Citation use `hashlib` for consistent hashing
- Visualization jupyter notebook for plotting results.
- Support for SGLang, which can be faster for certain supported models.
- Support for reasoning models, such as DeepSeek's R1 models, where we parse out the reasoning steps from the model's output.
- Other minor changes, such as adding documentation.

## 2024-10-04

Thanks to @8188zq and @chtmp223 for pointing out some issues in the current repo: some results are not fully reproducible due to random seeding problems.
This affects the results for ICL, Re-reranking, and the RAG tasks, where the demo sample for each question may differ.
We have updated the code to fix this issue, and will update the results on the paper and the spreadsheet soon to reflect the changes.
Specifically, we make sure the seeding is consistent across runs and independent of system settings.

Other minor changes:
 - Clean up `data.py` and remove unused code.
 - Update argument descriptions.
 - Log exceptions