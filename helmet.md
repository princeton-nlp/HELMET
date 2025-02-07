<!-- # HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly -->
<h1 class="subtitle is-3 publication-subtitle">
  <span>HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly</span>
  <img src="./static/images/logo.jpeg" alt="logo" width="30"/>
</h1>

By [Howard Yen](howard-yen.github.io)\
2025-02-07

Paper: https://arxiv.org/abs/2410.02694 \
Code & Data: https://github.com/princeton-nlp/HELMET \
Website: https://princeton-nlp.github.io/HELMET



- [How to compare long-context language models?](#how-to-compare-long-context-language-models)
- [Constructing HELMET](#constructing-helmet)
  - [Key improvements over existing benchmarks](#key-improvements-over-existing-benchmarks)
- [Analysis](#analysis)
  - [Diverse long-context applications call for diverse evaluation](#diverse-long-context-applications-call-for-diverse-evaluation)
  - [Model performance across tasks and lengths](#model-performance-across-tasks-and-lengths)
- [Using HELMET for future developments](#using-helmet-for-future-developments)
  - [Diverse domains](#diverse-domains)
  - [Avoid running expensive baselines](#avoid-running-expensive-baselines)
  - [Future works](#future-works)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


From summarizing numerous legal documents to learning new tasks on the fly, long-context language models (LCLMs) have immense potential to change the way we use and interact with language models.
Traditionally, language models have been limited by their context window, which is typically around 2k tokens (e.g., [GPT-3](https://arxiv.org/abs/2005.14165)).
Recently, model developers are constantly increasing the context window of their models, with recent models like [GPT-4o](https://openai.com/index/hello-gpt-4o/), [Claude](https://www.anthropic.com/news/claude-3-family), and [Gemini](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#ethics-safety) supporting context windows of up to millions of tokens.

For downstream users, it may be difficult to choose the right model for their applications, as the evaluation settings between different models are *inconsistent* and often *not reflective of real-world applications*.
Furthermore, existing benchmarks for LCLMs may show confusing and counterintuitive results, making it difficult to understand the strengths and weaknesses of different models (Figure 1).

In this work, we propose HELMET (How to Evaluate Long-Context Models Effectively and Thoroughly), a comprehensive benchmark for evaluating LCLMs that improves upon existing benchmarks in several ways—*diversity, controllability, and reliability*.
We evaluate over 50 recent LCLMs and find that simple synthetic tasks, such as needle-in-a-haystack, do not reflect real-world performance, and it is crucial to evaluate model across diverse applications to understand their capabilities.

<figure>
  <img src="./static/images/teaser.png" alt="logo" width="800"/>
  <figcaption>Figure 1: existing benchmarks show counterintuitive trends such as smaller models outperforming larger ones.</figcaption>
</figure>

\
Since the initial release, model developers have adopted HELMET for evaluating their models, such as [Microsoft's Phi-4](https://arxiv.org/abs/2412.08905), and we hope that HELMET will be useful for future development of LCLMs.
In the following sections, we will describe the construction of HELMET, our findings, and how researchers and users may use HELMET to differentiate between different LCLMs.

<!-- We evaluate over 50 recent models on diverse, application-centric tasks, which enables researchers and practitioners to compare models across different axes. -->

<!-- However, existing benchmarks for long-context language modeling primarily rely on either perplexity and synthetic tasks, such as needle-in-a-haystack, even though it is unclear how well the performance on these tasks would transfer to real-world applications.  -->

<!-- In this work, we propose HELMET (How to Evaluate Long-Context Models Effectively and Thoroughly), a comprehensive benchmark for evaluating LCLMs.
In contrast to previous benchmarks, HELMET is designed to include diverse, application-centric tasks, complemented with reliable evaluation settings. 
We evaluate over 50 recent models, enabling detail comparisons and understanding of existing models and architectures across diverse axes.
Our experiments reveal key findings: (1) synthetic tasks like needle-in-a-haystack (NIAH) do not reflect real-world performance, (2) diverse types of tasks exhibit distinct trends, and (3) open-source models still lag behind proprietary models on more complex tasks.
Ultimately, we advocate for a holistic evaluation across diverse tasks. -->

## How to compare long-context language models?

With the development of LCLMs across both industry and the open-source community, it is crucial to have a reliable method for evaluating and comparing these models.
However, current models are *often evaluated on different benchmarks* (Table 1).

<figure>
  <img src="./static/images/model_eval_comparison.png" alt="logo" width="400"/>
  <figcaption>Table 1: Model developers often evaluate on different sets of datasets.</figcaption>
</figure>

\
A common practice for evaluating long-context language models is to use perplexity or synthetic tasks, such as needle-in-a-haystack (NIAH).
However, recent works have shown that perplexity does not correlate well with downstream performance ([Fang et al., 2024](https://arxiv.org/abs/2410.23771)).
In our work, we show that synthetic tasks like NIAH do not correlate with real-world performance (Figure 2).

<figure>
  <img src="./static/images/correlation_syn.png" alt="syn" width="500"/>
  <figcaption>Figure 2: Simple synthetic tasks, such as NIAH, do not correlate well with downstream tasks, such as summarization or generation with citations.</figcaption>
</figure>

\
Among the existing benchmarks with realistic applications, such as ZeroScrolls ([Shaman et al., 2023](https://arxiv.org/abs/2308.14508)), LongBench ([Bai et al., 2024](https://arxiv.org/abs/2308.14508)), and InfiniteBench ([Zhang et al., 2024](https://arxiv.org/abs/2402.13718)), there are still limitations:

- Insufficient coverage of downstream tasks: often focused on specific domains
- Inadequate lengths for testing frontier LCLMs: context lengths < 128k tokens
- Unreliable metrics: N-gram matching metrics like ROUGE are noisy
- Incompatibility with base models: require instruction-tuning

Thus, we propose HELMET to address these limitations and provide a comprehensive evaluation of LCLMs.

## Constructing HELMET

We design HELMET with the following desiderata:
1. Diverse coverage of downstream tasks
2. Controllable length and complexity
3. Reliable evaluation for base and instruction-tuned models

We show an overview of the benchmark in Table 2.

<figure>
  <img src="./static/images/data_summary.png" alt="logo" width="800"/>
  <figcaption>Table 2: Overview of HELMET datasets.</figcaption>
</figure>

### Key improvements over existing benchmarks

***Reliable evaluation***: Many existing benchmarks still use n-gram based metrics, such as ROUGE, despite their poor correlation with human judgments ([Goyal et al., 2023](https://arxiv.org/abs/2209.12356)). We employ model-based evaluations that show better distinguishability between models and different input lengths (Figure 3). Furthermore, our human studies show that our metrics have high agreement with human judgments.

<figure>
  <img src="./static/images/model_eval.png" alt="logo" width="800"/>
  <figcaption>Figure 3: ROUGE cannot differentiate between models and lengths while model-based evaluations are better at separating models of different capacities.</figcaption>
</figure>

\
***Robust prompting***: Existing long-context benchmarks often require models to follow instructions, but many model developments revolve around base models, which has to rely on synthetic tasks or perplexity for evaluation.
Thus, we support base models for a subset of our tasks via in-context learning examples. This substantially improves the performance of base models, which is more reflective of real-world applications.


***Controllable length and difficulty***:
An important dimension to consider when evaluating LCLMs is the input length, as longer inputs can provide more information while challenging the model's ability to process noisy contexts.
In our tasks, we can control the input length by changing the number of retrieved passages (RAG, Cite, Re-rank), the number of demonstrations (ICL), or the length of the input document (LongQA, Summ).
Although LongQA and Summ can not be easily extended to longer contexts, we intentionally chose datasets with natural documents of length far greater than 100k tokens, such that they can still be used to evaluate frontier LCLMs.


## Analysis

<!-- With the release of numerous long-context language models, it may be difficult discerning their differences given the stark difference in the evaluation settings.
Furthermore, it's often unclear how these models will perform on specific applications, as most benchmarks often focus on synthetic tasks or specific applications.
In this work, we evaluate a comprehensive set of 51 models on HELMET.
To our best knowledge, this is the most thorough and controlled comparison of long-context models on diverse applications.
These models cover both leading proprietary and open-source models.
We also consider models with different architectures (e.g., full-attention transformers, hybrid architectures) and positional extrapolation techniques.
In this section, we will highlight a few key findings from our experiments.

### Simple synthetic tasks are poor predictors of real-world performance

One of the most popular task for evaluating LCLMs is the needle-in-a-haystack (NIAH) task and its many variants.
However, there has been no systematic evaluation of how well these synthetic tasks correlate with real-world performance.
In the following figure, we plot their Pearson rank correlation with real-world tasks in HELMET.

The most simple synthetic task—NIAH—has the lowest correlation with real-world tasks, while the more complex variants (e.g., RULER MK) have higher correlation.
The tasks with noisier, more distracting contexts are better at differentiating models, as the performance are less likely to be concentrated at 100% accuracy.
In contrast, a more realistic task, such as RAG, has much higher correlation with the real-world tasks.
Thus, **RAG is a better proxy for real-world tasks**.  -->

Our experiments and analysis include a comprehensive set of 59 LCLMs. To our knowledge this is the most thorough and controlled comparison of long-context models on diverse applications.
These models cover both leading proprietary and open-source models, and we also consider models with different architectures (e.g., full-attention transformers, hybrid architectures) and positional extrapolation techniques.
In this section, we will highlight a few key findings from our experiments.


### Diverse long-context applications call for diverse evaluation

Long-context benchmarks are often constructed with specific applications in mind, such as summarization or question answering, which limits the understanding of LCLMs in a broader context.
We examine model performance over a wide range of real tasks, and find that different categories do not always correlate with each other (Figure 4).

<figure>
   <img src="./static/images/correlation_category_inst.png" alt="syn" width="500"/>
   <figcaption>Figure 4: Different categories do not correlate well with each other.</figcaption>
</figure>

\
While some tasks moderately correlate with each other (e.g., RAG and MS-MARCO) due to their retrieval-based nature, others show little correlation (e.g., Summ and Cite).
Notably, ICL has the lowest correlation with other tasks, which suggests that it is a unique task that requires different capabilities from the model.
Therefore, model developers should evaluate across these distinct axes to draw a more holistic picture of the model's capabilities.


### Model performance across tasks and lengths

We present the results of the frontier proprietary models as well as a few open-source models on HELMET.
Additional results can be found in the paper and the website.

<figure>
   <img src="./static/images/results_length_main.png" alt="syn" width="800"/>
   <figcaption>Figure 5: HELMET results on selected instruction-tuned models.</figcaption>
</figure>

\
First, we observe that **open-source models lag behind closed-source models on complext tasks**.
Although the gap appears to small on more simple tasks, such as Recall, the gap widens on more complex ones, such as Cite.

Furthermore, **performance degradation with increasing lengths is category-dependent**. Even the most advanced models, such as GPT-4o and Gemini, experience significant decrease in performance on tasks like re-ranking. 
This change in performance cannot be observed from simply looking at the synthetic task performance.

Finally, **there is no clear winner across all categories**, thereby calling for evaluation across different axes.
Additional analysis, such as performance of different positional extrapolation methods and the lost-in-the-middle phenomenon, can be found in the paper.


## Using HELMET for future developments

### Diverse domains
With HELMET, practitioners can easily choose the right model for their applications by comparing models across diverse tasks. Given the increasing interest in LCLMs for both applications and other research fields, we hope that HELMET will be useful tool for the community.

### Avoid running expensive baselines
It is often expensive to run all the baselines for evaluating LCLMs, especially at long contexts given its computational and memory costs. 
For example, running the HELMET at all lengths on a 70B model requires a node with 8 * 80GB GPUs for hundreds of GPU hours, which can be costly.
By evaluating on HELMET, researchers can directly compare their models to existing ones simply by referencing our results, which covers 59 models of different sizes and architectures.


### Future works 

HELMET is a step towards a more comprehensive evaluation of long-context language models, but there are still many more exciting applications of LCLMs.
For example, we recently released [LongProc](https://arxiv.org/abs/2501.05414), a benchmark for evaluating LCLMs on *long-form generation* and *following procedures*.
Although summarization tasks have long outputs (up to 1k tokens), LongProc focuses on even longer outputs, up to 8k tokens.
Similar to HELMET, LongProc is also designed with reliable evaluation settings and diverse tasks.
We are working on integrating LongProc into HELMET's evaluation suite, and we hope that this will provide a more comprehensive evaluation of LCLMs on long-form tasks.

## Acknowledgements

We thank Mengzhou Xia, Howard Chen, Xi Ye, Yinghui He, Lucy He, Alexander Wettig, Sadhika Malladi, Adithya Bhaskar, and Joie Zhang for their helpful feedback.
We also thank the Microsoft Accelerate Foundation Models Research (AFMR) for Azure OpenAI credits.

## Citation

If you find HELMET useful, please consider citing our paper:

```
@inproceedings{yen2024helmetevaluatelongcontextlanguage,
      title={HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly}, 
      author={Howard Yen and Tianyu Gao and Minmin Hou and Ke Ding and Daniel Fleischer and Peter Izsak and Moshe Wasserblat and Danqi Chen},
      year={2025},
      booktitle={International Conference on Learning Representations (ICLR)},
}
```