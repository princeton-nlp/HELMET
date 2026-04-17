import json

from pydantic import BaseModel, Field

from datasets import load_dataset
from transformers import AutoTokenizer

from utils import calculate_metrics, parse_output
from data_module.base import register_task

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maps length suffix to token count
_LENGTH_VALUES = {
    "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
    "64k": 65536, "128k": 131072,
}

# Minimum document length in llama2 tokens to be included
_MIN_DOC_LENGTH = 65536
TRUNCATE_TOKENIZER = AutoTokenizer.from_pretrained("/scratch/gpfs/PLI/models/Llama-3.1-8B")

# Buffer for prompt instructions (template overhead)
_PROMPT_OVERHEAD = 300

_TRUNCATION_POSTFIX = " ... [the rest of the text is omitted]"

# ---------------------------------------------------------------------------
# Judge prompts (from scripts/eval_gpt4_summ.py)
# ---------------------------------------------------------------------------

FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
  - Examples:
    - Incomplete: "Summary:"
    - Incoherent: "Summary: The plaintiff the the the the able the the the the the the the the the the able the the the the the Ã�\n"
    - Repetitive: "Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants."

- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.
  - Examples:
    - "This case is about an apprenticeship test that had a disparate impact on Black apprenticeship applicants. The Equal Employment Opportunity Commission (EEOC) filed this lawsuit on December 27, 2004, in U.S. District Court for the Southern District of Ohio."
    - "The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Now, read the provided text, and evaluate the fluency using the rubric.

Text:<begin_of_text>{text}</begin_of_text>
"""

RECALL_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a civil lawsuit. The summary is based on a set of legal documents, and it should contain a short description of the background, the parties involved, and the outcomes of the case. The text should contain all the major points in the expert-written summary, which are given to you.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is well-supported by the provided summary.
- Score: the number of key points present in the provided summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Key points:
1. The case challenged curfews in Los Angeles and San Bernardino, California.
2. The curfews were issued in response to the nationwide protests following the police killing of George Floyd in Minneapolis.
3. The complaint argued that the curfews violated free speech, free assembly, free movement, and Due Process.
4. The complaint also argued that the San Bernardino curfew violated the Establishment Clause.
5. The complaint sought injunctive and declaratory relief.
6. The plaintiffs voluntarily dismissed the case on July 7, 2020.
7. The dismissal occurred because the city had rescinded the curfews and not attempted to reinstate them.

Summary: "In June 2020, Black Lives Matter - Los Angeles and several individuals filed a lawsuit in the U.S. District Court for the Central District of California against Los Angeles Mayor Eric Garcetti, other city officials, and the City of San Bernardino, challenging the constitutionality of curfew orders imposed during protests against police violence. The plaintiffs, represented by the ACLU of Southern California, argued that the curfews violated their First Amendment rights to free speech and assembly, as well as their freedom of movement, by suppressing political protests and other activities. The lawsuit also claimed that the curfews were not narrowly tailored to address any emergency and lacked sufficient notice. However, the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks."

Reasoning: The summary states that the plaintiffs challenged the constitutionality of curfew orders against Los Angeles and San Bernadino, so key point 1 is present. The summary does not mention that the curfew orders were issued in response to the nationwide protest that resulted from the police killing of George Floyd in Minneapolis, so key point 2 is missing. The summary does mention that the complaint argued that the curfews violated the First Amendment rights to free speech and assembly, so key point 3 is present. The summary does not mention that the complaint argued that the San Bernardino curfew violated the Establishment Clause, so key point 4 is missing. The summary does not mention that the complaint sought injunctive and declaratory relief, so key point 5 is missing. The summary mentions that the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks, so key point 6 and 7 are present. Finally, key points 1, 3, 6, and 7 are present in the summary, so the recall score is 4.

Output: {{"recall": 4}}


Example 2:

Key points:
1. Individuals with disabilities brought the case against various Illinois state officials.
2. The plaintiffs sought declaratory and injunctive relief, alleging inappropriate institutionalization when community-based care was possible.
3. In August 2011, a consent decree was entered, requiring the state to transition class members from nursing facilities to community-based settings.
4. The transition plan was updated in April 2018.
5. Monitoring of the transition is ongoing as of November 2018.

Summary: "Summary: Five Medicaid-eligible individuals with disabilities, Lenil Colbert, Constance Gray, Ernest Reeves, Kenya Lyles, and Dwight Scott, filed a class action lawsuit in the United States District Court for the Northern District of Illinois against Illinois state officials, including Governor Rod R. Blagojevich, Secretary of the Illinois Department of Human Services Carol L. Adams, Director of the Illinois Department of Healthcare and Family Services Barry S. Maram, and Director of the Illinois Department of Public Health Eric E. Whitaker. The plaintiffs alleged that the defendants' policies and practices effectively compel people with disabilities to enter nursing facilities in order to receive long-term care and assistance, forcing them to forego liberty, privacy, independence, and the opportunity to live in the communities of their choice. The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Reasoning: The summary states that the plaintiffs brought the case against various Illinois state officials, so key point 1 is present. The summary mentions that "the plaintiffs sought declaratory and injunctive relief" and the practices "compelled people with disabilities to enter nursing facilities... to forego ... the opportunity to live in the communities of their choice", so key point 2 is present. The summary does not mention that a consent decree was entered in August 2011, so key point 3 is missing. The summary does not mention that the transition plan was updated in April 2018, so key point 4 is missing. The summary does not mention that monitoring of the transition is ongoing as of November 2018, so key point 5 is missing. Therefore, key points 1 and 2 are present so the recall score is 2.

Output: {{"recall": 2}}

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"recall": 2}}.


Key points:
{keypoints}

Summary: "{summary}"
"""

PRECISION_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a civil lawsuit. The summary is based on a set of legal documents, and it should contain a short description of the background, the parties involved, and the outcomes of the case.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary. A sentence is considered supported if its major facts align with the information in the expert summary. A sentence is still considered supported even if some of its minor details, such as dates, entity names, or the names of laws and previous court cases, are not explicitly mentioned in the expert summary. A sentence is not supported if its major facts are not mentioned or contradicted in the expert summary.
- Score: the number of sentences in the provided summary that are supported by the expert summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Expert summary: "This lawsuit, brought in the the U.S. District Court for the Central District of California, was filed on June 3, 2020. The plaintiffs were represented by attorneys from the ACLU of Southern California. This lawsuit followed nation-wide protests that occurred in response to the killing of George Floyd by a police officer in Minneapolis. While most protests were peaceful, some ended in violence, property destruction, rioting, and looting. Many cities, including Los Angeles and San Bernardino, issued curfews in an attempt to quell these riots. This action challenged these curfews as violations of free speech and assembly, free movement, due process, and challenged the San Bernardino curfew as a violation of the establishment clause (the San Bernardino curfew included a provision that exempted attendants of religious meetings from the curfew.) The plaintiffs sought injunctive and declaratory relief that would void the curfew and prohibit the cities from enforcing them. The following day, June 4th, 2020, the case was assigned to District Judge Philip S. Gutierre and to Magistrate Judge Pedro V. Castillo. Judge Gutierrez informed the parties that he was part of a mandatory alternative dispute resolution (ADR) program and asked the parties to try to form an agreement before going to trial. On July 7, 2020, the plaintiffs voluntarily dismissed the complaint, citing that fact that the city had rescinded the curfews already and not attempted to reinstate them. The case is now closed."

Provided summary: "In June 2020, Black Lives Matter - Los Angeles and several individuals filed a lawsuit in the U.S. District Court for the Central District of California against Los Angeles Mayor Eric Garcetti, other city officials, and the City of San Bernardino, challenging the constitutionality of curfew orders imposed during protests against police violence. The plaintiffs, represented by the ACLU of Southern California, argued that the curfews violated their First Amendment rights to free speech and assembly, as well as their freedom of movement, by suppressing political protests and other activities. The lawsuit also claimed that the curfews were not narrowly tailored to address any emergency and lacked sufficient notice. However, the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks."

Reasoning: The first sentence in the provided summary is well supported by the expert summary even though some entity names are not explicitly mentioned. The second sentence is also well supported by the expert summary, as it mentions the ACLU of Southern California and the First Amendment rights. The third sentence is not supported by the expert summary, as it does not mention the lack of narrow tailoring or sufficient notice. The fourth sentence is well supported by the expert summary, as it mentions the voluntary dismissal of the case in July 2020. Therefore, the precision score is 3.

Output: {{"precision": 3, "sentence_count": 4}}


Example 2:

Expert summary: "On August 22, 2007, individuals with disabilities filed a lawsuit under the Americans with Disabilities Act (ADA), the Social Security Act, the Rehabilitation Act, and the Nursing Care Reform Act, against various Illinois state officials in the United States District Court for the Northern District of Illinois.  Plaintiffs, represented by private and public interest counsel, asked the court for declaratory and injunctive relief, claiming that they were institutionalized in a nursing facility even though they were capable of living in a more community-integrated setting with appropriate services.  Plaintiffs claimed that Defendants conditioned receipt of long-term care on remaining in an institutionalized setting, even though it would be less expensive for Plaintiffs to receive appropriate care in the community. The Court (Judge Joan H. Lefkow) certified a class as: \"all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and who, with appropriate supports and services, may be able to live in a community setting.\"  71 Fed.R.Serv.3d 1089. At a status hearing on January 7, 2011, the parties advised Magistrate Judge Maria Valdez that they could conclude settlement discussions without further assistance from the court. On Aug. 29, 2011, the parties jointly moved for the court to approve the consent decree they had agreed upon.  The court held a fairness hearing on Dec. 20, 2011, and ultimately accepted the decree. The consent decree established benchmarks for moving specific numbers of class members out of nursing facilities and into community-based settings. Over the course of the first two-and-a-half years, the decree compelled the state to move 1,100 class members into the community. It also required the state to provide up to $10 million in housing assistance to support the first group of transitioned adults. The decree also compelled the state to develop services needed to adequately support class members who choose to live in the community. It established a monitor to ensure compliance with the decree, and granted $1.2 million in attorneys' fees. The court approved an updated plan following the parties' cross-motion to enter into a cost-neutral plan and supplement and amend the December 2011 consent decree on November 16, 2016. The plan included the transition of class members into community-based settings, and continued evaluations and service plans for the class members. The court retained jurisdiction to oversee the full implementation of the plan. The court approved an updated plan on April 5, 2018. Monitoring by the court appointed monitor (Gail P. Hutchings) is ongoing as of May 20, 2020."

Provided: "Summary: Five Medicaid-eligible individuals with disabilities, Lenil Colbert, Constance Gray, Ernest Reeves, Kenya Lyles, and Dwight Scott, filed a class action lawsuit in the United States District Court for the Northern District of Illinois against Illinois state officials, including Governor Rod R. Blagojevich, Secretary of the Illinois Department of Human Services Carol L. Adams, Director of the Illinois Department of Healthcare and Family Services Barry S. Maram, and Director of the Illinois Department of Public Health Eric E. Whitaker. The plaintiffs alleged that the defendants' policies and practices effectively compel people with disabilities to enter nursing facilities in order to receive long-term care and assistance, forcing them to forego liberty, privacy, independence, and the opportunity to live in the communities of their choice. The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Reasoning: The first sentence is supported as the expert summary states that "individuals with disabilities filed a lawsuit... against various Illinois state officials", even though some minor details (the name of the people) are not mentioned. The second sentence is not supported as the expert summary does not discuss how the plaintiffs alleged that the defendants' policies forced them to forego their rights. The third sentence is mostly supported as the expert summary mentions that the plaintiffs sought declaratory and injunctive relief, but it does not mention the attorneys' fees and costs, which are minor details. The fourth sentence is supported as the expert summary mentions the class action certification by the court. The fifth sentence is not supported as the expert summary does not mention the defendants' denial of the allegations. The sixth sentence is not supported as the expert summary states that the case was settled through the consent decree, while the provided summary states that the case is ongoing. Therefore, the precision score is 3.

Output: {{"precision": 2, "sentence_count": 6}}

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 2, "sentence_count": 6}}.

Expert summary:<begin_of_text>{expert_summary}</begin_of_text>

Provided summary:<begin_of_text>{summary}</begin_of_text>
"""


# ---------------------------------------------------------------------------
# Structured output models for judge responses
# ---------------------------------------------------------------------------

class FluencyResponse(BaseModel):
    explanation: str = Field(description="Brief reasoning for the fluency score")
    fluency: int = Field(ge=0, le=1, description="0=incoherent/repetitive/incomplete, 1=coherent and fluent")


class RecallResponse(BaseModel):
    explanation: str = Field(description="Brief reasoning for the recall score")
    recall: int = Field(ge=0, description="Number of key points present in the summary")


class PrecisionResponse(BaseModel):
    explanation: str = Field(description="Brief reasoning for the precision score")
    precision: int = Field(ge=0, description="Number of supported sentences in the summary")
    sentence_count: int = Field(ge=0, description="Total number of sentences in the summary")


# ---------------------------------------------------------------------------
# Judge client (lazy-initialized)
# ---------------------------------------------------------------------------

_judge_client = None


def _call_judge(prompt, response_format):
    """Call GPT-4o to judge a single response with structured output."""
    global _judge_client
    if _judge_client is None:
        import openai
        _judge_client = openai.OpenAI()
    response = _judge_client.responses.parse(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        temperature=0.3,
    )
    return response.output_parsed


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_multi_lexsum(dataset: str, task_config, data_args):
    """Load and preprocess the Multi-LexSum summarization dataset.

    Documents are filtered for length (>65536 llama2 tokens) and truncated
    to fit the target context window: input_length - 300 (prompt) - generation_max_length.
    """
    seed = getattr(data_args, "seed", 42)
    max_test_samples = getattr(data_args, "max_test_samples", None)
    shots = task_config.shots

    # Parse target input length from dataset name (e.g. multi_lexsum_128k -> 131072)
    length_suffix = dataset.rsplit("_", 1)[-1]
    input_length = _LENGTH_VALUES[length_suffix]
    truncation_length = input_length - _PROMPT_OVERHEAD - task_config.generation_max_length

    user_template = "You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n{context}\n\nNow please summarize the case."
    system_template = "Summary:"
    prompt_template = user_template + "\n\n" + system_template

    all_data = load_dataset("allenai/multi_lexsum", name="v20230518", trust_remote_code=True)
    all_data = all_data.filter(lambda x: x["summary/short"] is not None)
    train_data = all_data["train"]

    # Load keypoints from local file
    keypoints_map = {}
    with open("data/multi_lexsum/multi_lexsum_val.jsonl") as f:
        for line in f:
            d = json.loads(line)
            keypoints_map[d["id"]] = {
                "keypoints": d["summary/short_keypoints"],
                "expert_summary": d["summary/long"],
            }

    all_data = all_data.map(lambda x: {
        "context": "\n\n".join(x["sources"]),
        "demo": "" if shots == 0 else (
            "Example summaries:\n\n"
            + "\n\n".join(["Summary: {}".format(ex["summary/short"]) for ex in train_data.shuffle().select(range(shots))])
            + "\n\nNow, write a summary of the following legal documents.\n"
        ),
        "answer": x["summary/short"],
        "question": "",
    })

    data = all_data["validation"]

    # Filter for long documents
    tokenizer = TRUNCATE_TOKENIZER
    data = data.filter(lambda x: len(tokenizer(x["context"])["input_ids"]) >= _MIN_DOC_LENGTH, num_proc=32)

    # Truncate context to target length
    sep_len = len(tokenizer(_TRUNCATION_POSTFIX)["input_ids"])

    def truncate(sample):
        tokens = tokenizer(sample["context"], return_offsets_mapping=True)
        if len(tokens["input_ids"]) > truncation_length:
            sample["context"] = sample["context"][:tokens["offset_mapping"][truncation_length - sep_len][1]] + _TRUNCATION_POSTFIX
        return sample

    data = data.map(truncate, num_proc=16)

    if max_test_samples is not None and len(data) > max_test_samples:
        data = data.shuffle(seed=seed).select(range(max_test_samples))

    def post_process(output, example):
        prediction = output["output"]
        answer = example["answer"]

        mets = calculate_metrics(prediction, answer)
        parsed_pred = parse_output(prediction, system_template)
        if parsed_pred is not None:
            new_mets = calculate_metrics(parsed_pred, answer)
            mets = {k: max(v, new_mets[k]) for k, v in mets.items()}

        summary_text = (parsed_pred if parsed_pred is not None else prediction).strip()

        # Look up keypoints and expert summary for this example
        kp_data = keypoints_map.get(example["id"], {})
        kps = kp_data.get("keypoints", [])
        expert_summary = kp_data.get("expert_summary", "")

        # Judge calls: fluency, recall, precision
        fluency_input = FLUENCY_PROMPT.format(text=summary_text)
        recall_input = RECALL_PROMPT.format(
            keypoints="\n".join([f"{i+1}. {kp}" for i, kp in enumerate(kps)]),
            summary=summary_text,
        )
        precision_input = PRECISION_PROMPT.format(
            expert_summary=expert_summary,
            summary=summary_text,
        )

        fluency_scores = _call_judge(fluency_input, FluencyResponse)
        recall_scores = _call_judge(recall_input, RecallResponse)
        precision_scores = _call_judge(precision_input, PrecisionResponse)

        rec = recall_scores.recall / len(kps) if len(kps) > 0 else 0.0
        prec = precision_scores.precision / precision_scores.sentence_count if precision_scores.sentence_count > 0 else 0.0
        f1 = fluency_scores.fluency * 2 * (rec * prec) / (rec + prec) if rec + prec > 0 else 0.0

        mets["judge_f1"] = f1
        mets["judge_fluency"] = fluency_scores.fluency
        mets["judge_recall"] = rec
        mets["judge_precision"] = prec
        mets["primary_metric"] = f1

        return mets, {
            "parsed_output": parsed_pred,
            "fluency_scores": fluency_scores.model_dump(),
            "recall_scores": recall_scores.model_dump(),
            "precision_scores": precision_scores.model_dump(),
        }

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": post_process,
    }


# ---------------------------------------------------------------------------
# Register multi_lexsum variants (4k through 128k)
# ---------------------------------------------------------------------------

for _length_name, _length_val in _LENGTH_VALUES.items():
    register_task(
        f"multi_lexsum_{_length_name}",
        loader=load_multi_lexsum,
        input_max_length=_length_val,
        generation_max_length=400,
        use_chat_template=True,
        shots=2,
        primary_metric="judge_f1",
    )
