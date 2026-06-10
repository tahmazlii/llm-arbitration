import instructor
import anthropic

from .models import Critique,Verdict

client = instructor.from_anthropic(anthropic.Anthropic())

LOGIC_CRITIC_PROMPT = (
    "You are a Logical Consistency Critic. You evaluate whether the reasoning "
    "in an LLM output is valid and whether conclusions are actually supported by "
    "their premises. Look for non-sequiturs, unsupported leaps, internal "
    "contradictions, and circular reasoning. Judge the structure of the argument, "
    "not the factual accuracy of individual claims. For each problem, quote the "
    "exact span, describe the logical flaw, and assign a severity. Score 1-5 where "
    "5 means the reasoning is fully sound."
)

COMPLETENESS_CRITIC_PROMPT = (
    "You are a Completeness Critic. You evaluate whether an LLM output fully "
    "addresses every part of the original question and flag anything left out. "
    "Look for unanswered sub-questions, ignored constraints, and missing context "
    "the answer needed to be useful. For each gap, quote the relevant span (or note "
    "what is absent), describe what is missing, and assign a severity. Score 1-5 "
    "where 5 means the output is fully complete."
)

ACCURACY_CRITIC_PROMPT = (
    "You are a Factual Accuracy Critic. You evaluate whether the claims "
    "in an LLM output are verifiable and internally consistent. For each "
    "problem you find, quote the exact span, describe what's wrong, and "
    "assign a severity. Score 1-5 where 5 means no issues."
)

ADJUDICATOR_PROMPT = (
    "You are an Adjudicator. You receive an LLM output, three independent critiques "
    "of it (accuracy, logic, completeness), and a report of where the critics "
    "disagreed. Your job is to weigh the evidence and produce a final verdict. "
    "Reason through each disagreement explicitly: for factual disputes, judge which "
    "claim is correct; for logical disputes, trace the reasoning; for completeness "
    "disputes, re-read the original question. Uphold issues that are real (confirmed "
    "issues) and overrule those that aren't (dismissed flags), giving your reason for "
    "each decision. Then assign an overall quality score 1-10 and a confidence level."
)

def run_critic(role_prompt: str, output_to_evaluate: str, question: str | None = None , model: str="claude-sonnet-4-6") -> Critique:
    user_content = f"Evaluate this output:\n\n{output_to_evaluate}"
    if question:
        user_content = f"Original question:\n{question}\n\n{user_content}"
    return client.messages.create(
        model = model,
        max_tokens = 1024,
        response_model= Critique,
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_content},

        ],
    )

def run_adjudicator(output, critiques, report, question=None, model = "claude-opus-4-7") -> Verdict: 
    critiques_text = "\n\n".join(c.model_dump_json(indent=2) for c in critiques)
    report_text = report.model_dump_json(indent=2)

    user_content = (
        f"Original output being evaluated: \n{output}\n\n"
        f"The three critiques: \n{critiques_text}\n\n"
        f"Disagreement report: \n{report_text}"
    )
    if question: 
        user_content = f"Original question:\n{question}\n\n" + user_content

    return client.messages.create(
        model=model,
        max_tokens=2048,
        response_model=Verdict,
        messages=[
            {"role": "system", "content": ADJUDICATOR_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
