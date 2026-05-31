import instructor
import anthropic

from .models import Critique

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

def run_critic(role_prompt: str, output_to_evaluate: str, model: str="claude-sonnet-4-6") -> Critique:
    return client.messages.create(
        model = model,
        max_tokens = 1024,
        response_model= Critique,
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": f"Evaluate this output:\n\n{output_to_evaluate}"},

        ],
    )

