from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from .models import Critique
from .critics import run_critic, ACCURACY_CRITIC_PROMPT, LOGIC_CRITIC_PROMPT, COMPLETENESS_CRITIC_PROMPT
from langgraph.graph import StateGraph, START, END

class ArbitrationState(TypedDict):
    question: str
    output: str
    critiques: Annotated[list[Critique],add]

def logic_node(state: ArbitrationState) -> dict:
    critique = run_critic(LOGIC_CRITIC_PROMPT, state["output"])
    return {"critiques": [critique]}

def accuracy_node(state: ArbitrationState) -> dict:
    critique = run_critic(ACCURACY_CRITIC_PROMPT, state["output"])
    return {"critiques": [critique]}

def completeness_node(state: ArbitrationState) -> dict:
    critique = run_critic(COMPLETENESS_CRITIC_PROMPT, state["output"], question=state["question"])
    return {"critiques": [critique]}

def collector_node(state: ArbitrationState) -> dict:
    return {}

builder = StateGraph(ArbitrationState)

builder.add_node("logic", logic_node)
builder.add_node("accuracy", accuracy_node)
builder.add_node("completeness", completeness_node)
builder.add_node("collector", collector_node)

builder.add_edge(START, "logic")
builder.add_edge(START, "accuracy")
builder.add_edge(START, "completeness")

builder.add_edge("logic", "collector")
builder.add_edge("accuracy", "collector")
builder.add_edge("completeness", "collector")

builder.add_edge("collector", END)

graph = builder.compile()



