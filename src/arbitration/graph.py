from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from operator import add
from .models import Critique, DisagreementReport, CriticFlag, SpanOverlap
from .critics import run_critic, ACCURACY_CRITIC_PROMPT, LOGIC_CRITIC_PROMPT, COMPLETENESS_CRITIC_PROMPT
from langgraph.graph import StateGraph, START, END

class ArbitrationState(TypedDict):
    question: str
    output: str
    critiques: Annotated[list[Critique],add]
    disagreements: NotRequired[DisagreementReport]

def logic_node(state: ArbitrationState) -> dict:
    critique = run_critic(LOGIC_CRITIC_PROMPT, state["output"])
    return {"critiques": [critique]}

def accuracy_node(state: ArbitrationState) -> dict:
    critique = run_critic(ACCURACY_CRITIC_PROMPT, state["output"])
    return {"critiques": [critique]}

def completeness_node(state: ArbitrationState) -> dict:
    critique = run_critic(COMPLETENESS_CRITIC_PROMPT, state["output"], question=state["question"])
    return {"critiques": [critique]}

def quotes_match(a: str, b: str) -> bool:
    return a in b or b in a

def collector_node(state: ArbitrationState) -> dict:
    critiques = state["critiques"]
    scores = [c.score for c in critiques]
    spread = max(scores) - min(scores)
    highest = max(critiques, key = lambda c : c.score ).dimension
    lowest = min(critiques, key = lambda c: c.score).dimension

    all_flags = []
    for c in critiques: 
        for issue in c.issues:
            all_flags.append((c.dimension, issue.quote, issue.severity))

    groups = [] # each group: {"span": <representative quote>, "flags": [list of (dim, quote, sev)]}
    for dim, quote, sev in all_flags:
        placed = False 
        for g in groups:
            if quotes_match(quote, g["span"]):
                g["flags"].append((dim, quote, sev))
                placed = True
                break
        if not placed: 
            groups.append({"span": quote, "flags": [(dim, quote, sev)]})
    
    overlaps = [] 
    for g in groups: 
        dims = {dim for dim, quote, sev in g["flags"]}
        if len(dims) >= 2:
            flags = [CriticFlag(dimension=dim, severity=sev) for dim, quote, sev in g["flags"]]
            overlaps.append(SpanOverlap(span = g["span"], flags = flags))


    report = DisagreementReport(
        score_spread = spread,
        highest_dimension = highest,
        lowest_dimension = lowest,
        overlaps = overlaps,
    )
    return {"disagreements" : report}

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



