"""
ai_agents.py — LangGraph multi-agent workflow for market risk.
"""
from typing import Dict, TypedDict, Any, List
from langgraph.graph import StateGraph, END
from llm.provider_router import ResilientProviderRouter
from langchain_core.messages import HumanMessage, AIMessage

# Initialize the resilient LLM (Agents don't know which backend is active)
llm = ResilientProviderRouter()

class RiskState(TypedDict):
    regime_data: Dict[str, Any]
    metrics_data: Dict[str, Any]
    stress_data: Dict[str, Any]
    
    # Outputs from agents
    regime_analysis: str
    var_analysis: str
    stress_analysis: str
    final_report: str
    
    # Metadata for UI
    provider_status: str

# --- Agent Functions ---

def supervisor_agent(state: RiskState) -> RiskState:
    # Supervisor orchestrates and doesn't do much heavy lifting here
    # Just initialize the provider status if empty
    if "provider_status" not in state:
        state["provider_status"] = "Unknown"
    return state

def regime_analyst(state: RiskState) -> RiskState:
    regime = state["regime_data"]["regime"]
    probs = state["regime_data"]["probabilities"]
    
    prompt = f"Analyze the current market regime: '{regime}' with probabilities {probs}. Provide 2 sentences."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["regime_analysis"] = response.content
    state["provider_status"] = response.response_metadata.get("provider", "Unknown")
    return state

def var_risk_agent(state: RiskState) -> RiskState:
    metrics = state["metrics_data"]
    
    prompt = f"Analyze these portfolio risk metrics: {metrics}. Focus on VaR and Drawdown. Provide 2 sentences."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["var_analysis"] = response.content
    state["provider_status"] = response.response_metadata.get("provider", state["provider_status"])
    return state

def stress_testing_agent(state: RiskState) -> RiskState:
    worst = state["stress_data"]["worst_scenario"]
    loss = state["stress_data"]["worst_loss"]
    
    prompt = f"The portfolio is vulnerable to '{worst}' with a {loss}% loss. Provide a 2 sentence risk warning."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["stress_analysis"] = response.content
    state["provider_status"] = response.response_metadata.get("provider", state["provider_status"])
    return state

def risk_committee_reporter(state: RiskState) -> RiskState:
    # Synthesizes everything (optional, but good for LangGraph showcase)
    # We will just pass through the individual analyses for the UI to display in expanders.
    return state

# --- Build the Graph ---

workflow = StateGraph(RiskState)

workflow.add_node("Supervisor", supervisor_agent)
workflow.add_node("RegimeAnalyst", regime_analyst)
workflow.add_node("VaRRisk", var_risk_agent)
workflow.add_node("StressTesting", stress_testing_agent)
workflow.add_node("RiskCommittee", risk_committee_reporter)

workflow.set_entry_point("Supervisor")
workflow.add_edge("Supervisor", "RegimeAnalyst")
workflow.add_edge("Supervisor", "VaRRisk")
workflow.add_edge("Supervisor", "StressTesting")

# After parallel agent execution, go to committee
workflow.add_edge("RegimeAnalyst", "RiskCommittee")
workflow.add_edge("VaRRisk", "RiskCommittee")
workflow.add_edge("StressTesting", "RiskCommittee")

workflow.add_edge("RiskCommittee", END)

app_graph = workflow.compile()

def run_risk_agents(regime_data: Dict, metrics_data: Dict, stress_data: Dict) -> RiskState:
    initial_state = RiskState(
        regime_data=regime_data,
        metrics_data=metrics_data,
        stress_data=stress_data,
        regime_analysis="",
        var_analysis="",
        stress_analysis="",
        final_report="",
        provider_status="Unknown"
    )
    
    # Invoke the graph
    final_state = app_graph.invoke(initial_state)
    return final_state
