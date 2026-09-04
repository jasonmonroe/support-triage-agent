# models/agentic_state.py

# Pythton Libraries
from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    query: str  # The current user query
    expanded_query: str  # The expanded version of the user query
    context: List[Dict[str, Any]]  # Retrieved documents (content and metadata)
    response: str  # The generated response to the user query
    precision_score: float  # The precision score of the response
    groundedness_score: float  # The groundedness score of the response
    groundedness_loop_count: int  # Counter for groundedness refinement loops
    precision_loop_count: int  # Counter for precision refinement loops
    feedback: str
    query_feedback: str
    groundedness_check: bool
    loop_max_iter: int
    AI_ROLE: str
