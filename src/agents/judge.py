from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from src.config import JUDGE_MODEL
from src.utils.prompts import JUDGE_SYSTEM_PROMPT
from src.graph.state import AutoDataState

class JudgeScore(BaseModel):
    criterion: str
    score: int
    
class JudgeOutput(BaseModel):
    criteria_scores: List[JudgeScore]
    total_percentage: float = Field(description="Total score from 0 to 100")
    feedback: str = Field(description="Brief reasoning for the score")

def get_llm(model_name: str, temperature: float = 0.7):
    return ChatOllama(model=model_name, temperature=temperature)

@traceable(name="Judge Agent", run_type="chain")
def judge_node(state: AutoDataState):
    print("[Judge] Grading answers...")
    llm = get_llm(JUDGE_MODEL, 0.1).with_structured_output(JudgeOutput)
    
    base_msg = f"Question: {state['question']}\nReference: {state['reference_answer']}\nRubric: {state['rubric']}"
    
    # Evaluate Weak
    weak_msg = f"{base_msg}\n\nSolver Answer:\n{state['weak_answer']}"
    weak_eval = llm.invoke([SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=weak_msg)])
    
    # Evaluate Strong
    strong_msg = f"{base_msg}\n\nSolver Answer:\n{state['strong_answer']}"
    strong_eval = llm.invoke([SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=strong_msg)])
    
    # Combine feedback
    feedback = f"Weak Score: {weak_eval.total_percentage} | Strong Score: {strong_eval.total_percentage}\nJudge Notes (Weak): {weak_eval.feedback}\nJudge Notes (Strong): {strong_eval.feedback}"
    
    print(f"[Judge Result] Weak: {weak_eval.total_percentage} | Strong: {strong_eval.total_percentage}")
    return {
        "weak_score": weak_eval.total_percentage,
        "strong_score": strong_eval.total_percentage,
        "feedback": feedback
    }
