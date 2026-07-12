from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from src.config import CHALLENGER_MODEL
from src.utils.prompts import CHALLENGER_SYSTEM_PROMPT
from src.graph.state import AutoDataState

class RubricCriterion(BaseModel):
    criterion: str
    weight: int = Field(description="Positive (+2 to +5) or negative (-1 to -5)")

class ChallengerOutput(BaseModel):
    question: str
    answer: str
    rubric: List[RubricCriterion]

def get_llm(model_name: str, temperature: float = 0.7):
    return ChatOllama(model=model_name, temperature=temperature)

@traceable(name="Challenger Agent", run_type="chain")
def challenger_node(state: AutoDataState):
    print(f"\n[Challenger] Generating question for topic: {state.get('topic', 'General')}")
    llm = get_llm(CHALLENGER_MODEL, 0.7).with_structured_output(ChallengerOutput)
    
    prompt = CHALLENGER_SYSTEM_PROMPT
    user_msg = f"Topic: {state.get('topic', 'General')}\nContext:\n{state['context']}"
    
    if state.get("iteration_count", 0) > 0:
        print("[Challenger] Received feedback, refining question...")
        weak_score = state.get("weak_score", 0)
        strong_score = state.get("strong_score", 0)
        
        user_msg += f"\n\nPREVIOUS SCORES:\nWeak Model Score: {weak_score}%\nStrong Model Score: {strong_score}%\n"
        user_msg += f"JUDGE FEEDBACK: {state.get('feedback', '')}\n\n"
        
        if weak_score > 50:
            user_msg += "CRITICAL INSTRUCTION: The Weak Model scored too high! Your previous question was TOO EASY. You MUST make the new question significantly harder by requiring multi-step reasoning across distant chunks. DO NOT just rephrase the old question."
        elif strong_score < 70:
            user_msg += "CRITICAL INSTRUCTION: The Strong Model failed! Your previous question was ILLOGICAL or unanswerable. Make the new question perfectly grounded in the text and perfectly clear, but still complex."
        
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_msg)])
    
    return {
        "question": response.question,
        "reference_answer": response.answer,
        "rubric": [{"criterion": r.criterion, "weight": r.weight} for r in response.rubric],
        "iteration_count": state.get("iteration_count", 0) + 1
    }
