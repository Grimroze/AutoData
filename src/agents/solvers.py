from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langsmith import traceable
from src.config import WEAK_MODEL, STRONG_MODEL
from src.graph.state import AutoDataState

def get_llm(model_name: str, temperature: float = 0.7):
    return ChatOllama(model=model_name, temperature=temperature)

@traceable(name="Weak Solver Agent", run_type="chain")
def weak_solver_node(state: AutoDataState):
    print("[Weak Solver] Attempting question...")
    llm = get_llm(WEAK_MODEL, 0.2)
    user_msg = f"Context:\n{state['context']}\n\nQuestion:\n{state['question']}\n\nAnswer the question based ONLY on the context."
    response = llm.invoke([HumanMessage(content=user_msg)])
    return {"weak_answer": response.content}
    
@traceable(name="Strong Solver Agent", run_type="chain")
def strong_solver_node(state: AutoDataState):
    print("[Strong Solver] Attempting question...")
    llm = get_llm(STRONG_MODEL, 0.2)
    user_msg = f"Context:\n{state['context']}\n\nQuestion:\n{state['question']}\n\nAnswer the question based ONLY on the context."
    response = llm.invoke([HumanMessage(content=user_msg)])
    return {"strong_answer": response.content}
