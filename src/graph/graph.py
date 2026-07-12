from langgraph.graph import StateGraph, START, END
from src.graph.state import AutoDataState
from src.agents.challenger import challenger_node
from src.agents.solvers import weak_solver_node, strong_solver_node
from src.agents.judge import judge_node
from src.config import MAX_RETRIES

def route_after_judge(state: AutoDataState) -> str:
    # Condition: Accepted if Strong is good and Weak is bad
    if state["strong_score"] >= 70 and state["weak_score"] <= 50:
        print(">>> ROUTER: PERFECT! Weak failed, Strong passed. End of loop.")
        return END
        
    # Condition: Max retries reached
    if state["iteration_count"] >= MAX_RETRIES:
        print(">>> ROUTER: MAX RETRIES REACHED. Discarding this attempt.")
        state["final_qa_pair"] = None
        return END
        
    # Condition: Too Easy (Weak passed)
    if state["weak_score"] > 50:
        print(">>> ROUTER: REJECTED (Too Easy). Weak model scored too high. Refining...")
        return "challenger"
        
    # Condition: Flawed (Strong failed)
    if state["strong_score"] < 70:
        print(">>> ROUTER: REJECTED (Illogical). Strong model failed. Refining...")
        return "challenger"
        
    # Fallback
    return "challenger"

def build_graph():
    builder = StateGraph(AutoDataState)
    
    builder.add_node("challenger", challenger_node)
    builder.add_node("weak_solver", weak_solver_node)
    builder.add_node("strong_solver", strong_solver_node)
    builder.add_node("judge", judge_node)
    
    builder.add_edge(START, "challenger")
    
    # Parallel execution for solvers
    builder.add_edge("challenger", "weak_solver")
    builder.add_edge("challenger", "strong_solver")
    
    # Judge waits for both solvers (LangGraph handles parallel joins if designed correctly, 
    # but for simplicity in StateGraph without complex map-reduce, we can sequence them, 
    # or rely on LangGraph's default parallel fan-out/fan-in if state keys don't conflict)
    # Actually, to avoid overwriting state issues without a reducer, we should add them in sequence 
    # OR use separate keys (which we do: weak_answer, strong_answer).
    builder.add_edge(["weak_solver", "strong_solver"], "judge")
    
    # Conditional routing
    builder.add_conditional_edges("judge", route_after_judge)
    
    return builder.compile()

def save_graph_image(app, output_path="output/graph_diagram.png"):
    """Save the compiled graph as a PNG image."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(graph_png)
        print(f"[Graph] Saved graph visualization to {output_path} ")
    except Exception as e:
        print(f"[Graph] Could not save graph image: {e}")
        print("[Graph] Tip: Install 'pip install pyppeteer' if image generation fails.")

# Run this file directly to just get the graph image: python -m src.graph.graph
if __name__ == "__main__":
    app = build_graph()
    save_graph_image(app)
