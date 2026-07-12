import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.rag.vector_store import get_vector_store
from src.graph.graph import build_graph, save_graph_image
from src.config import CHALLENGER_MODEL

def extract_topics(db):
    print("\n[Extractor] Analyzing documents to dynamically extract topics...")
    # Fetch some sample documents from the DB to understand the dataset
    db_data = db.get(limit=10)
    if not db_data or not db_data.get('documents'):
        return ["General Concept"]
        
    sample_text = "\n\n".join(db_data['documents'])
    
    llm = ChatOllama(model=CHALLENGER_MODEL, temperature=0.3)
    prompt = """You are an expert data analyst. Read the provided text chunks and extract a list of 3 to 5 core topics, themes, or complex concepts that would make excellent subjects for multi-hop reasoning questions.
    Return ONLY a valid JSON list of strings, for example: ["Topic 1", "Topic 2"]. Do not include markdown formatting or any other text."""
    
    try:
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=sample_text)])
        content = response.content.strip()
        
        # Clean up possible Markdown
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        topics = json.loads(content)
        if isinstance(topics, list) and len(topics) > 0:
            return topics
    except Exception as e:
        print(f"[Extractor] Failed to extract topics automatically: {e}")
        
    return ["General Theme", "Specific Mechanism"]

def main():
    print("Initializing AutoData RAG-Instruct Pipeline...")
    
    if not os.path.exists("output"):
        os.makedirs("output")
        
    # 1. Initialize Vector DB
    db = get_vector_store()
    if not db:
        print("Vector DB is empty or missing. Exiting.")
        return
        
    # 2. Dynamically extract topics from the uploaded data!
    topics = extract_topics(db)
    print(f"\n[Extractor] Discovered Topics from your data: {topics}")
    
    # 3. Build Graph & Save Visualization
    app = build_graph()
    save_graph_image(app)
    
    dataset = []
    
    for i, topic in enumerate(topics):
        print(f"\n{'='*50}\nStarting Generation Loop {i+1} for topic: '{topic}'\n{'='*50}")
        
        # Retrieve context specific to the discovered topic
        docs = db.similarity_search(topic, k=2)
        if not docs:
            print(f"No documents found for topic {topic}. Skipping.")
            continue
            
        context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        initial_state = {
            "iteration_count": 0,
            "topic": topic,
            "context": context_str,
            "question": "",
            "reference_answer": "",
            "rubric": [],
            "weak_answer": "",
            "strong_answer": "",
            "weak_score": 0,
            "strong_score": 0,
            "feedback": "",
            "final_qa_pair": None
        }
        
        # Invoke the LangGraph workflow
        result = app.invoke(initial_state)
        
        strong_score = result.get("strong_score", 0)
        weak_score = result.get("weak_score", 100)
        
        if strong_score >= 70 and weak_score <= 50:
            dataset.append({
                "topic": result.get("topic", ""),
                "context": result.get("context", ""),
                "question": result.get("question", ""),
                "reference_answer": result.get("reference_answer", ""),
                "strong_answer": result.get("strong_answer", ""),
                "rubric": result.get("rubric", []),
                "scores": {
                    "weak_score": weak_score,
                    "strong_score": strong_score
                }
            })
            print(f">>> Successfully generated 1 QA pair for '{topic}'.")
        else:
            print(f">>> Failed to generate valid pair for '{topic}' within max retries.")
            
    # Save the final dataset
    with open("output/synthetic_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nDone! Generated {len(dataset)} QA pairs. Saved to output/synthetic_dataset.json")

if __name__ == "__main__":
    main()
