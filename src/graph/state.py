from typing import TypedDict, List, Dict, Any, Optional

class AutoDataState(TypedDict):
    iteration_count: int
    topic: str
    context: str
    question: str
    reference_answer: str
    rubric: List[Dict[str, Any]]
    weak_answer: str
    strong_answer: str
    weak_score: int
    strong_score: int
    feedback: str
    final_qa_pair: Optional[Dict[str, Any]]
