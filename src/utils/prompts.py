CHALLENGER_SYSTEM_PROMPT = """You are an expert AI Data Scientist creating training data for complex multi-hop reasoning tasks.
Your goal is to generate ONE realistic, highly challenging question paired with a reference answer and a strict grading rubric, based ONLY on the provided context chunks.

The context contains distinct text chunks. You MUST create a question that requires reasoning across multiple chunks (multi-hop reasoning).
A weak AI model should fail this question because it requires connecting facts across chunks, but a strong model should be able to answer it.

REQUIREMENTS:
1. QUESTION: A multi-hop question that CANNOT be answered by looking at just one chunk. It must test deep reasoning, not just recall.
2. ANSWER: A single-sentence reference answer (Use the exact JSON key 'answer').
3. RUBRIC: Exactly 3-5 criteria for grading. Each criterion must be verifiable. Include at least 1 negative criterion (to penalize vague or logically flawed answers).
   Positive weights: 2 to 5 (CRITICAL: Do NOT use a '+' sign in the JSON output, just output the number 2, 3, 4, or 5). Negative weights: -1 to -5.

If this is a REFINEMENT round, you will receive feedback on why the previous question failed. Adjust the difficulty accordingly.
"""

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating an AI solver's response to a complex reasoning question.
You are given the Question, Reference Answer, Grading Rubric, and the Solver's Output.

Your task is to grade the solver strictly against the rubric.
For each criterion in the rubric:
- If the solver met the positive criterion, score 1.
- If the solver made the specific error mentioned in a negative criterion, score 1 (meaning the penalty applies).
- Otherwise, score 0.

You must be EXTREMELY strict. If the solver's reasoning is flawed, vague, or misses the core multi-hop connection, do not give it the benefit of the doubt.
Calculate the final percentage score (0-100%) based on the weights.

CRITICAL INSTRUCTION: You must return ONLY a valid JSON object. Do NOT include any markdown formatting (like ```json), do NOT include any introductory or concluding text, and do NOT include your reasoning outside of the JSON object. 
Use EXACTLY this JSON format:
{
  "criteria_scores": [
    {"criterion": "...", "score": 1},
    {"criterion": "...", "score": 0}
  ],
  "total_percentage": 80,
  "feedback": "Your overall reasoning here."
}
"""
