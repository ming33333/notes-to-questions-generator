"""
local_qa_generator.py

Generate questions + answers from notes using a local LLM (LlamaCpp or GPT4All).
Falls back to a simple offline heuristic if no model is loaded.
"""

import json
from typing import List, Dict
from gpt4all import GPT4All
model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")



def get_prompt(notes: str, num_questions: int = 6) -> str:
    return f"""
You are a concise study assistant.
Given the notes below, generate {num_questions} question-and-answer pairs.
Each answer should be short (1-2 sentences) and accurate based on the notes.
Include the specific source text from the notes the Q&A is based on.
Return ONLY valid JSON: a list of objects with keys "question", "answer", and "text".

Example:
[
  {{
    "question": "What is X?",
    "answer": "X is ...",
    "text": "The sentence or passage from notes about X."
  }}
]

Notes:
{notes}
"""

# --- Local Model Integrations ---
def try_llamacpp_generate(notes: str, model_path: str, num_questions: int = 6) -> List[Dict]:
    from langchain.llms import LlamaCpp
    llm = LlamaCpp(model_path=model_path, n_ctx=2048, temperature=0.2, max_tokens=1024)
    prompt = get_prompt(notes, num_questions)
    raw = llm(prompt)
    return safe_json_parse(raw)

def try_gpt4all_generate(notes: str, model_path: str, num_questions: int = 6) -> List[Dict]:
    from gpt4all import GPT4All
    model = GPT4All(model_path)
    prompt = get_prompt(notes, num_questions)
    raw = model.generate(prompt, max_tokens=512, temperature=0.2)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return safe_json_parse(raw)

# --- JSON Parsing Helper ---
def safe_json_parse(raw: str) -> List[Dict]:
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw
        start = s.find('[')
        end = s.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        print("Warning: Could not parse JSON. Returning empty list.")
        return []

# --- Fallback mock generator ---
def mock_generate(notes: str, num_questions: int = 6) -> List[Dict]:
    import re
    sentences = [s.strip() for s in re.split(r'[\n\.]+', notes) if s.strip()]
    results = []
    for i, s in enumerate(sentences[:num_questions]):
        q = f"What is the main idea of: '{s[:40]}...'"
        a = f"The main idea is that {s[:80]}."
        results.append({"question": q, "answer": a, "text": s})
    while len(results) < num_questions:
        results.append({
            "question": "Summarize a key idea from the notes.",
            "answer": "This section explains one of the main concepts.",
            "text": sentences[0] if sentences else ""
        })
    return results

# --- Public API ---
def generate_qa_from_notes(
    notes: str,
    num_questions: int = 6,
    llamacpp_model_path: str = None,
    gpt4all_model_path: str = None
) -> List[Dict]:
    if llamacpp_model_path:
        try:
            print("Trying LlamaCpp model...")
            return try_llamacpp_generate(notes, llamacpp_model_path, num_questions)
        except Exception as e:
            print("LlamaCpp failed:", e)
    if gpt4all_model_path:
        try:
            print("Trying GPT4All model...")
            return try_gpt4all_generate(notes, gpt4all_model_path, num_questions)
        except Exception as e:
            print("GPT4All failed:", e)
    print("Using mock generator (no model found).")
    return mock_generate(notes, num_questions)

# --- Example Usage ---
if __name__ == "__main__":
    sample_notes = """
    Terraform uses Infrastructure as Code (IaC) to define and provision cloud infrastructure.
    It supports multiple providers like AWS, Azure, and GCP.
    State files track deployed resources and must be stored securely.
    Variables make configurations reusable and modular.
    """

    LLAMA_MODEL_PATH = None  # e.g. "./models/llama-2-7b.Q4_K_M.gguf"
    GPT4ALL_MODEL_PATH = "~/.cache/gpt4all/orca-mini-3b-gguf2-q4_0.gguf"


    qa_pairs = generate_qa_from_notes(
        sample_notes,
        num_questions=5,
        llamacpp_model_path=LLAMA_MODEL_PATH,
        gpt4all_model_path=GPT4ALL_MODEL_PATH
    )

    print(json.dumps(qa_pairs, indent=2))
