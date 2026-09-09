import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in .env")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Module 1: Diagnostic Assessment Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssessmentGenerateRequest(BaseModel):
    track: str = Field(..., example="Data & AI")
    question_count: int = Field(default=5, ge=3, le=10)

class QuestionItem(BaseModel):
    id: str
    category: str
    question: str
    options: List[str]
    correct_option_index: int
    rationale: str

class GradeSubmissionRequest(BaseModel):
    track: str
    questions: List[QuestionItem]
    user_answers: Dict[str, int]  # question_id -> chosen index

@app.post("/api/assessment/generate")
def generate_assessment(payload: AssessmentGenerateRequest):
    prompt = f"""
    Generate an objective diagnostic test for the track: '{payload.track}'.
    Count: exactly {payload.question_count} questions.

    Composition:
    - 50% Conceptual Architecture & Fundamentals
    - 30% Code output prediction or bug finding
    - 20% Applied Problem Solving

    Return strictly JSON matching this structure:
    {{
      "questions": [
        {{
          "id": "q1",
          "category": "e.g., Python, SQL, Statistics, Machine Learning",
          "question": "question text",
          "options": ["Option 0", "Option 1", "Option 2", "Option 3"],
          "correct_option_index": 0,
          "rationale": "Clear explanation of why this option is correct"
        }}
      ]
    }}
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a technical exam generator that returns strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(completion.choices[0].message.content)
        return {"track": payload.track, "questions": data.get("questions", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/assessment/grade")
def grade_assessment(payload: GradeSubmissionRequest):
    category_stats: Dict[str, Dict[str, int]] = {}
    detailed_feedback = []
    correct_count = 0

    for q in payload.questions:
        cat = q.category
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        
        category_stats[cat]["total"] += 1
        user_choice = payload.user_answers.get(q.id)
        is_correct = (user_choice == q.correct_option_index)

        if is_correct:
            category_stats[cat]["correct"] += 1
            correct_count += 1

        detailed_feedback.append({
            "question_id": q.id,
            "category": cat,
            "user_choice": user_choice,
            "correct_choice": q.correct_option_index,
            "is_correct": is_correct,
            "rationale": q.rationale
        })

    category_scores = {
        cat: round((stat["correct"] / stat["total"]) * 100, 1)
        for cat, stat in category_stats.items()
    }

    total_score = round((correct_count / len(payload.questions)) * 100, 1) if payload.questions else 0.0

    return {
        "total_score": total_score,
        "category_scores": category_scores,
        "detailed_feedback": detailed_feedback
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)