from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import os
import requests
import uvicorn

app = FastAPI(title="ResearchLens API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GEMINI
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    print("GEMINI_API_KEY found.")
else:
    print("WARNING: GEMINI_API_KEY not found.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

SYSTEM_INSTRUCTION = """
You are Lens Assistant, the AI research assistant inside ResearchLens.

You are a GENERAL research assistant. You are NOT limited to genome editing,
biology, agriculture, or any other single field.

You can help with biology, biotechnology, agriculture, medicine, chemistry,
physics, environmental science, computer science, engineering, mathematics,
statistics, bioinformatics, literature review, research methodology,
experimental design, academic writing, research planning, and scientific
questions from any field.

Answer the user's actual question.

Give a direct answer first and then explain it clearly.

For research questions:
- explain scientific concepts clearly
- distinguish established knowledge from uncertainty
- help formulate research questions
- identify possible research gaps
- help design experiments
- explain research papers
- help with literature synthesis
- do not assume every question is about genome editing

If you do not know something, say so instead of inventing information.
"""


class AssistantRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []


class PaperAskRequest(BaseModel):
    question: str
    paper: dict


class GenericRequest(BaseModel):
    question: str


# =========================
# GEMINI FUNCTION
# =========================

def ask_gemini(question, history=None):

    if not client:
        return "Gemini API key is not configured."

    history = history or []

    conversation = []

    for item in history[-12:]:
        role = item.get("role", "")
        content = item.get("content", "")

        if role and content:
            conversation.append(
                f"{role.upper()}: {content}"
            )

    conversation.append(
        f"USER: {question}"
    )

    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + "\n".join(conversation)
        + "\n\nASSISTANT:"
    )

    last_error = None

    for model_name in GEMINI_MODELS:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=2048,
                ),
            )

            if response and response.text:
                print("Gemini model used:", model_name)
                return response.text

        except Exception as e:

            last_error = str(e)

            print(
                "Gemini model failed:",
                model_name
            )
            print(last_error)

    return (
        "Lens Assistant could not connect to Gemini right now.\n\n"
        "Technical detail: "
        + str(last_error)
    )


# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {
        "app": "ResearchLens",
        "status": "running"
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY)
    }


# =========================
# LENS ASSISTANT
# =========================

@app.post("/assistant")
def assistant(request: AssistantRequest):

    answer = ask_gemini(
        request.question,
        request.history
    )

    return {
        "question": request.question,
        "answer": answer
    }


@app.get("/assistant")
def assistant_get(q: str):

    answer = ask_gemini(q)

    return {
        "question": q,
        "answer": answer
    }


# =========================
# LITERATURE SEARCH
# =========================

@app.get("/search/literature")
def search_literature(q: str):

    try:

        response = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": q,
                "per-page": 15,
                "sort": "relevance_score:desc"
            },
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        papers = []

        for work in data.get("results", []):

            authors = []

            for author in work.get(
                "authorships", []
            )[:8]:

                name = author.get(
                    "author", {}
                ).get("display_name")

                if name:
                    authors.append(name)

            location = work.get(
                "primary_location"
            ) or {}

            source = location.get(
                "source"
            ) or {}

            abstract = ""

            inverted = work.get(
                "abstract_inverted_index"
            )

            if inverted:

                words = []

                for word, positions in inverted.items():

                    for position in positions:
                        words.append(
                            (position, word)
                        )

                words.sort()

                abstract = " ".join(
                    word for _, word in words
                )

            paper_url = (
                location.get("landing_page_url")
                or work.get("doi")
                or work.get("id")
            )

            papers.append({
                "id": work.get("id"),
                "title": work.get("title")
                or "Untitled paper",
                "year": work.get(
                    "publication_year"
                ),
                "authors": authors,
                "journal": source.get(
                    "display_name"
                ) or "Unknown journal",
                "abstract": abstract,
                "doi": work.get("doi"),
                "url": paper_url,
                "open_access": (
                    work.get(
                        "open_access", {}
                    ).get(
                        "is_oa", False
                    )
                ),
                "cited_by": work.get(
                    "cited_by_count", 0
                )
            })

        return {
            "query": q,
            "count": len(papers),
            "papers": papers
        }

    except Exception as e:

        return {
            "query": q,
            "count": 0,
            "papers": [],
            "error": str(e)
        }


# =========================
# ASK LENS ABOUT PAPER
# =========================

@app.post("/paper/ask")
def paper_ask(request: PaperAskRequest):

    paper = request.paper

    paper_context = f"""
TITLE:
{paper.get('title', '')}

AUTHORS:
{', '.join(paper.get('authors', []))}

JOURNAL:
{paper.get('journal', '')}

YEAR:
{paper.get('year', '')}

ABSTRACT:
{paper.get('abstract', '')}

DOI:
{paper.get('doi', '')}
"""

    prompt = f"""
You are helping a researcher understand this scientific paper.

PAPER:
{paper_context}

QUESTION:
{request.question}

Answer specifically using the information available about this paper.
If the available information is insufficient, clearly say so.
"""

    return {
        "answer": ask_gemini(prompt)
    }


# =========================
# RESEARCH GAPS
# =========================

@app.post("/research-gaps")
def research_gaps(request: GenericRequest):

    prompt = f"""
Analyze this research topic:

{request.question}

Provide:

1. What is already known
2. What remains unclear
3. Knowledge gaps
4. Methodological gaps
5. Possible research questions
6. Possible future research directions
"""

    return {
        "answer": ask_gemini(prompt)
    }


# =========================
# EXPERIMENT DESIGNER
# =========================

@app.post("/experiment")
def experiment(request: GenericRequest):

    prompt = f"""
Help design a research experiment for:

{request.question}

Provide:

1. Objective
2. Hypothesis
3. Experimental groups
4. Controls
5. Variables
6. Workflow
7. Replication
8. Measurements
9. Statistical analysis
10. Expected interpretation
11. Limitations
"""

    return {
        "answer": ask_gemini(prompt)
    }


# =========================
# RESEARCH PROTOCOL
# =========================

@app.post("/protocol")
def protocol(request: GenericRequest):

    prompt = f"""
Create a research protocol outline for:

{request.question}

Include:

1. Objective
2. Materials
3. Experimental design
4. Workflow
5. Controls
6. Data collection
7. Analysis
8. Expected results
9. Troubleshooting
10. Safety or ethical considerations
"""

    return {
        "answer": ask_gemini(prompt)
    }


# =========================
# EVIDENCE
# =========================

@app.post("/evidence")
def evidence(request: GenericRequest):

    prompt = f"""
Analyze this research question:

{request.question}

Explain:

1. Evidence supporting the claim
2. Evidence that could contradict it
3. Important variables
4. Alternative explanations
5. Experiments that could distinguish explanations
6. Evidence needed for a strong conclusion
"""

    return {
        "answer": ask_gemini(prompt)
    }


# =========================
# RESEARCH MAP
# =========================

@app.post("/research-map")
def research_map(request: GenericRequest):

    prompt = f"""
Create a research map for:

{request.question}

Organize it into:

Main Topic
|
+-- Major concepts
|
+-- Mechanisms
|
+-- Research questions
|
+-- Methods
|
+-- Unresolved areas
|
+-- Future research
"""

    return {
        "answer": ask_gemini(request.question)
    }


# =========================
# LITERATURE SYNTHESIS
# =========================

@app.post("/literature/synthesize")
def literature_synthesize(request: GenericRequest):

    prompt = f"""
Create a structured literature synthesis for:

{request.question}

Include:

1. Background
2. Major findings
3. Areas of agreement
4. Areas of disagreement
5. Important mechanisms
6. Methodological trends
7. Knowledge gaps
8. Future directions

Clearly distinguish established findings from interpretation.
"""

    return {
        "answer": ask_gemini(prompt)
    }


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )