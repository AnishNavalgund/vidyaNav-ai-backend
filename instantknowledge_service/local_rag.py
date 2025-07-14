# instantknowledge_service/rag_agent.py

import numpy as np
from typing import List
from PyPDF2 import PdfReader
import re
import logging

from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from instantknowledge_service.agent import instant_knowledge_agent, model
from instantknowledge_service.schema import AnswerResponse


def extract_clean_chunks_from_pdf(pdf_path: str, chunk_size: int = 10) -> List[str]:
    # Read PDF
    reader = PdfReader(pdf_path)
    raw_text = " ".join([page.extract_text() or "" for page in reader.pages])

    if not raw_text.strip():
        raise ValueError("No readable text extracted from PDF.")

    # clean_text = re.sub(r"\s+", " ", raw_text)  # Normalize whitespace
    clean_text = re.sub(r"(https?:\/\/[^\s]+)", "", raw_text)  # Remove URLs
    # clean_text = re.sub(r"\b(?:Page|FormEditor|Editor|Contact form|You are here)\b.*", "", clean_text)  # Remove known boilerplate

    # Keep punctuation, hyphens, parentheses
    clean_text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-]", "", clean_text)

    clean_text = clean_text.strip()
    words = clean_text.split()

    logging.info(f"Extractraction completed")

    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    filtered_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip().split()) >= 10]
    logging.info(f"Chunking done")

    if not filtered_chunks:
        logging.warning("No valid chunks after filtering")
        return [clean_text]

    return filtered_chunks


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # to Prevent division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))

async def get_answer_with_uploaded_textbook(
    question: str,
    grade_level: int,
    language: str,
    pdf_path: str
) -> AnswerResponse:
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    chunks = extract_clean_chunks_from_pdf(pdf_path)
    if not chunks:
        raise ValueError("No readable text found in uploaded PDF.")

    embedded = []
    for chunk in chunks:
        emb = embed_model.get_embeddings([
            TextEmbeddingInput(task_type="RETRIEVAL_DOCUMENT", text=chunk)
        ])[0].values
        embedded.append({"chunk": chunk, "embedding": emb})

    query_emb = embed_model.get_embeddings([
        TextEmbeddingInput(task_type="RETRIEVAL_QUERY", text=question)
    ])[0].values

    ranked_chunks = sorted(
    [{"chunk": e["chunk"], "score": cosine_similarity(query_emb, e["embedding"])} for e in embedded],
    key=lambda x: x["score"],
    reverse=True)

    top_chunks = [r["chunk"] for r in ranked_chunks[:3]]
    logging.info(f"Passing top chunks to Gemini.")

    context_str = "\n\n".join([f"📘 Context Snippet:\n{chunk}" for chunk in top_chunks])

    prompt = (
        f"You are helping a Grade {grade_level} student understand a concept in {language}.\n"
        f"Use simple, relatable explanations. Try to include analogies relevant to Indian classrooms.\n\n"
        f"{context_str}\n\n"
        f"Student Question: {question}\n"
        f"Answer:"
    )
    logging.info(f"Agent called")
    result = await instant_knowledge_agent.run(prompt)
    result.output.model_used = model.model_name
    result.output.source_chunks = [chunk[:150] + "..." for chunk in top_chunks]
    return result.output
