import numpy as np
from typing import List
from PyPDF2 import PdfReader
import re
import logging
import pysbd
from instantknowledge_service.schema import AnswerResponse
from instantknowledge_service.agent import get_instant_knowledge_answer

def extract_semantic_chunks(
    pdf_path: str,
    chunk_size: int = 4,
    overlap: int = 1,
    language: str = "english"
) -> List[str]:
    lang_map = {
        "english": "en",
        "en": "en",
        "hindi": "hi",
        "hi": "hi",
        "german": "de",
        "de": "de"
    }
    lang_key = lang_map.get(language.lower().strip())
    if not lang_key:
        raise ValueError(f"Unsupported language: {language}")
    reader = PdfReader(pdf_path)
    raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
    if not raw_text.strip():
        raise ValueError("No readable text extracted from PDF.")
    clean_text = re.sub(r"(https?://\S+)", "", raw_text)
    clean_text = re.sub(r"[^a-zA-Z\u0900-\u097F\u0C80-\u0CFF0-9\s.,!?;:'\"()\-]", "", clean_text)
    clean_text = clean_text.strip()
    try:
        segmenter = pysbd.Segmenter(language=lang_key, clean=True)
        sentences = segmenter.segment(clean_text)
    except Exception as e:
        raise ValueError(f"Error using pysbd for language '{lang_key}': {e}")
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.split()) >= 10:
            chunks.append(chunk.strip())
    if not chunks:
        logging.warning("No valid chunks extracted.")
        return [clean_text]
    logging.info(f"Extracted {len(chunks)} multilingual chunks.")
    return chunks

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
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
    from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    chunks = extract_semantic_chunks(pdf_path, language=language)
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
    result = get_instant_knowledge_answer(prompt)
    result.model_used = "gemini-2.0-flash-lite"
    result.source_chunks = [chunk[:150] + "..." for chunk in top_chunks]
    return result
