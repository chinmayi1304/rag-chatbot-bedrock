"""
Lambda: rag-query
Triggered by: API Gateway POST /query

Request body:
  {
    "question": "What is the refund policy?",
    "top_k": 4          (optional, default 4)
  }

What it does:
  1. Embeds the user's question using Titan Embeddings
  2. Searches ChromaDB for the top-k most similar chunks
  3. Builds a RAG prompt: [context chunks] + [user question]
  4. Calls Claude Haiku on Bedrock to generate the answer
  5. Returns the answer + the source chunks used (for transparency)

Architecture decision — why return source chunks?
  This is called "citation" in RAG systems. It lets the user verify
  where the answer came from. It also helps you debug bad answers
  (wrong chunks retrieved = chunking or embedding problem).
  Every production RAG system does this. Shows you understand the full stack.

Architecture decision — why Claude Haiku over Claude Sonnet?
  Haiku = fastest + cheapest Claude model (~$0.25/1M input tokens vs $3/1M for Sonnet).
  For RAG Q&A the answer quality difference is minimal — the context does most of the work.
  SA mindset: always justify your model choice with cost/performance reasoning.
"""

import json
import os
import logging

import boto3
import chromadb
from chromadb.config import Settings

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock = boto3.client("bedrock-runtime", region_name=os.environ["REGION"])

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
COLLECTION_NAME = os.environ["CHROMA_COLLECTION"]
EMBED_MODEL_ID = os.environ["EMBED_MODEL_ID"]
LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
TOP_K = int(os.environ.get("TOP_K_CHUNKS", "4"))

# System prompt — instructs Claude to stay grounded in the provided context
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents.

Rules:
- Answer using information from the context below. Do not use any outside knowledge.
- If the answer is not in the context, say: "I couldn't find information about that in the uploaded documents."
- Be concise and direct. Quote relevant sentences from the context when helpful.
- At the end of your answer, mention which document/page the information came from."""


def get_chroma_collection():
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_text(text: str) -> list[float]:
    """Embed text using Bedrock Titan Embeddings."""
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    body = json.loads(response["body"].read())
    return body["embedding"]


def retrieve_chunks(question: str, top_k: int) -> list[dict]:
    """
    Embed the question and find the most semantically similar chunks in ChromaDB.
    ChromaDB returns chunks ordered by cosine distance (lower = more similar).
    """
    query_embedding = embed_text(question)
    collection = get_chroma_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity_score": round(1 - results["distances"][0][i], 3),
            # Convert distance to similarity: 1 - distance (cosine)
            # Score of 1.0 = identical, 0.0 = completely unrelated
        })

    return chunks


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the RAG prompt — this is the core of the RAG pattern.
    Format: system context + retrieved chunks + user question.

    The quality of this prompt directly determines answer quality.
    Common prompt engineering mistakes:
      - Dumping all chunks without labels → model can't cite sources
      - No instruction to stay grounded → model hallucinates
      - Question buried at the end → model doesn't focus on it
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk["metadata"].get("filename", "unknown")
        chunk_idx = chunk["metadata"].get("chunk_index", "?")
        score = chunk["similarity_score"]
        context_parts.append(
            f"[Source {i}: {filename}, chunk {chunk_idx}, relevance {score}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""Here are the relevant sections from the uploaded documents:

{context}

---

Based ONLY on the context above, please answer this question:
{question}"""


def call_claude_haiku(prompt: str) -> str:
    """
    Call Claude Haiku via Bedrock using the Messages API.
    Bedrock free trial: 1M tokens/month for Claude Haiku.
    After trial: ~$0.25 per 1M input tokens (extremely cheap).
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def lambda_handler(event, context):
    """Handle API Gateway event."""
    logger.info("Query received: %s", json.dumps(event))

    # ── Parse request ───────────────────────────────────────────────────
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return response(400, {"error": "Request body must be valid JSON"})

    question = body.get("question", "").strip()
    if not question:
        return response(400, {"error": "question field is required"})

    top_k = int(body.get("top_k", TOP_K))
    top_k = min(max(top_k, 1), 10)  # clamp between 1 and 10

    logger.info("Question: '%s', top_k=%d", question, top_k)

    # ── Retrieve relevant chunks ────────────────────────────────────────
    try:
        chunks = retrieve_chunks(question, top_k)
    except Exception as e:
        logger.error("ChromaDB retrieval failed: %s", str(e))
        return response(500, {"error": "Failed to search documents. Is ChromaDB running?"})

    if not chunks:
        return response(200, {
            "answer": "No documents have been ingested yet. Please upload a PDF first.",
            "sources": [],
        })

    logger.info("Retrieved %d chunks, top similarity: %s", len(chunks), chunks[0]["similarity_score"])

    # ── Build prompt + call Claude ──────────────────────────────────────
    prompt = build_rag_prompt(question, chunks)

    try:
        answer = call_claude_haiku(prompt)
    except Exception as e:
        logger.error("Bedrock call failed: %s", str(e))
        return response(500, {"error": "Failed to generate answer. Check Bedrock model access."})

    # ── Return answer + sources ─────────────────────────────────────────
    return response(200, {
        "answer": answer,
        "question": question,
        "sources": [
            {
                "filename": c["metadata"].get("filename", "unknown"),
                "chunk_index": c["metadata"].get("chunk_index"),
                "relevance_score": c["similarity_score"],
                "preview": c["text"][:200] + "...",  # first 200 chars as preview
            }
            for c in chunks
        ],
    })


def response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }
