"""
Lambda: rag-ingestor
Triggered by: S3 PutObject event (when a PDF is uploaded)

What it does:
  1. Downloads the PDF from S3
  2. Extracts text using PyMuPDF (fitz)
  3. Splits text into overlapping chunks (chunk_size=500 words, overlap=50)
  4. Calls Bedrock Titan Embeddings to convert each chunk to a vector
  5. Stores all vectors + metadata in ChromaDB on EC2
  6. Updates DynamoDB with the document status

Architecture decision — why chunk with overlap?
  Without overlap, a sentence split across two chunks loses context.
  50-word overlap means the last 50 words of chunk N are the first 50 of chunk N+1.
  This ensures retrieval finds complete thoughts, not cut-off sentences.

Architecture decision — why ChromaDB on EC2 vs Pinecone/OpenSearch?
  Pinecone free tier is limited. OpenSearch Serverless has a $0.24/OCU-hour minimum.
  ChromaDB on t2.micro = $0/month on free tier for 12 months.
  Trade-off: you manage the instance. Fine for a portfolio project.
"""

import json
import os
import io
import uuid
import logging
from datetime import datetime, timezone

import boto3
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
bedrock = boto3.client("bedrock-runtime", region_name=os.environ["REGION"])

DOCS_TABLE = os.environ["DOCS_TABLE"]
CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
COLLECTION_NAME = os.environ["CHROMA_COLLECTION"]
EMBED_MODEL_ID = os.environ["EMBED_MODEL_ID"]

CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # overlap in words


def get_chroma_collection():
    """Connect to ChromaDB running on EC2 and get (or create) the collection."""
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create — safe to call multiple times
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )
    return collection


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        full_text += f"\n[Page {page_num + 1}]\n{page_text}"
    doc.close()
    return full_text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Word-based chunking is simpler and more predictable than character-based.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide window with overlap

    return [c for c in chunks if len(c.strip()) > 50]  # discard tiny fragments


def embed_text(text: str) -> list[float]:
    """
    Call Bedrock Titan Embeddings to convert text to a vector.
    Titan Embeddings v1 produces 1536-dimensional vectors.
    Free tier: 1M input tokens/month during trial period.
    """
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    body = json.loads(response["body"].read())
    return body["embedding"]


def lambda_handler(event, context):
    """Handle S3 trigger event."""
    logger.info("Ingestor triggered: %s", json.dumps(event))

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        logger.info("Processing s3://%s/%s", bucket, key)

        # ── Download PDF from S3 ────────────────────────────────────────
        response = s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = response["Body"].read()

        # ── Extract text ────────────────────────────────────────────────
        full_text = extract_text_from_pdf(pdf_bytes)
        if not full_text:
            logger.warning("No text extracted from %s — skipping", key)
            continue

        logger.info("Extracted %d characters from %s", len(full_text), key)

        # ── Chunk text ──────────────────────────────────────────────────
        chunks = chunk_text(full_text)
        logger.info("Created %d chunks from %s", len(chunks), key)

        # ── Embed each chunk + store in ChromaDB ────────────────────────
        doc_id = str(uuid.uuid4())
        collection = get_chroma_collection()

        chunk_ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            chunk_id = f"{doc_id}_{i}"

            chunk_ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "filename": key,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        # ChromaDB upsert — safe if called multiple times for same doc
        collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d chunks in ChromaDB for doc_id=%s", len(chunks), doc_id
        )

        # ── Save metadata to DynamoDB ───────────────────────────────────
        table = dynamodb.Table(DOCS_TABLE)
        table.put_item(
            Item={
                "docId": doc_id,
                "filename": key,
                "s3Key": key,
                "chunkCount": len(chunks),
                "status": "READY",
                "uploadedAt": datetime.now(timezone.utc).isoformat(),
                "textLength": len(full_text),
            }
        )

        logger.info("Document %s fully ingested. docId=%s", key, doc_id)

    return {"statusCode": 200, "body": "Ingestion complete"}
