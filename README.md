# RAG Document Q&A Chatbot — AWS Bedrock + ChromaDB

Ask questions about any PDF using AWS Bedrock (Claude Haiku + Titan Embeddings) and ChromaDB as the vector store. Entirely on AWS free tier.

---

## Architecture

```
INGESTION:  PDF → S3 → Lambda (chunk) → Bedrock Titan Embeddings → ChromaDB (EC2)
QUERY:      Question → API Gateway → Lambda → Titan (embed) → ChromaDB (search) → Claude Haiku → Answer
```

### Why these choices?
| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| LLM | Claude Haiku | GPT-4o, Gemini | Bedrock = AWS IAM auth, no separate API key, free trial |
| Embeddings | Titan Embed v1 | OpenAI Ada | Same — stays within AWS, free trial |
| Vector store | ChromaDB on EC2 | Pinecone, OpenSearch | ChromaDB is free + self-hosted; OpenSearch Serverless = $0.24/OCU-hr minimum |
| Chunking | Word-based overlap | Fixed character | More predictable token counts, overlap preserves sentence context |
| LLM model | Haiku | Sonnet | RAG quality is context-driven; Haiku is 12x cheaper with comparable results |

---

## Cost breakdown (all free tier)

| Service | Free tier | This project uses |
|---|---|---|
| Lambda | 1M requests/month | ~50 requests/day = 1.5k/month |
| S3 | 5 GB, 12 months | ~100 MB of PDFs |
| DynamoDB | 25 GB, 25 RCU/WCU | < 1 MB |
| EC2 t2.micro | 750 hr/month, 12 months | 730 hr/month (always on) |
| Bedrock (trial) | 1M tokens/month | ~5k tokens per query |
| API Gateway | 1M calls, 12 months | ~50/day = 1.5k/month |

**Estimated monthly cost: $0**

---

## Prerequisites

```bash
node --version     # v18+
python --version   # 3.12+
aws --version      # AWS CLI v2
cdk --version      # CDK v2 — install: npm install -g aws-cdk
```

### AWS setup (one time)
```bash
# Configure credentials
aws configure
# Enter: Access Key ID, Secret Key, region (ap-south-1), output format (json)

# Bootstrap CDK (one time per account/region)
cdk bootstrap aws://YOUR_ACCOUNT_ID/ap-south-1
```

### Enable Bedrock model access (one time, free)
1. Open AWS Console → Amazon Bedrock → Model access
2. Request access to:
   - **Amazon Titan Embeddings G1 - Text** (amazon.titan-embed-text-v1)
   - **Claude Haiku** (anthropic.claude-haiku-20240307-v1:0)
3. Takes 1–5 minutes to activate

---

## Step-by-step deployment

### Step 1 — Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot-bedrock
cd rag-chatbot-bedrock/infra
npm install
```

### Step 2 — Deploy infrastructure
```bash
cd infra
cdk deploy

# This creates:
# - S3 bucket for PDFs
# - DynamoDB table for document metadata
# - EC2 t2.micro running ChromaDB in Docker
# - 2 Lambda functions (ingestor + query)
# - API Gateway with CORS enabled
```

**Wait for the deploy to finish (~5 minutes). Copy these outputs:**
```
Outputs:
RagChatbotStack.ApiUrl = https://abc123.execute-api.ap-south-1.amazonaws.com/prod/
RagChatbotStack.BucketName = rag-docs-123456789-ap-south-1
RagChatbotStack.ChromaInstanceIp = 13.x.x.x
```

### Step 3 — Wait for ChromaDB to start
EC2 runs the Docker install on first boot. Wait 2–3 minutes, then verify:
```bash
curl http://YOUR_CHROMA_IP:8000/api/v1/heartbeat
# Expected: {"nanosecond heartbeat": ...}
```

### Step 4 — Configure frontend
Open `frontend/index.html` and update:
```js
const API_URL = "https://abc123.execute-api.ap-south-1.amazonaws.com/prod/";
const BUCKET_NAME = "rag-docs-123456789-ap-south-1";
```

### Step 5 — Test with curl
```bash
# Upload a PDF directly to S3 (triggers ingestor Lambda automatically)
aws s3 cp sample.pdf s3://YOUR_BUCKET_NAME/

# Watch ingestor Lambda logs
aws logs tail /aws/lambda/rag-ingestor --follow

# Wait for "Document fully ingested" in logs (~30 seconds)

# Query the API
curl -X POST https://YOUR_API_URL/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### Step 6 — Open the frontend
```bash
# Serve locally
cd frontend
python3 -m http.server 3000
# Open http://localhost:3000
```

---

## Testing it properly (for your README)

### 1. Upload a real document
Use a product manual, terms of service, or any PDF with known content.

### 2. Ask specific questions
- "What is the return policy?" → should cite the exact section
- "Who should I contact for support?" → should find the contact info
- "What are the system requirements?" → should retrieve the specs section

### 3. Screenshot everything for your README
- The QuickSight-style chat interface showing an answer with sources
- The AWS Console showing both Lambda functions
- The X-Ray service map (if enabled)
- The ChromaDB collection info (curl http://CHROMA_IP:8000/api/v1/collections)

---

## What to add next (for interviews — "what would you do in production?")

1. **Authentication** — Add Cognito User Pools to API Gateway. Right now anyone with the URL can query your documents.
2. **Per-user document isolation** — ChromaDB supports named collections. Give each user their own collection: `user_{cognito_sub}`.
3. **Streaming responses** — Use Bedrock's `invoke_model_with_response_stream` so the answer streams token by token (like ChatGPT).
4. **Reranking** — After ChromaDB retrieves top-k chunks, use a cross-encoder reranker to re-score them. Improves answer accuracy.
5. **Conversation memory** — Store chat history in DynamoDB, pass last 3 turns to Claude as context.
6. **Better chunking** — Use semantic chunking (split on sentence boundaries) instead of word-count chunking.
7. **Swap ChromaDB for OpenSearch** — For production scale, use OpenSearch Serverless with k-NN. More expensive but managed, scalable, and HIPAA eligible.

---

## Cleanup (destroy everything)
```bash
cd infra
cdk destroy
# Also terminate the EC2 instance from the console if it doesn't auto-delete
```

---

## Architecture decision records (for interviewers)

**ADR 1: Why RAG over fine-tuning?**
Fine-tuning bakes knowledge into the model weights — you can't update it without retraining. RAG retrieves fresh context at inference time, so adding a new document is as simple as re-ingesting. For a document Q&A use case, RAG is the correct choice.

**ADR 2: Why 500-word chunks with 50-word overlap?**
Too-small chunks lose context. Too-large chunks dilute relevance (a 2000-word chunk retrieved for a narrow question brings too much irrelevant text into the prompt). 500 words ≈ 650 tokens — fits comfortably in the prompt without exhausting context. 50-word overlap prevents losing information at chunk boundaries.

**ADR 3: Why cosine similarity, not Euclidean distance?**
Cosine similarity measures the angle between vectors, ignoring magnitude. For text embeddings this is better — a long paragraph and a short sentence with similar meaning will have different vector magnitudes but similar direction. Cosine handles this correctly.

**ADR 4: Why return source chunks to the user?**
Hallucination detection. If the model cites a source chunk, the user can verify it. In production, a trust layer checks that the answer's claims are grounded in the retrieved context.
