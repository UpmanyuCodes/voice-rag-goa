# 🌴 VoiceRAG Goa — HH Goa 2026 Task 2

**Voice-Enabled Retrieval-Augmented Generation** across 14 Indic languages, built for the HH Goa 2026 Shortlisting Task 2.

[![#RAGInGoa](https://img.shields.io/badge/%23RAGInGoa-2026-2D6A4F?style=for-the-badge)](https://forms.gle/MNvCjcv23Hn2Eeu58)
![Tests](https://img.shields.io/badge/Tests-42%20Passing-52A77A?style=for-the-badge)
![Latency](https://img.shields.io/badge/Latency-<200ms-C9A96E?style=for-the-badge)

---

## 🎯 Pipeline Shape

```
Voice Input → Sarvam STT → Safety Guardrail → Vector Retrieval → Domain Guardrail → Groq LLM → Grounding Check → Answer
```

---

## ✅ Feature Checklist (per PDF requirements)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| STT: Sarvam or ElevenLabs | ✅ | `src/stt/sarvam_stt.py` + ElevenLabs fallback |
| 4 Chunking Strategies | ✅ | Fixed Window, Semantic Boundary, Metadata-Aware, Hierarchical |
| Latency < 200ms (retrieval→answer) | ✅ | In-memory NumPy cosine + Groq ~80ms |
| P50/P70/P100 Benchmark | ✅ | `src/benchmark/run_benchmarks.py` |
| Model Harness (tool calls, retries, structured I/O) | ✅ | `src/harness/orchestrator.py` |
| Guardrails (safety, off-topic, hallucination) | ✅ | `src/guardrails/` (3 independent layers) |
| Dataset: ai4bharat/MSMARCO-XI | ✅ | `src/data/loader.py` |
| Live working link | ✅ | [Deploy below] |
| GitHub repo | ✅ | This repo |

---

## ⚡ Latency Architecture (Sub-200ms Budget)

| Stage | Budget |
|-------|--------|
| Query Embedding (char n-gram, in-process) | ~0.5ms |
| NumPy cosine similarity search (in-memory) | ~2-5ms |
| Guardrails pre + domain check | ~1-3ms |
| Groq LLaMA 3.3-70B generation | ~60-120ms |
| Grounding guardrail | ~1ms |
| LRU Cache overhead | ~0.2ms |
| **Total P50 (cached)** | **< 5ms** |
| **Total P50 (uncached)** | **~70-130ms** |

> **Note:** STT latency (Sarvam ~300-600ms) is a separate step. The 200ms target applies to: Chunking/Retrieval + Vector DB + Generation, as specified in the PDF.

---

## 🧩 Chunking Strategies

1. **Fixed Window** (`fixed_window`) — Character-level sliding windows with configurable overlap
2. **Semantic Boundary** (`semantic_boundary`) — Splits at Indic sentence boundaries (।, ॥, ?, !)
3. **Metadata-Aware** (`metadata_aware`) — Enriches chunks with language, query type, and relevance tags
4. **Hierarchical** (`hierarchical`) — Parent (full passage) + Child (fine-grained) multi-vector index

---

## 🛡️ Guardrails (3 Layers)

1. **Pre-Retrieval Safety** — Prompt injection, bypass attempts, unsafe keywords
2. **Domain Off-Topic Check** — Validates retrieved passages are relevant to the query
3. **Post-Generation Grounding** — Confirms the answer is grounded in retrieved context

---

## 🚀 Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
cp .env.example .env
# Fill in SARVAM_API_KEY and GROQ_API_KEY
```

### 3. Start the API server
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000** for the Tropical Goa UI.

### 4. Run unit tests
```bash
python -m pytest tests/ -v
```

### 5. Run latency benchmark
```bash
python -m src.benchmark.run_benchmarks
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Tropical Goa Web UI |
| `/query` | POST | Text query → RAG answer |
| `/voice` | POST | Audio upload → STT → RAG |
| `/voice-base64` | POST | Base64 audio → STT → RAG |
| `/benchmark` | GET | Run P50/P70/P100 latency suite |
| `/strategies` | GET | List chunking strategies |
| `/health` | GET | System health + chunk count |
| `/cache/stats` | GET | LRU cache hit/miss stats |

---

## 🗂️ Project Structure

```
voice-rag-goa-1/
├── src/
│   ├── api/          # FastAPI backend (main.py, routes.py)
│   ├── benchmark/    # Latency profiler & test queries
│   ├── chunking/     # 4 chunking strategies + registry
│   ├── data/         # MSMARCO-XI loader & sample data
│   ├── guardrails/   # Safety, off-topic, grounding
│   ├── harness/      # Orchestrator, schemas, LLM client, retry
│   ├── retrieval/    # Vector store, embeddings, LRU cache
│   └── stt/          # Sarvam, ElevenLabs, local STT
├── web/              # Tropical Goa UI (index.html, styles.css, app.js)
├── tests/            # 42 unit + integration tests
├── Dockerfile
└── requirements.txt
```

---

## 📹 Videos

- **Video 1 (Process):** Team working — 90 seconds — `#RAGInGoa`
- **Video 2 (Demo):** End-to-end voice demo — `#RAGInGoa`

---

## #RAGInGoa 🌴
