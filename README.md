
## 🧠 Intelligent Q&A Assistant (RAG + LLM Agents)

An end-to-end LLM-powered Question Answering system that uses Retrieval-Augmented Generation (RAG), LangChain Agents, and tool calling to answer domain-specific questions from large collections of documents.

---


## 🚀 Features

- 📄Document Ingestion & Preprocessing


  - Parses PDFs using PyMuPDF & Unstructured

  - Text chunking + embedding with sentence-transformers

  - Vector storage with FAISS (metadata-aware retrieval)

- 🔍 Retrieval-Augmented Generation (RAG)

  - Contextual Q&A with top-k retrieval

  - Reranking for better relevance (Cohere / Hugging Face models)

  - Fallback to LLM-only response when no strong context

🤖 Agentic Tool Use (LangChain Agents)

Calculator (Wolfram plugin)

Web search (Bing API)

File reader for CSVs / structured data

Human-in-the-loop escalation

📝 Prompt Engineering & Memory

Structured system prompts for safe output

Conversation history with buffer + summarization

Confidence estimation to avoid hallucination

📊 Evaluation & Logging

Benchmarked with EM, F1, BLEU, and latency metrics

Logs all responses, sources, and feedback

Feedback loop for iterative model improvement

☁️ Deployment

FastAPI backend containerized with Docker

GCP Cloud Run deployment (works on AWS/Azure too)

CI/CD with GitHub Actions + Cloud Build
