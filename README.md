# Twitch RAG LLM Agent

A lightweight **Retrieval-Augmented Generation (RAG) agent** that combines document retrieval with Large Language Models (LLMs) for context-aware and accurate responses. The system retrieves a document filled with 1MB of twitch chat response dataset, used as an attempt to simulate behavior of twitch chat. This system integrates monitoring, tracing, and observability tools such as Prometheus with Grafana UI and OpenTelemetry with Jaeger UI for production-ready deployments.

Previous versions used retrievers to create qa pipelines, I intergrate with functional middleware for system context enrichment (also the fact that 1.0.0+ doesnt have langchain_classic.chains module properly implemented).

---

## Overview

This project implements a small-scale RAG system with:

* **Document retrieval** using FAISS integrated with agent middleware prompts
* **LLM agent** integration via LangChain 1.0.0+
* **Context-aware response generation** using both user posted documents and pregenerated twitch chat documents
* **Observability** via Prometheus, Grafana, OpenTelemetry, and Jaeger
* **Lightweight, modular implementation** suitable for weaker CPU-based servers

The backend is implemented in **FastAPI** with automatic OpenAPI schema generation, making it easy to integrate with clients or other services.

The frontend lives in the `chat-bot` directory and is a **React Native (Expo + TypeScript)** application using **Kubb code generation** for type-safe integration with both the chat backend and the Python encryption server.

---

## Features

* **FastAPI backend** with auto-generated OpenAPI schema
* **RAG agent** with vector store caching using FAISS
* **PDF and in-memory document ingestion**
* **Support for Twitch chat style retrieval and context injection**
* **Monitoring and tracing**:

  * **Prometheus**: metrics collection
  * **Grafana**: dashboards
  * **OpenTelemetry**: distributed tracing
  * **Jaeger**: trace visualization
* **Mobile frontend** using TypeScript + Expo + Kubb for strong typing

---

## Architecture

```
┌───────────────┐          ┌───────────────┐
│ FastAPI LLM   │ <------> │ FAISS Vector  │
│ Backend       │          │ Store         │
└───────────────┘          └───────────────┘
        │
        │ OpenAPI / REST
        ▼
┌───────────────┐
│ React Native  │
│ Expo Frontend │
│ chat-bot/     │
└───────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Observability & Metrics      │
│ - Prometheus                 │
│ - Grafana                    │
│ - OpenTelemetry              │
│ - Jaeger                     │
└──────────────────────────────┘
```

---

## Getting Started

### Prerequisites

* Docker & Docker Compose
* Python 3.10+ environment
* Node.js 20+ (for mobile frontend)

### Installation

```bash
git clone https://github.com/nam2184/tiny-rag-llm-agent.git
cd tiny-rag-llm-agent
compose -f 'docker-compose.yml' up -d --build
```

* Backend models and vector stores are persisted in `tiny-rag-llm-agent/models` and `tiny-rag-llm-agent/vector_store`.
* Grafana dashboards and provisioning are stored in `grafana/provisioning`.

---

## Usage

* Ingest documents (PDF or in-memory) and query the RAG agent.
* The backend automatically exposes OpenAPI schema for client generation.

### Mobile Frontend

* Navigate to the `chat-bot` directory:

```bash
cd chat-bot
npm install
npm run start
```

* The frontend communicates with the backend using **Kubb-generated TypeScript types** for strong type safety.

### Observability
Example ports
* **Prometheus** metrics: `http://localhost:9090`
* **Grafana dashboards**: `http://localhost:3000`
* **Jaeger traces**: `http://localhost:16686`
* **OpenTelemetry** automatically collects traces from the FastAPI backend.

---

## Contributing

Contributions are welcome! You can submit Pull Requests for bug fixes, feature enhancements, or observability improvements.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

