# Agentic RAG with Knowledge Graph

Agentic knowledge retrieval redefined with an AI agent system that combines traditional RAG (vector search) with knowledge graph capabilities to analyze and provide insights about big tech companies and their AI initiatives. The system uses PostgreSQL with pgvector for semantic search and Neo4j with Graphiti for temporal knowledge graphs. The goal is to create Agentic RAG at its finest.

Built with:

- Pydantic AI for the AI Agent Framework
- Graphiti for the Knowledge Graph
- Postgres with PGVector for the Vector Database
- Neo4j for the Knowledge Graph Engine (Graphiti connects to this)
- FastAPI for the Agent API
- Claude Code for the AI Coding Assistant (See `CLAUDE.md`, `PLANNING.md`, and `TASK.md`)

## Overview

This system includes three main components:

1. **Document Ingestion Pipeline**: Processes markdown documents using semantic chunking and builds both vector embeddings and knowledge graph relationships
2. **AI Agent Interface**: A conversational agent powered by Pydantic AI that can search across both vector database and knowledge graph
3. **Streaming API**: FastAPI backend with real-time streaming responses and comprehensive search capabilities

## Prerequisites

- Python 3.11 or higher
- PostgreSQL database (such as Neon)
- Neo4j database (for knowledge graph)
- LLM Provider API key (OpenAI, Ollama, Gemini, etc.)

## Installation

### 1. Set up a virtual environment

```bash
# Create and activate virtual environment
python -m venv venv       # python3 on Linux
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up required tables in Postgres

Execute the SQL in `sql/schema.sql` to create all necessary tables, indexes, and functions.

Be sure to change the embedding dimensions on lines 31, 67, and 100 based on your embedding model. OpenAI's text-embedding-3-small is 1536 and nomic-embed-text from Ollama is 768 dimensions, for reference.

Note that this script will drop all tables before creating/recreating!

### 4. Set up Neo4j

You have a couple easy options for setting up Neo4j:

#### Option A: Using Local-AI-Packaged (Simplified setup - Recommended)
1. Clone the repository: `git clone https://github.com/coleam00/local-ai-packaged`
2. Follow the installation instructions to set up Neo4j through the package
3. Note the username and password you set in .env and the URI will be bolt://localhost:7687

#### Option B: Using Neo4j Desktop
1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and add a local DBMS
3. Start the DBMS and set a password
4. Note the connection details (URI, username, password)

### 5. Configure environment variables

Create a `.env` file in the project root:

```bash
# Database Configuration (example Neon connection string)
DATABASE_URL=postgresql://username:password@ep-example-12345.us-east-2.aws.neon.tech/neondb

# Neo4j Configuration  
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Provider Configuration (choose one)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_CHOICE=gpt-4o-mini

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-api-key
EMBEDDING_MODEL=text-embedding-3-small

# Ingestion Configuration
INGESTION_LLM_CHOICE=gpt-4.1-nano  # Faster model for processing

# Application Configuration
APP_ENV=development
LOG_LEVEL=INFO
APP_PORT=8058
```

For other LLM providers:
```bash
# Ollama (Local)
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_CHOICE=qwen2.5:14b-instruct

# OpenRouter
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-openrouter-key
LLM_CHOICE=anthropic/claude-3-5-sonnet

# Gemini
LLM_PROVIDER=gemini
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
LLM_API_KEY=your-gemini-key
LLM_CHOICE=gemini-2.5-flash
```

## Quick Start

### 1. Prepare Your Documents

Add your markdown documents to the `documents/` folder:

```bash
mkdir -p documents
# Add your markdown files here
```

**Note**: For a comprehensive example with extensive content, you can copy the provided `markdowns` folder:
```bash
cp -r markdowns/* documents/
```

### 2. Run Document Ingestion

**Important**: You must run ingestion first to populate the databases before the agent can provide meaningful responses.

```bash
# Basic ingestion with semantic chunking
python -m ingestion.ingest

# Clean existing data and re-ingest everything
python -m ingestion.ingest --clean
```

The ingestion process will:
- Parse and semantically chunk your documents
- Generate embeddings for vector search
- Extract entities and relationships for the knowledge graph
- Store everything in PostgreSQL and Neo4j

NOTE that this can take a while because knowledge graphs are very computationally expensive!

### 3. Configure Agent Behavior (Optional)

Before running the API server, you can customize when the agent uses different tools by modifying the system prompt in `agent/prompts.py`. The system prompt controls:
- When to use vector search vs knowledge graph search
- How to combine results from different sources
- The agent's reasoning strategy for tool selection

### 4. Start the API Server (Terminal 1)

```bash
# Start the FastAPI server
python -m agent.api

# Server will be available at http://localhost:8058
```

### 6. Test the System

#### Health Check
```bash
curl http://localhost:8058/health
```

#### Chat with the Multi-Agent System
```bash
curl -X POST "http://localhost:8058/chat/mas" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are knowledge graphs and how do they work?",
  }'
```

```

## API Documentation

Visit http://localhost:8058/docs for interactive API documentation once the server is running.

## Key Features

- **Hybrid Search**: Seamlessly combines vector similarity and graph traversal
- **Temporal Knowledge**: Tracks how information changes over time
- **Streaming Responses**: Real-time AI responses with Server-Sent Events
- **Flexible Providers**: Support for multiple LLM and embedding providers
- **Semantic Chunking**: Intelligent document splitting using LLM analysis
- **Production Ready**: Comprehensive testing, logging, and error handling

## Project Structure

```
.
├── agent
│   ├── agent.py
│   ├── api.py
│   ├── db_utils.py
│   ├── graph_utils.py
│   ├── models.py
│   ├── prompts.py
│   ├── providers.py
│   └── tools.py
├── agents
│   ├── base_agent.py
│   ├── coordinator.py
│   ├── graph.py
│   ├── specialists
│   │   ├── knowledge_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── synthesis_agent.py
│   │   └── validation_agent.py
│   ├── state.py
│   └── utils
│       ├── result_aggregation.py
│       └── task_decomposition.py
├── benchmark
│   ├── benchmark_runner.py
│   ├── results2.json
│   ├── test_queries.json
│   └── visualizer.py
├── ingestion
│   ├── chunker.py
│   ├── embedder.py
│   ├── graph_builder.py
│   └── ingest.py
├── markdowns
├── pdfs
├── preprocessing
│   └── pdf_converter.py
├── requirements.txt
├── sql
│   └── schema.sql
├── venv
├── .env
├── .gitignore
└── README.md
```