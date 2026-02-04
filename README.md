# 🌱 Smart Agriculture RAG Assistant

AI-powered question-answering system for smart agriculture using Retrieval Augmented Generation (RAG) with LangChain, LangGraph, and Pinecone.

## 🎥 Demo Video

Watch the system in action: **[System Demonstration on YouTube](https://youtu.be/C_KdRnQYtls)**


---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

---

## 🎯 Overview

This project implements an intelligent Q&A system specialized in smart agriculture topics. It uses RAG (Retrieval Augmented Generation) to provide accurate, context-aware answers by:

1. **Ingesting** agricultural documents (PDFs) into a vector database
2. **Retrieving** relevant document chunks based on user queries
3. **Generating** comprehensive answers using GPT models with retrieved context

The system leverages LangGraph for sophisticated agent orchestration, allowing the AI to decide when to retrieve information and how to formulate responses.

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                      │
│                      (Streamlit Chat UI)                    │
│                          main.py                            │
│  - Session Management                                       │
│  - Chat History                                             │
│  - Error Handling                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend Layer                          │
│                    (backend/core.py)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              RAGAgent (LangGraph)                    │   │
│  │                                                      │   │
│  │  ┌─────────────┐      ┌──────────────┐               │   │
│  │  │  Agent Node │─────▶│  Conditional │              │   │
│  │  │  (Reasoning)│      │  Edge Logic  │               │  │
│  │  └─────────────┘      └──────┬───────┘               │  │
│  │                               │                      │  │
│  │                       Need Context?                  │  │
│  │                        ┌─────┴─────┐                 │  │
│  │                        │           │                 │  │
│  │                       Yes         No                 │  │
│  │                        │           │                 │  │
│  │              ┌─────────▼───┐      │                  │  │
│  │              │  RAG Node   │      │                  │  │
│  │              │  (Retrieve) │      │                  │  │
│  │              └──────┬──────┘      │                  │  │
│  │                     │             │                  │  │
│  │                     └──────┬──────┘                  │  │
│  │                            ▼                         │  │
│  │                    ┌──────────────┐                  │  │
│  │                    │  END (Answer)│                  │  │
│  │                    └──────────────┘                  │  │
│  └───────────────────────────────────────────────────   ┘  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│                                                             │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │  Vector Database │         │   Document Ingestion    │  │
│  │   (Pinecone)     │◀────────│   (rag/ingestion.py)    │  │
│  │                  │         │                         │  │
│  │  - Embeddings    │         │  - PDF Loading          │  │
│  │  - Similarity    │         │  - Text Splitting       │  │
│  │    Search        │         │  - Embedding            │  │
│  └──────────────────┘         │  - Storage              │  │
│                                └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion Flow:**
   ```
   PDF Files → Load → Clean Metadata → Split (1000 chars) → 
   Embed (OpenAI) → Store in Pinecone
   ```

2. **Query Flow:**
   ```
   User Query → Streamlit UI → RAGAgent → 
   Decision (Need Context?) →
     Yes: Retrieve from Pinecone → Generate Answer → Return
     No: Generate Direct Answer → Return
   ```

### Components

#### 1. **UI Layer** (`main.py`)
- **Streamlit-based chat interface**
- Manages conversation history
- Handles user input/output
- Provides visual feedback (spinners, errors)
- Sidebar with information and controls
- Custom CSS styling for better UX

#### 2. **Backend Layer** (`backend/core.py`)
- **RAGAgent Class**: Orchestrates the entire RAG pipeline
- **LangGraph Integration**: Manages agent decision-making
- **Tool System**: Defines retrieval tool for context gathering
- **State Management**: Tracks conversation and tool calls

#### 3. **Data Layer** (`rag/ingestion.py`, `rag/ingestion_class.py`)
- **PDF Processing**: Loads and parses agricultural documents
- **Text Chunking**: Splits documents into manageable pieces
- **Embedding Generation**: Converts text to vectors
- **Vector Storage**: Persists embeddings in Pinecone

---

## 🛠️ Technology Stack

### Core Technologies

| Technology | Version | Purpose | Why Chosen |
|------------|---------|---------|------------|
| **Python** | 3.11+ | Programming Language | Broad ML/AI ecosystem support |
| **LangChain** | 1.2.8 | LLM Framework | Industry-standard RAG framework |
| **LangGraph** | 1.0.5 | Agent Orchestration | Advanced workflow control for agents |
| **Streamlit** | 1.53.1 | Web UI | Rapid prototyping, beautiful UI out-of-the-box |
| **Pinecone** | 7.3.0 | Vector Database | **Cloud-hosted, production-ready for live applications. No local setup needed - perfect for demonstrating real-world deployment scenarios** |
| **OpenAI GPT**  | Language Model | High-quality, fast responses |
| **OpenAI Embeddings** | text-embedding-3-small | Text Embeddings | **Best balance of quality and speed for short-term projects. Proven accuracy with minimal latency** |

### Supporting Libraries

- **pypdf** (6.6.2): PDF parsing and text extraction
- **python-dotenv** (1.2.1): Environment variable management
- **langchain-community** (0.4.1): Document loaders and utilities
- **langchain-text-splitters** (1.1.0): Text chunking algorithms

### Architecture Decisions

#### Why Pinecone?
**Pinecone was chosen for several strategic reasons:**

   - Demonstrates scalability from prototype to production
   - Prior experience with Pinecone
   

#### Why OpenAI Embeddings?
**text-embedding-3-small was selected because:**

1. **Quality**: State-of-the-art embedding quality for semantic search
4. **Reliability**: Production-tested, stable API
5. **Project Timeline**: For a short-term project, using proven technology reduces risk and accelerates development


---

## ✨ Features

- ✅ **Intelligent Document Retrieval**: Uses semantic search to find relevant information
- ✅ **Context-Aware Responses**: Generates answers based on actual document content
- ✅ **Agent Orchestration**: LangGraph manages when to retrieve vs. answer directly
- ✅ **Beautiful UI**: Modern chat interface with Streamlit
- ✅ **Conversation Memory**: Maintains chat history for follow-up questions
- ✅ **Error Handling**: Graceful error messages and recovery
- ✅ **PDF Support**: Processes agricultural PDFs automatically
- ✅ **Scalable Architecture**: Ready for production deployment
- ✅ **Class-Based Design**: Modular, maintainable code structure

---

## 🔧 Prerequisites

Before you begin, ensure you have the following:

### Required Software
- **Python 3.11 or 3.12** (Python 3.14 not fully supported by all dependencies)
- **Git** for version control
- **pip** (comes with Python)

### Required API Keys

#### 1. OpenAI API Key
To use GPT models and embeddings:

1. **Sign up** at [OpenAI Platform](https://platform.openai.com/)
2. **Add payment method** (required for API access)
3. **Create API key**:
   - Go to [API Keys page](https://platform.openai.com/api-keys)
   - Click **"Create new secret key"**
   - Give it a name (e.g., "Smart Agriculture RAG")
   - **Copy the key immediately** (you won't see it again!)
   - Save it securely


#### 2. Pinecone API Key
To use the vector database:

1. **Sign up** at [Pinecone](https://www.pinecone.io/)
2. **Free tier available**: 1 index, 100K vectors (sufficient for this project)
3. **Get API key**:
   - Go to [Pinecone Console](https://app.pinecone.io/)
   - Click on **"API Keys"** in the sidebar
   - Copy your API key
4. **Create an index**:
   - Click **"Create Index"**
   - **Name**: `rag-agriculture`
   - **Dimensions**: `1536` (for text-embedding-3-small)
   - **Metric**: `cosine`
   - **Region**: Choose closest to your location
   - Click **"Create Index"**

**Note:** Keep both API keys secure and never share them publicly!

---

## 📥 Installation

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd agentic_RAG
```

### Step 2: Create Virtual Environment

**Option A: Using uv (Recommended - Fast)**
```bash
pip install uv
uv venv .venv --python 3.11
```

**Option B: Using standard venv**
```bash
python3.11 -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows (Git Bash):**
```bash
source .venv/Scripts/activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed langchain-1.2.8 streamlit-1.53.1 ...
```

### Step 5: Configure Environment Variables

1. Create a `.env` file in the project root:
   ```bash
   touch .env
   ```

2. Add your API keys:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here
   
   # Pinecone Configuration
   PINECONE_API_KEY=your-pinecone-api-key-here
   INDEX_NAME=rag-agriculture
   


---

## 🚀 Usage

### Step 1: Prepare Your Documents

1. Create the docs folder (if not exists):
   ```bash
   mkdir -p rag/docs
   ```

2. Add your PDF files:
   ```bash
   cp /path/to/your/agriculture-pdfs/*.pdf rag/docs/
   ```

### Step 2: Run Document Ingestion

Process and store documents in Pinecone:

```bash
.venv/Scripts/python.exe rag/ingestion_class.py
```

**Expected Output:**
```
============================================================
🌱 Starting Smart Agriculture Document Ingestion Pipeline
============================================================

📁 Step 1: Loading PDF files...
✅ Loaded 109 documents from docs

🧹 Step 2: Cleaning document metadata...

✂️  Step 3: Splitting documents into chunks...
✅ Split 109 documents into 420 chunks

💾 Step 4: Storing in Pinecone vector database...
✅ Successfully stored all chunks in Pinecone!

============================================================
✅ Ingestion pipeline completed successfully!
============================================================
```


```

### Step 3: Start the Application

Run the Streamlit app:

```bash
.venv/Scripts/streamlit run main.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

The application will automatically open in your default browser.

### Step 4: Ask Questions!

Examples of questions you can ask:
- "What is precision agriculture?"
- "How do IoT sensors improve crop yields?"
- "Explain drip irrigation systems"
- "What are the benefits of smart farming?"

---

## 📁 Project Structure

```
agentic_RAG/
├── backend/
│   └── core.py                 # RAGAgent class with LangGraph orchestration
├── rag/
│   ├── docs/                  # PDF documents folder 
│   └── ingestion_class.py    # Class-based ingestion pipeline
├── main.py                    # Streamlit application (UI + entry point)
├── .env                       # Environment variables (not in git)
├── requirements.txt          # Pip dependencies
└── README.md                 # This file
```

### Key Files

#### `main.py`
Streamlit application containing:
- Page configuration and custom CSS
- Chat interface with message history
- Sidebar with info and controls
- Integration with backend RAGAgent
- Error handling and user feedback

#### `backend/core.py`
Contains the `RAGAgent` class:
- `__init__()`: Initialize embeddings, vectorstore, and LangGraph
- `_create_retrieve_tool()`: Define document retrieval tool
- `_build_graph()`: Construct LangGraph workflow
- `query()`: Main entry point for queries

#### `rag/ingestion_class.py`
Class-based document processing:
- `DocumentIngestor`: Main class for ingestion pipeline
- `load_pdf_files()`: Load PDFs from directory
- `clean_documents()`: Clean metadata
- `split_documents()`: Chunk text with configurable size
- `store_in_vectordb()`: Embed and store in Pinecone
- `ingest()`: Run complete pipeline

---




