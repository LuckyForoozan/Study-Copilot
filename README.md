# 📚 Study Copilot: Multimodal Vision RAG & Agentic Assistant

**Study Copilot** is an AI-powered study and document research assistant designed specifically for complex documents and lecture slides. Built with **Multimodal Vision RAG**, **Hybrid Search (BM25 + Dense Vectors)**, and **Autonomous Tool Calling**, it understands both written text and visual diagrams (charts, flowcharts, infographics) with exact page citations.



## 🌟 Key Features

* **👁️ Multimodal Vision RAG:** Analyzes complex slide visuals, diagrams, icons, and infographics using `Llama 3.2 11B Vision` alongside raw text extraction.
* **🔍 Hybrid Search (BM25 + Dense Retrieval):** Combines exact keyword matching (BM25) with semantic vector search (ChromaDB) to never miss exact headings, slide titles, or concepts.
* **🎯 Cross-Encoder Reranking:** Uses `ms-marco-MiniLM-L-6-v2` to re-score candidate chunks for maximum precision.
* **🖼️ Visual Slide Preview:** Displays the exact visual slide thumbnail in Streamlit alongside retrieved answers for verification.
* **🤖 Autonomous AI Agent:** Powered by `gpt-oss-120b` on Groq, capable of deciding when to answer from slides, generate flashcards, or search the live web.
* **🌐 Web Search Fallback:** Automatically queries DuckDuckGo if the requested information is absent from the uploaded document.
* **🎴 Flashcard Generator:** Instant study quizzes and flashcards derived strictly from your document content.
* **💬 Conversational Memory:** Multi-turn chat interface with sliding context window history.


## 🏗️ Architecture & Pipeline


[PDF Upload]
     │
     ├──► PyMuPDF: Page-to-Image Rendering
     │        └──► Groq Llama 3.2 Vision (Diagrams, Icons & Layout Analysis)
     └──► Text Extraction & Recursive Chunking
              │
              ├──► ChromaDB (Dense Vector Search via all-MiniLM-L6-v2)
              └──► BM25 (Sparse Exact-Keyword Search)
                       │
                 [Hybrid Candidate Pool]
                       │
                 [Cross-Encoder Reranker]
                       │
                 [Top Context + Slide Images]
                       │
                 [Groq Agent Orchestrator]
                   ├── Tool: Document Q&A (Page Citations)
                   ├── Tool: DuckDuckGo Live Search
                   └── Tool: Flashcard Generator
                       │
                 [Streamlit Interactive UI]




## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM & Inference:** Groq API (`openai/gpt-oss-120b`, `llama-3.2-11b-vision-preview`)
* **Vector Database:** ChromaDB (In-Memory Ephemeral Client)
* **Embedding Model:** `all-MiniLM-L6-v2` (Default Chroma Embedding Function)
* **Reranker:** FastEmbed (`Xenova/ms-marco-MiniLM-L-6-v2`)
* **PDF & Vision Processing:** PyMuPDF (`fitz`), Pillow
* **External Tools:** DuckDuckGo Search (`duckduckgo-search`)



