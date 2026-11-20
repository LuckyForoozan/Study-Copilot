import io
import re
import math
import json
import hashlib
import base64
from collections import Counter
import fitz  # PyMuPDF
import chromadb
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastembed.rerank.cross_encoder import TextCrossEncoder

class BM25:
    """Zero-dependency BM25 implementation for exact keyword and heading matching."""
    def __init__(self, docs):
        self.raw_docs = docs
        self.tokenized_docs = [re.findall(r"\w+", d.lower()) for d in docs]
        self.size = len(docs)
        self.avgdl = sum(len(d) for d in self.tokenized_docs) / (self.size or 1)
        self.dfs = Counter()
        for d in self.tokenized_docs:
            for w in set(d):
                self.dfs[w] += 1
        self.idfs = {
            w: math.log(1 + (self.size - c + 0.5) / (c + 0.5))
            for w, c in self.dfs.items()
        }

    def search(self, query, top_k=10):
        q_tokens = re.findall(r"\w+", query.lower())
        scored = []
        for idx, d in enumerate(self.tokenized_docs):
            counts = Counter(d)
            doc_len = len(d)
            score = sum(
                self.idfs.get(w, 0) * (counts[w] * 2.5) / (counts[w] + 1.5 * (0.25 + 0.75 * doc_len / (self.avgdl or 1)))
                for w in q_tokens
                if w in counts
            )
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scored[:top_k]]

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

st.title("Study Copilot")
MODEL = "openai/gpt-oss-120b"

with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Clear Cache & Re-process PDF"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! Reloading...")
        st.rerun()

@st.cache_resource
def get_embedder():
    return DefaultEmbeddingFunction()  # all-MiniLM-L6-v2, runs locally

@st.cache_resource
def get_reranker():
    return TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_llm():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def render_page_to_png(page):
    """Renders a PDF page to PNG image bytes."""
    pix = page.get_pixmap(dpi=150)
    return pix.tobytes("png")

def describe_slide(image_bytes):
    """Sends the slide image to a vision model to transcribe text and describe diagrams."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = get_llm().chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Transcribe all text on this slide. In detail, describe any diagrams, charts, icons, visual layouts, and relationships."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Vision analysis error: {e}")
        return ""


@st.cache_data(show_spinner="Analyzing PDF slides & visuals with Vision model...")
def process(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    # Increased chunk_size to 1500 to keep text and vision descriptions unified
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = []
    page_images = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        p_num = page_num + 1
        
        # 1. Render page to image
        img_bytes = render_page_to_png(page)
        page_images[p_num] = img_bytes

        # 2. Extract raw text + vision description
        raw_text = page.get_text() or ""
        visual_desc = describe_slide(img_bytes)
        
        full_page_content = f"Slide {p_num} Text:\n{raw_text}\n\nVisuals & Diagram Details:\n{visual_desc}"
        
        # 3. Chunk and collect
        for piece in splitter.split_text(full_page_content):
            if len(piece.strip()) > 30:
                chunks.append({"text": piece, "page": p_num})

    embeddings = get_embedder()([c["text"] for c in chunks])
    return chunks, embeddings, len(doc), page_images

@st.cache_resource
def build_collection(file_bytes):
    chunks, embeddings, _, _ = process(file_bytes)
    client = chromadb.EphemeralClient()
    name = "notes_" + hashlib.md5(file_bytes).hexdigest()
    collection = client.get_or_create_collection(name=name)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=[[float(x) for x in e] for e in embeddings],
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return collection

# ---------- tools the agent can use ----------
def search_web(query):
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in results) or "No results."
    except Exception as e:
        return f"Web search failed: {e}"

def generate_flashcards(topic, n, context):
    r = get_llm().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Create flashcards ONLY from the context. Format: Q: ... / A: ... (cite the page)."},
            {"role": "user", "content": f"Context:\n{context}\n\nCreate {n} flashcards about: {topic}"},
        ],
    )
    return r.choices[0].message.content

TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web. Use ONLY when the answer is not in the document context.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "generate_flashcards",
        "description": "Create flashcards from the document context. Use when the user asks for flashcards or a quiz.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}, "n": {"type": "integer"}}, "required": ["topic"]},
    }},
]

uploaded = st.file_uploader("Upload your PDF", type="pdf")

if uploaded:
    chunks, embeddings, n_pages,page_images = process(uploaded.getvalue())
    st.session_state["page_images"] = page_images
    st.session_state["chunks"] = chunks
    st.session_state["bm25"] = BM25([c["text"] for c in chunks])
    collection = build_collection(uploaded.getvalue())
    st.success(f"{n_pages} pages → {len(chunks)} chunks → {len(embeddings)} embeddings ({len(embeddings[0])} numbers each)")
    st.info(f"Stored {collection.count()} chunks in ChromaDB")

    i = st.number_input("Chunk number", min_value=1, max_value=len(chunks), value=1, step=1)
    st.write(f"Page {chunks[i-1]['page']}")
    st.text(chunks[i-1]["text"])
    st.write(embeddings[i-1])

    # NEW: chat history lives in session_state; reset it when a new PDF is uploaded
    file_id = hashlib.md5(uploaded.getvalue()).hexdigest()  # NEW
    if st.session_state.get("file_id") != file_id:  # NEW
        st.session_state.file_id = file_id  # NEW
        st.session_state.messages = []  # NEW

    # NEW: show the previous conversation
    for m in st.session_state.messages:  # NEW
        with st.chat_message(m["role"]):  # NEW
            if m.get("tools"):  # NEW
                st.caption("Tools used: " + ", ".join(m["tools"]))  # NEW
            st.write(m["content"])  # NEW

    question = st.chat_input("Ask something about your PDF")  # CHANGED (was st.text_input)
    if question:
        with st.chat_message("user"):  # NEW
            st.write(question)  # NEW

        with st.chat_message("assistant"):  # NEW
            # 1. Dense Vector Search (ChromaDB)
            q_emb = [float(x) for x in get_embedder()([question])[0]]
            res = collection.query(query_embeddings=[q_emb], n_results=10)
            vector_docs = res["documents"][0]
            vector_metas = res["metadatas"][0]

            # 2. Sparse BM25 Keyword Search
            bm25 = st.session_state.get("bm25")
            bm25_indices = bm25.search(question, top_k=10) if bm25 else []

            # 3. Combine unique candidates from both search methods (Hybrid Pool)
            candidate_map = {}
            for d, m in zip(vector_docs, vector_metas):
                candidate_map[d] = m["page"]
            for idx in bm25_indices:
                chunk = chunks[idx]
                candidate_map[chunk["text"]] = chunk["page"]

            candidate_docs = list(candidate_map.keys())
            candidate_metas = [{"page": candidate_map[d]} for d in candidate_docs]

            # 4. Cross-Encoder Reranking
            scores = list(get_reranker().rerank(question, candidate_docs))
            ranked = sorted(zip(scores, candidate_docs, candidate_metas), key=lambda x: x[0], reverse=True)[:3]
            with st.expander("Context (retrieved information)"):  # CHANGED (was a subheader)
                for score, doc, meta in ranked:
                    p = meta['page']
                    st.write(f"**Page {p}** (score {score:.2f})")
                    if "page_images" in st.session_state and p in st.session_state["page_images"]:
                        st.image(st.session_state["page_images"][p], caption=f"Slide {p}", width=400)
                    st.text(doc)

            context = "\n\n".join(f"[Page {m['page']}]\n{d}" for _, d, m in ranked)

            system_prompt = (
                 "You are a study assistant. Answer using the document context below. "
                "The context comes from slides, so the text order may look messy; reorganize it clearly. "
                "Cite the page number like (Page 9) after each fact from the document. "
                "If the answer is NOT in the context, call the search_web tool immediately without outputting any preliminary text. Only after you receive the tool results, present the final answer and note that it comes from the web. "
                "Never announce that you will use a tool; invoke it directly. "
                "If the user asks for flashcards or a quiz, call the generate_flashcards tool. "
                "Do not add facts that are not in the context."
               "Read ALL context pages before answering. If one page is an overview listing the main categories and another page only details one of them, answer from the overview page. "
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

            # NEW: keep only the last 6 past messages (trims history to fit the context window)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]  # NEW

            messages = [{"role": "system", "content": system_prompt}] + history + [  # CHANGED
                {"role": "user", "content": user_prompt},
            ]
            used_tools = []
            with st.spinner("Thinking..."):
                for step in range(3):
                    try:
                        reply = get_llm().chat.completions.create(
                            model=MODEL, 
                            messages=messages, 
                            tools=TOOLS, 
                            tool_choice="auto",
                            temperature=0.1
                        )
                        msg = reply.choices[0].message
                    except Exception:
                        # Fallback: if Groq has issues with tool schemas, query directly in plain text
                        clean_system = "You are a study assistant. Answer the user's question directly in plain text using the available context. Do not use or output any tool calls."
                        clean = [{"role": "system", "content": clean_system}] + [
                            m for m in messages[1:]
                            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) not in ("tool", "system")
                            and not (m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None))
                        ]
                        try:
                            reply = get_llm().chat.completions.create(model=MODEL, messages=clean, temperature=0.1)
                            msg = reply.choices[0].message
                        except Exception:
                            reply = get_llm().chat.completions.create(model="llama-3.3-70b-versatile", messages=clean, temperature=0.1)
                            msg = reply.choices[0].message
                        break

                    if not msg.tool_calls:
                        break

                    messages.append(msg)
                    for call in msg.tool_calls:
                        args = json.loads(call.function.arguments or "{}")
                        if call.function.name == "search_web":
                            result = search_web(args.get("query", question))
                        else:
                            result = generate_flashcards(args.get("topic", question), args.get("n", 5), context)
                        used_tools.append(call.function.name)
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

                    # If step 1 just finished a tool, instruct the model to provide the final written answer
                    if step == 1:
                        messages.append({"role": "user", "content": "Based on the tool results above, provide the final answer to my question in plain text now."})

            st.subheader("Answer")
            if used_tools:  # NEW
                st.caption("Tools used: " + ", ".join(used_tools))  # NEW
            
            final_content = msg.content or "No text response generated. Please see the retrieved context."
            st.write(final_content)

        # NEW: save this turn to the history
        st.session_state.messages.append({"role": "user", "content": question})  # NEW
        st.session_state.messages.append({"role": "assistant", "content": final_content, "tools": used_tools})  # NEW





