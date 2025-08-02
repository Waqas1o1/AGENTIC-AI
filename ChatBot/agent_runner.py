import os
import numpy as np
from openai import OpenAI
from agents import Agent, Runner, function_tool, ModelSettings
import faiss
import tiktoken
import pickle

# ========== CONFIG ========== #
DOCUMENT_DIR = "S:\\wooglobe\\documents"
INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunks.pkl"
EMBED_MODEL = "text-embedding-3-small"
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY", ""
)
openai_client = OpenAI()


# ========== 1. Load Docs ========== #
def load_documents():
    files = [
        "terms_of_submition.txt",
        "terms_of_use.txt",
        "privacyPolicy.txt",
        "wooglobeApperanceRelease.txt",
        "faq.txt",
        "FAQS WooGlobe.txt",
        "Quactions & Answer.txt",
    ]
    texts = []
    for fn in files:
        path = os.path.join(DOCUMENT_DIR, fn)
        with open(path, encoding="utf-8") as f:
            texts.append(f"--- {fn} ---\n{f.read()}")
    return texts


# ========== 2. Chunk Text ========== #
def chunk_text(text, max_tokens=500):
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    toks = encoding.encode(text)
    chunks = []
    for i in range(0, len(toks), max_tokens):
        chunk = encoding.decode(toks[i : i + max_tokens])
        chunks.append(chunk)
    return chunks


# ========== 3. Build or Load FAISS Index ========== #
def build_or_load_index(rebuild=False):
    if not rebuild and os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("Loading FAISS index and chunks from disk...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    print("Building FAISS index from documents...")
    docs = load_documents()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d))

    embeddings = []
    for chunk in all_chunks:
        resp = openai_client.embeddings.create(input=chunk, model=EMBED_MODEL)
        embeddings.append(np.array(resp.data[0].embedding, dtype="float32"))

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack(embeddings))

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    return index, all_chunks


# ========== 4. Load Index ========== #
index, all_chunks = build_or_load_index(rebuild=False)


# ========== 5. Tool ========== #
@function_tool
def search_knowledge(query: str) -> str:
    resp = openai_client.embeddings.create(input=query, model=EMBED_MODEL)
    qvec = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    D, I = index.search(qvec, k=3)
    hits = [all_chunks[i] for i in I[0]]
    return "\n\n".join(hits) if hits else "I'm sorry, no relevant content found."


# ========== 6. Agent ========== #
agent = Agent(
    name="WooglobeAssistant",
    instructions=(
        "You are a support assistant for Wooglobe. "
        "You answer using the provided company documents: Terms of Use, Privacy Policy, Appearance Release, and FAQs. "
        "If the information is not found in these documents, say: 'I'm sorry, I couldn't find that information in the Wooglobe documentation.' "
        "Do NOT guess or provide general knowledge, math help, or external advice. "
        "You must use the 'search_knowledge' tool to answer all questions."
    ),
    tools=[search_knowledge],
    model_settings=ModelSettings(tool_choice="required"),
)


# ========== 7. Runner ========== #
async def main():
    runner = Runner()
    print("Agent ready. Ask anything; type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break
        res = await runner.run(agent, q)
        print("Bot:", res.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
