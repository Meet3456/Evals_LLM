import glob
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# import torch

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # rag_eval_deepeval/
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

# Intialize the embedding model


class QwenEmbeddings(Embeddings):
    """Adapter: makes a SentenceTransformer usable by LangChain/Chroma."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        # self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # documents: no instruction prompt
        return self.model.encode(
            texts,
            # batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        # queries: Qwen3 expects the "query" instruction prompt
        return self.model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
        ).tolist()


_st_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    # model_kwargs=torch.float32,
    processor_kwargs={
        "padding_side": "left"
    },  # renamed from tokenizer_kwargs, per your warning
)

embedding_model = QwenEmbeddings(_st_model)


# 1. LOAD the data - read each transcript, throw away the VTT timestamps
def load_transcripts():
    docs = []

    for path in glob.glob(str(DATA_DIR / "*.vtt")):
        lines = []
        with open(path) as file:
            for line in file:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)
        text = " ".join(lines)
        # print("Intial text : ",text[:100])

        session_taken = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session_taken}))

        print("First doc : ", docs[0])

    return docs


# 2. BUILD the vectore-store chunk, embed once, and keep it on disk so we don't re-embed:
def load_and_build_store():

    # if the vectore-store is already created just return it
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embedding_model)

    # else - create the vectore-store
    docs = load_transcripts()

    if not docs:
        raise FileNotFoundError(
            f"No .vtt files found in {DATA_DIR}. Check the path and file extension."
        )

    # create chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(documents=docs)

    print("Len of chunks created : ", len(chunks))

    return Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=DB_DIR
    )


def build_retriever():
    # return the vectore-store created in load_And_build_store function as retriever
    return load_and_build_store().as_retriever(search_kwargs={"k": 5})


# 3 - Testing the retriever:
if __name__ == "__main__":
    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")
