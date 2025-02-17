#!/usr/bin/env python3
import os
import shutil
import re
import glob
import argparse
import logging
import pandas as pd
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_station_name(name: str) -> str:
    return re.sub(r"\W+", "", name.lower())
    

def load_markdown_documents(directory: str) -> list[Document]:
    """Load raw .md files from disk and apply MarkdownHeaderTextSplitter to keep headings in body text."""
    docs = []
    for file_path in glob.glob(os.path.join(directory, "**/*.md"), recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "header_1"), ("##", "header_2")])
        splitted = header_splitter.split_text(raw_text)
        structured_chunks = []
        for chunk in splitted:
            heading_prefix = [chunk.metadata[key] for key in ["header_1", "header_2"] if key in chunk.metadata]
            chunk_text = f"{'\n'.join(heading_prefix)}\n{chunk.page_content}"
            structured_chunks.append(Document(page_content=chunk_text, metadata={"source": os.path.basename(file_path)}))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for doc in structured_chunks:
            docs.extend(text_splitter.split_documents([doc]))
    logger.info("Loaded %d Markdown document chunks", len(docs))
    return docs

def load_csv_documents(directory: str) -> list[Document]:
    """
    Load CSV files row by row, preserving the original row text and
    adding a separate 'norm_station_name' field if 'station_name' is present.
    """
    documents = []
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        for idx, row in df.iterrows():
            row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            if "station_name" in df.columns:
                norm = normalize_station_name(str(row["station_name"]))
                row_text += f"\nnorm_station_name: {norm}"
            documents.append(Document(page_content=row_text, metadata={"source": os.path.basename(csv_file), "row": idx}))
    logger.info("Loaded %d CSV documents from %d files", len(documents), len(csv_files))
    return documents

def load_all_documents(directory: str) -> list[Document]:
    """Combine Markdown and CSV documents into a single list."""
    docs = load_markdown_documents(directory) + load_csv_documents(directory)
    logger.info("Total documents loaded: %d", len(docs))
    return docs

def create_and_save_vector_store(documents: list[Document], vector_store_dir: str, embedding_model: str, source_type: str) -> None:
    """
    Creates a FAISS vector store from the provided documents using Hugging Face embeddings,
    saves the index locally, and prints a few sample documents for evaluation.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(documents, embeddings)
    store_path = os.path.join(vector_store_dir, source_type)

    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.makedirs(store_path, exist_ok=True)
    vector_store.save_local(store_path)
    logger.info(f"Saved {source_type} vector store with {len(documents)} documents to '{store_path}'")


def main():
    parser = argparse.ArgumentParser(description="Generate and save FAISS vector store from Markdown and CSV files.")
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_DIR", "data"),
                        help="Directory containing Markdown and CSV files")
    parser.add_argument("--vector_store_dir", type=str, default=os.getenv("VECTOR_STORE_DIR", "vector_store"),
                        help="Directory to save the vector store")
    parser.add_argument("--embedding_model", type=str, default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/distiluse-base-multilingual-cased-v1"),
                        help="Hugging Face embedding model")
    args = parser.parse_args()

    md_docs = load_markdown_documents(args.data_dir)
    csv_docs = load_csv_documents(args.data_dir)
    # Create separate vector stores
    create_and_save_vector_store(md_docs, args.vector_store_dir, args.embedding_model, "subscriptions")
    create_and_save_vector_store(csv_docs, args.vector_store_dir, args.embedding_model, "stations")

if __name__ == "__main__":
    main()
