import json
import textwrap
from pathlib import Path
import shutil

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "raw_docs"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "study_notes"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def load_documents(data_dir):
    records = []
    failed = []

    for path in sorted(data_dir.glob("**/*")):

        if path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(path))
                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

                if len(text.strip()) < 20:
                    failed.append(
                        (path.name, "no extractable text")
                    )
                    continue

                records.append({
                    "source": path.name,
                    "pages": len(reader.pages),
                    "text": text
                })

            except Exception as e:
                failed.append((path.name, str(e)))

        elif path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(errors="ignore")

            records.append({
                "source": path.name,
                "pages": 1,
                "text": text
            })

    return records, failed


print("=" * 60)
print("LOADING DOCUMENTS")
print("=" * 60)

documents, failed_files = load_documents(DATA_DIR)

print("Documents:", len(documents))
print("Pages:", sum(d["pages"] for d in documents))
print("Failed:", len(failed_files))

for name, reason in failed_files:
    print(" -", name, ":", reason)


def chunk_text(text, chunk_size=500, overlap=75):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


print()
print("=" * 60)
print("CREATING CHUNKS")
print("=" * 60)

all_chunks = []
counter = 0

for doc in documents:

    for chunk in chunk_text(doc["text"]):

        all_chunks.append({
            "id": f"chunk_{counter}",
            "text": chunk,
            "source": doc["source"]
        })

        counter += 1


print("Total chunks:", len(all_chunks))


# Backup old vector store
if VECTOR_STORE_DIR.exists():

    backup_dir = BASE_DIR / "data" / "vector_store_old"

    if backup_dir.exists():
        print("Removing previous temporary backup...")
        shutil.rmtree(backup_dir)

    print()
    print("=" * 60)
    print("BACKING UP OLD VECTOR STORE")
    print("=" * 60)

    shutil.move(
        str(VECTOR_STORE_DIR),
        str(backup_dir)
    )

    print("Backup:", backup_dir)


VECTOR_STORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)


print()
print("=" * 60)
print("CREATING CHROMADB")
print("=" * 60)

client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_DIR)
)

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)


print()
print("=" * 60)
print("ADDING DOCUMENT CHUNKS")
print("=" * 60)

if all_chunks:

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[
            {"source": c["source"]}
            for c in all_chunks
        ]
    )


count = collection.count()

print("Collection:", COLLECTION_NAME)
print("Chunks stored:", count)


config = {
    "embedding_model_name": EMBEDDING_MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "collection_name": COLLECTION_NAME,
    "ollama_model": "llama3.2"
}


with open(
    VECTOR_STORE_DIR / "pipeline_config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )


print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print("Vector store:", VECTOR_STORE_DIR)
print("Collection:", COLLECTION_NAME)
print("Count:", count)

print()
print("Backup kept at:")
print(BASE_DIR / "data" / "vector_store_old")