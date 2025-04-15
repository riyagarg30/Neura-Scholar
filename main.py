import asyncio
import logging
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Model libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Pipeline API with CRUD and Embedding", version="0.3")

# ---------------------------
# Model Configuration Section
# ---------------------------

DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Cache for embedding models.
embedding_model_cache = {}


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model. If already loaded, retrieve it from cache.
    The associated tokenizer is handled automatically.
    """
    if model_name in embedding_model_cache:
        logging.info("Model '%s' loaded from cache", model_name)
        return embedding_model_cache[model_name]
    else:
        logging.info("Loading embedding model: '%s'", model_name)
        model = SentenceTransformer(model_name)
        embedding_model_cache[model_name] = model
        return model


# Global summarizer model variable.
summarizer_model = None


def load_summarizer_model():
    """
    Load a summarization pipeline using a default summarizer model
    (facebook/bart-large-cnn).
    """
    global summarizer_model
    if summarizer_model is None:
        logging.info("Loading summarizer model")
        summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer_model


async def compute_embedding(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Compute and return the embedding for a given text using the specified or default model.
    The computation is executed asynchronously.
    """
    actual_model_name = model_name or DEFAULT_EMBEDDING_MODEL
    model = load_embedding_model(actual_model_name)
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
    return embedding


# ---------------------------
# VectorDB Abstraction Section
# ---------------------------


class VectorDB:
    """
    A simple in-memory abstraction for a vector database.
    Replace with an actual vector store (e.g., pgvector) for production use.
    """

    def __init__(self):
        # Pre-populated dummy documents.
        self.documents = [
            {
                "id": "1",
                "text": "Research paper 1 on topic A",
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "id": "2",
                "text": "Research paper 2 on topic B",
                "embedding": [0.2, 0.1, 0.4],
            },
            {
                "id": "3",
                "text": "Research paper 3 on topic A",
                "embedding": [0.15, 0.25, 0.35],
            },
        ]

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        """
        Simulate a vector similarity search.
        In a real system, you would compute a cosine similarity with the document embeddings.
        """
        logging.info("Performing vector search")

        # Dummy similarity measure: absolute difference of the sum of embedding values.
        def similarity(doc):
            return abs(sum(query_embedding) - sum(doc["embedding"]))

        sorted_docs = sorted(self.documents, key=similarity)
        return sorted_docs[:top_k]

    async def add_document(self, doc: dict) -> dict:
        self.documents.append(doc)
        logging.info("Document added with id: %s", doc["id"])
        return doc

    async def get_document(self, doc_id: str) -> Optional[dict]:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None

    async def update_document(
        self, doc_id: str, updated_fields: dict
    ) -> Optional[dict]:
        for doc in self.documents:
            if doc["id"] == doc_id:
                doc.update(updated_fields)
                logging.info("Document updated with id: %s", doc_id)
                return doc
        return None

    async def delete_document(self, doc_id: str) -> bool:
        for idx, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                self.documents.pop(idx)
                logging.info("Document deleted with id: %s", doc_id)
                return True
        return False

    async def list_documents(self) -> List[dict]:
        return self.documents


# Instantiate a global vector database.
vector_db = VectorDB()

# ---------------------------
# Pydantic Models Section
# ---------------------------


class SearchRequest(BaseModel):
    query: str
    model_name: Optional[str] = None  # Dynamic selection for query embedding.
    top_k: Optional[int] = 5


class SearchResponse(BaseModel):
    query: str
    embedding: List[float]
    retrieved_documents: List[dict]


class SummarizeRequest(BaseModel):
    doc_id: Optional[str] = None
    text: Optional[str] = None


class SummarizeResponse(BaseModel):
    summary: str


# Models for CRUD operations on documents.
class DocumentBase(BaseModel):
    text: str
    embedding: Optional[List[float]] = None


class DocumentCreate(DocumentBase):
    id: Optional[str] = None  # If not provided, an ID will be generated.
    model_name: Optional[str] = (
        None  # Optional model selection for computing embedding.
    )


class DocumentUpdate(BaseModel):
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    model_name: Optional[str] = None  # For recomputing embedding if text is updated.


class Document(DocumentBase):
    id: str


# ---------------------------
# Endpoints Section
# ---------------------------


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Complete RAG pipeline endpoint:
    - Receives a query as text.
    - Generates its embedding using a SentenceTransformer.
    - Performs a vector lookup against an abstracted vector DB.
    """
    # Compute embedding for the query text.
    query_embedding = await compute_embedding(request.query, request.model_name)
    # Perform asynchronous vector search.
    retrieved_docs = await vector_db.search(
        query_embedding=query_embedding, top_k=request.top_k
    )
    return SearchResponse(
        query=request.query,
        embedding=query_embedding,
        retrieved_documents=retrieved_docs,
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(request: SummarizeRequest):
    """
    Summarization endpoint:
    - Accepts either a document ID (to fetch stored text) or raw text.
    - Generates and returns a summary.
    """
    if request.doc_id:
        doc = await vector_db.get_document(request.doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        text_to_summarize = doc["text"]
    elif request.text:
        text_to_summarize = request.text
    else:
        raise HTTPException(
            status_code=400, detail="Either 'doc_id' or 'text' must be provided"
        )

    summarizer = load_summarizer_model()
    loop = asyncio.get_event_loop()
    summary_result = await loop.run_in_executor(
        None, lambda: summarizer(text_to_summarize, truncation=True)
    )
    summary_text = summary_result[0]["summary_text"]
    return SummarizeResponse(summary=summary_text)


# --- CRUD Endpoints for Documents ---


@app.post("/documents", response_model=Document)
async def create_document(doc: DocumentCreate):
    """
    Create a new document. If an embedding is not provided,
    it is computed from the provided text using the specified model or the default.
    """
    doc_id = doc.id if doc.id is not None else str(uuid.uuid4())
    # Compute embedding if not provided.
    if not doc.embedding:
        computed_embedding = await compute_embedding(doc.text, doc.model_name)
    else:
        computed_embedding = doc.embedding

    new_doc = {"id": doc_id, "text": doc.text, "embedding": computed_embedding}
    created = await vector_db.add_document(new_doc)
    return created


@app.get("/documents", response_model=List[Document])
async def list_documents():
    """
    Retrieve a list of all documents in the vector DB.
    """
    docs = await vector_db.list_documents()
    return docs


@app.get("/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    """
    Retrieve a specific document by its ID.
    """
    doc = await vector_db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.put("/documents/{doc_id}", response_model=Document)
async def update_document(doc_id: str, doc_update: DocumentUpdate):
    """
    Update a document with new values for 'text' and/or 'embedding'.
    If the text is updated and a new embedding is not explicitly provided,
    the embedding is recomputed using the updated text.
    """
    update_data = doc_update.dict(exclude_unset=True)
    # If text was updated and no embedding provided, recompute the embedding.
    if "text" in update_data and update_data.get("embedding") is None:
        new_embedding = await compute_embedding(
            update_data["text"], doc_update.model_name
        )
        update_data["embedding"] = new_embedding

    updated = await vector_db.update_document(doc_id, update_data)
    if updated is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return updated


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document from the vector DB by its ID.
    """
    success = await vector_db.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": "Document deleted successfully"}


# ---------------------------
# Main Entrypoint
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
