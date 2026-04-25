import os
import json
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Lazy imports so the server starts fast even before models are loaded
_model = None
_faiss = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_lib
        _faiss = _faiss_lib
    return _faiss


# ---------------------------------------------------------------------------
# Policy Analyzer  (Member 1 — RAG Implementation)
# ---------------------------------------------------------------------------

class PolicyAnalyzer:
    """
    Retrieves relevant compliance policy content for a user query using RAG.

    Pipeline:
        1. Load policy .txt files from the policies/ directory
        2. Chunk documents into overlapping passages
        3. Embed chunks with sentence-transformers (all-MiniLM-L6-v2)
        4. Build a FAISS flat-L2 index for fast similarity search
        5. retrieve(query) -> list of { policy, context } dicts
    """

    CHUNK_SIZE     = 300   # characters per chunk
    CHUNK_OVERLAP  = 60    # overlap between consecutive chunks

    def __init__(self, policies_dir: str = None):
        if policies_dir is None:
            # Resolve relative to this file → project_root/policies/
            here = Path(__file__).resolve().parent          # app/modules/
            policies_dir = str(here.parent.parent / "policies")

        self.policies_dir = Path(policies_dir)
        self.chunks: List[str]  = []   # raw text of every chunk
        self.metadata: List[Dict] = [] # { policy_name, filename } per chunk

        self._index = None  # FAISS index (built in _build_index)

        self._load_and_index()

    # ------------------------------------------------------------------
    # 1. Document loading
    # ------------------------------------------------------------------
    def _load_documents(self) -> List[Dict[str, str]]:
        """Read all .txt policy files from policies_dir."""
        docs = []
        if not self.policies_dir.exists():
            print(f"[PolicyAnalyzer] WARNING: policies directory not found at {self.policies_dir}")
            return docs

        for txt_file in sorted(self.policies_dir.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            policy_name = txt_file.stem.replace("_", " ").title()
            docs.append({"policy_name": policy_name, "filename": txt_file.name, "content": content})
            print(f"[PolicyAnalyzer] Loaded: {txt_file.name} ({len(content)} chars)")

        return docs

    # ------------------------------------------------------------------
    # 2. Chunking
    # ------------------------------------------------------------------
    def _chunk_document(self, content: str) -> List[str]:
        """Split content into overlapping character-level chunks."""
        chunks = []
        start = 0
        while start < len(content):
            end = start + self.CHUNK_SIZE
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------
    # 3 + 4. Embedding & FAISS index
    # ------------------------------------------------------------------
    def _build_index(self, embeddings: np.ndarray):
        """Build a FAISS flat L2 index from chunk embeddings."""
        faiss = _get_faiss()
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        return index

    def _load_and_index(self):
        """Full pipeline: load → chunk → embed → index."""
        docs = self._load_documents()
        if not docs:
            print("[PolicyAnalyzer] No policy documents found. Retrieval will return empty results.")
            return

        model = _get_model()

        for doc in docs:
            doc_chunks = self._chunk_document(doc["content"])
            for chunk in doc_chunks:
                self.chunks.append(chunk)
                self.metadata.append({
                    "policy_name": doc["policy_name"],
                    "filename":    doc["filename"],
                })

        print(f"[PolicyAnalyzer] Embedding {len(self.chunks)} chunks...")
        embeddings = model.encode(self.chunks, convert_to_numpy=True, show_progress_bar=False)
        self._index = self._build_index(embeddings)
        print(f"[PolicyAnalyzer] FAISS index built. {len(self.chunks)} chunks indexed.")

    # ------------------------------------------------------------------
    # 5. Retrieval (public API)
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant policy chunks for a given query.

        Returns:
            List of dicts with keys:
                - policy  : human-readable policy name
                - context : relevant text chunk
                - score   : L2 distance (lower = more relevant)
        """
        if self._index is None or len(self.chunks) == 0:
            return [{"policy": "No Policy Loaded", "context": "No policy documents are available.", "score": 0.0}]

        model = _get_model()
        query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

        k = min(top_k, len(self.chunks))
        distances, indices = self._index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "policy":  self.metadata[idx]["policy_name"],
                "context": self.chunks[idx],
                "score":   float(round(dist, 4)),
            })

        return results

    def retrieve_as_text(self, query: str, top_k: int = 3) -> str:
        """
        Convenience method returning a single concatenated string of top-k
        results — useful for passing as policy_context to the Decision Agent.
        """
        results = self.retrieve(query, top_k=top_k)
        parts = [f"[{r['policy']}]:\n{r['context']}" for r in results]
        return "\n\n---\n\n".join(parts)
