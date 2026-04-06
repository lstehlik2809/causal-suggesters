from typing import List, Optional
from .model_suggester import ModelSuggester, RelationshipsResponse, _SYSTEM
from .enums import RelationshipStrategy


class AugmentedModelSuggester(ModelSuggester):
    """
    RAG-enhanced ModelSuggester. Retrieves relevant causal knowledge
    from a vector store (default: CauseNet) before prompting the LLM.

    The embedding model is fully local (sentence-transformers) — no OpenAI
    embedding API dependency, unlike pywhyllm's original implementation.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "causenet",
        similarity_threshold: float = 0.7,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self._setup_retriever(embedding_model, collection_name)

    def _setup_retriever(self, embedding_model: str, collection_name: str):
        from sentence_transformers import SentenceTransformer
        import chromadb

        self.embedder = SentenceTransformer(embedding_model)
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def load_knowledge_base(self, entries: List[dict]):
        """
        Populate the vector store.
        Each entry: {"id": str, "text": str, "metadata": dict (optional)}
        """
        texts = [e["text"] for e in entries]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(
            ids=[e["id"] for e in entries],
            embeddings=embeddings,
            documents=texts,
            metadatas=[e.get("metadata", {}) for e in entries],
        )

    def _retrieve_context(self, query: str) -> Optional[str]:
        if self.collection.count() == 0:
            return None

        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return None

        # With cosine space, distance in [0, 2]; similarity = 1 - distance
        docs = [
            doc
            for doc, dist in zip(
                results["documents"][0], results["distances"][0]
            )
            if (1 - dist) >= self.similarity_threshold
        ]
        return "\n".join(docs) if docs else None

    def suggest_relationships(
        self,
        all_factors: List[str],
        domain_expertises: List[str],
        strategy: RelationshipStrategy = RelationshipStrategy.Pairwise,
    ) -> List[List[str]]:
        context = self._retrieve_context(" ".join(all_factors))

        rag_block = ""
        if context:
            rag_block = (
                f"\n\n## Relevant causal knowledge from knowledge base:\n{context}\n\n"
                "Use this knowledge to inform your suggestions, but do not blindly copy it. "
                "Apply causal reasoning to determine which relationships are relevant "
                "to the specific variables in this query."
            )

        prompt = f"""Variables: {all_factors}
Domain expertise context: {domain_expertises}
Strategy: {strategy.value}
{rag_block}
Suggest directed causal relationships (edges) between these variables.
Only include edges where there is a plausible direct causal mechanism.
Do NOT include spurious correlations or reverse causation.
Do NOT include variables that are effects of both treatment and outcome (colliders)."""

        result = self.llm.complete_structured(prompt, RelationshipsResponse, _SYSTEM)
        return result.edges
