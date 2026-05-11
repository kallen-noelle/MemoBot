from pydantic import BaseModel, ConfigDict, Field


class VectorConfig(BaseModel):
    """Config section for a vector database"""

    use: str = Field(
        ...,
        description="Class path of the model provider(e.g. langchain_openai.ChatOpenAI)",
    )
    api_key: str | None = Field(default_factory=lambda: None, description="DashScope API key")

    embedding_model: str | None = Field(default_factory=lambda: None, description="Name of the embedding model")
    top_k_graph: int = Field(default_factory=lambda: None, description="Top-k results for graph")
    top_k_documents: int = Field(default_factory=lambda: None, description="Top-k results for documents")
    top_k_bm25: int = Field(default_factory=lambda: None, description="Top-k results for BM25")
    chunk_size: int = Field(default_factory=lambda: None, description="Size of each chunk")
    chunk_overlap: int = Field(default_factory=lambda: None, description="Overlap between adjacent chunks")
 