from openmockllm.tei.schemas.core import TeiBaseModel


class EmbeddingModel(TeiBaseModel):
    """Embedding model information"""

    pooling: str


class ClassifierModel(TeiBaseModel):
    """Classifier/Reranker model information"""

    id2label: dict[str, str]
    label2id: dict[str, int]


class ModelTypeEmbedding(TeiBaseModel):
    """Model type for embedding models"""

    embedding: EmbeddingModel


class ModelTypeClassifier(TeiBaseModel):
    """Model type for classifier models"""

    classifier: ClassifierModel


class ModelTypeReranker(TeiBaseModel):
    """Model type for reranker models"""

    reranker: ClassifierModel


# ModelType is a union of different model types
ModelType = ModelTypeEmbedding | ModelTypeClassifier | ModelTypeReranker


class Info(TeiBaseModel):
    """TEI model information"""

    # Model info
    model_id: str
    model_sha: str | None = None
    model_dtype: str
    model_type: ModelType

    # Router parameters
    max_concurrent_requests: int
    max_input_length: int
    max_batch_tokens: int
    max_client_batch_size: int
    max_batch_requests: int | None = None
    auto_truncate: bool
    tokenization_workers: int

    # Router info
    version: str
    sha: str | None = None
    docker_label: str | None = None
