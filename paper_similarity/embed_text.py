from typing import Optional
import os

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai


EMBEDDING_MODEL_NAME = "text-embedding-004"
PROJECT_NAME = os.environ.get("PROJECT_NAME", "paper-vector-search")
LOCATION = os.environ.get("LOCATION", "us-central1")
vertexai.init(project=PROJECT_NAME, location=LOCATION)


def embed_text(
    text: str,
    title: Optional[str] = None,
    task: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    model = TextEmbeddingModel.from_pretrained(model_name=EMBEDDING_MODEL_NAME)
    embedding_input = TextEmbeddingInput(text, task_type=task, title=title)
    embeddings = model.get_embeddings(texts=[embedding_input])
    if embeddings:
        return embeddings[0].values
    else:
        raise RuntimeError("Failed to obtain embeddings")


if __name__ == "__main__":
    test_text = "This is an example text"
    embedding_vector = embed_text(test_text)
    print(embedding_vector)
    print(len(embedding_vector))
