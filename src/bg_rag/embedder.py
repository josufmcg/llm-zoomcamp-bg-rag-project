"""ONNX-based text embedding model wrapper.

Uses the Xenova/all-MiniLM-L6-v2 ONNX model for encoding text
into 384-dimensional normalized vectors. Performs mean pooling
over token embeddings with attention mask, followed by L2 normalization.

Usage:
    embedder = Embedder("models/Xenova/all-MiniLM-L6-v2")
    vector = embedder.encode("What is a Fighter?")  # shape: (384,)
    vectors = embedder.encode_batch(["text1", "text2"])  # shape: (2, 384)
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer


class Embedder:
    """Text embedding model using ONNX Runtime.

    Loads a tokenizer and ONNX model from a local directory.
    Produces 384-dimensional L2-normalized embeddings.

    Args:
        path: Path to the directory containing tokenizer.json and model.onnx.
    """

    def __init__(self, path: str | Path = "models/Xenova/all-MiniLM-L6-v2") -> None:
        path = Path(path)
        if not (path / "tokenizer.json").exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {path}. "
                "Run 'uv run python scripts/download_model.py' first."
            )
        if not (path / "model.onnx").exists():
            raise FileNotFoundError(
                f"model.onnx not found in {path}. "
                "Run 'uv run python scripts/download_model.py' first."
            )

        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string into a vector.

        Args:
            text: The text to encode.
            normalize: If True, L2-normalize the output vector.

        Returns:
            A 1-D numpy array of shape (384,).
        """
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode multiple texts into vectors.

        Args:
            texts: List of text strings to encode.
            normalize: If True, L2-normalize each output vector.

        Returns:
            A 2-D numpy array of shape (len(texts), 384).
        """
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)

        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array(
                [e.ids for e in encoded], dtype=np.int64
            )
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        # Run ONNX model inference
        hidden = self.session.run(None, feed)[0]  # shape: (batch, seq_len, 384)

        # Mean pooling: average over non-padding tokens
        mask = feed["attention_mask"][..., None]  # shape: (batch, seq_len, 1)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)  # shape: (batch, 384)

        # L2 normalization
        if normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / norms

        return pooled
