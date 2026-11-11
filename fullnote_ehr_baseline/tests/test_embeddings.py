from __future__ import annotations

import numpy as np

from fullnote_ehr.embeddings import (
    encode_note_chunks,
    finalize_token_sums,
    generate_embedding_artifact,
    load_embedding_artifact,
)
from helpers import dataset_row, write_two_parts


class FakeTokenizer:
    pad_token_id = 0
    all_special_ids = [101, 102]
    vocab_size = 200

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == "literal-special-tokens":
            return [101, 7, 102]
        return list(range(1, int(text)))

    def build_inputs_with_special_tokens(self, token_ids):
        return [101, *token_ids, 102]

class FakeEncoder:
    hidden_size = 3

    def encode_notes(self, texts):
        vectors = []
        tokens = []
        chunks = []
        for text in texts:
            length = len(text.split())
            vector = np.array([length, length + 1, length + 2], dtype=np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
            tokens.append(length)
            chunks.append(1)
        return (
            np.vstack(vectors),
            np.asarray(tokens, dtype=np.int64),
            np.asarray(chunks, dtype=np.int32),
        )

    def manifest(self):
        return {
            "backend": "fake-test",
            "hidden_size": self.hidden_size,
            "max_length": 8,
            "pooling": "deterministic test vectors",
            "l2_normalize": True,
        }


def test_non_overlapping_chunking_and_content_masks():
    chunks = encode_note_chunks(FakeTokenizer(), "13", max_length=6)
    assert [chunk.content_tokens for chunk in chunks] == [4, 4, 4]
    assert chunks[0].input_ids == (101, 1, 2, 3, 4, 102)
    assert chunks[0].content_mask == (0, 1, 1, 1, 1, 0)


def test_literal_special_token_ids_remain_note_content():
    chunks = encode_note_chunks(
        FakeTokenizer(),
        "literal-special-tokens",
        max_length=6,
    )
    assert chunks[0].input_ids == (101, 101, 7, 102, 102)
    assert chunks[0].content_mask == (0, 1, 1, 1, 0)
    assert chunks[0].content_tokens == 3


def test_weighted_mean_and_l2_normalization():
    sums = np.array([[3.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    result = finalize_token_sums(sums, np.array([1, 2]), l2_normalize=True)
    np.testing.assert_allclose(result[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(result[1], [1.0, 0.0], rtol=1e-6)


def test_failure_safe_artifact_generation_and_loading(tmp_path):
    data = tmp_path / "data"
    output = tmp_path / "embeddings"
    data.mkdir()
    rows = [
        dataset_row(1, 101, "train", 1),
        dataset_row(2, 102, "train", 0),
        dataset_row(3, 103, "validation", 0),
        dataset_row(4, 104, "test", 1),
    ]
    write_two_parts(data, rows)
    manifest = generate_embedding_artifact(
        data_dir=data,
        output_dir=output,
        encoder=FakeEncoder(),
        note_block_size=2,
    )
    embeddings, index, loaded = load_embedding_artifact(output)
    assert manifest["status"] == "COMPLETE"
    assert loaded["shape"] == [4, 3]
    assert embeddings.shape == (4, 3)
    assert index["note_id"].tolist() == ["note-1", "note-2", "note-3", "note-4"]
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(4))
    assert not (output / "embedding_progress.json").exists()
