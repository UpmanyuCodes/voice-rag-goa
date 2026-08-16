"""Dataset Loader for MSMARCO-XI.

Provides a unified interface to load grounded corpus passages from either
offline curated datasets or online Hugging Face streaming.
"""

from typing import List, Dict, Any, Optional
from src.data.sample_data import SAMPLE_RECORDS


class DatasetLoader:
    """Loads MSMARCO-XI records for RAG indexing and retrieval."""

    def __init__(self, use_offline_samples: bool = True):
        self.use_offline_samples = use_offline_samples

    def load_corpus(
        self,
        language: str = "hi",
        limit: int = 50,
        split: str = "validation"
    ) -> List[Dict[str, Any]]:
        """Loads a list of MSMARCO-XI records.
        
        Args:
            language: Language code (e.g. 'hi', 'bn', 'ta', 'en')
            limit: Maximum records to load
            split: Dataset split ('validation' or 'train')
            
        Returns:
            List of dictionary records containing query, passages, answers, and metadata.
        """
        if self.use_offline_samples:
            # Filter samples matching language or return all if language is 'all'
            if language == "all":
                return SAMPLE_RECORDS[:limit]
            
            filtered = [r for r in SAMPLE_RECORDS if r.get("language") == language]
            if not filtered:
                # Fallback to all samples
                return SAMPLE_RECORDS[:limit]
            return filtered[:limit]

        # Online Hugging Face Streaming fallback
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "ai4bharat/MSMARCO-XI",
                data_files={split: f"{split}/{language}{'val' if split=='validation' else 'train'}.parquet"},
                split=split,
                streaming=True
            )
            records = []
            for item in ds.take(limit):
                records.append(item)
            return records
        except Exception as e:
            # Graceful fallback to offline curated records
            print(f"[DatasetLoader] Warning: HF streaming failed ({e}), falling back to offline samples.")
            return SAMPLE_RECORDS[:limit]
