"""
Vector Memory - Advanced pattern recognition and memory system for NEXUS

This module implements a sophisticated vector database system for:
- Long-term pattern storage and retrieval
- Semantic similarity search for market patterns
- Experience replay for continual learning
- Multi-modal embeddings (price, volume, sentiment)
- Hierarchical memory organization
- Real-time pattern matching and adaptation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json
import time
import uuid
import pickle
from concurrent.futures import ThreadPoolExecutor

# Vector databases
import faiss
import chromadb
try:
    import lancedb
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

from sentence_transformers import SentenceTransformer

# Embeddings and preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from nexus.utils.logger import get_nexus_logger, PerformanceLogger

# Set up loggers
logger = get_nexus_logger("nexus.core.memory")
perf_logger = PerformanceLogger("vector_memory")

class PatternEncoder(nn.Module):
    """Neural encoder for converting market patterns to embeddings."""

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        """
        Initialize pattern encoder.

        Args:
            input_dim: Dimensionality of input features
            embedding_dim: Dimensionality of output embeddings
        """
        super().__init__()

        # Define encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input pattern to embedding.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        return self.encoder(x)

class PatternSimilarity:
    """Pattern similarity calculator with various metrics."""

    def __init__(self, metric: str = "cosine"):
        """
        Initialize similarity calculator.

        Args:
            metric: Similarity metric ("cosine", "euclidean", "dot")
        """
        self.metric = metric

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate similarity between vectors.

        Args:
            x: First vector
            y: Second vector

        Returns:
            Similarity score (higher means more similar)
        """
        if self.metric == "cosine":
            # Cosine similarity
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)

            if norm_x == 0 or norm_y == 0:
                return 0.0

            return np.dot(x, y) / (norm_x * norm_y)

        elif self.metric == "euclidean":
            # Euclidean distance (converted to similarity)
            dist = np.linalg.norm(x - y)
            return 1.0 / (1.0 + dist)

        elif self.metric == "dot":
            # Dot product
            return np.dot(x, y)

        else:
            raise ValueError(f"Unknown similarity metric: {self.metric}")

    def batch_calculate(self, x: np.ndarray, queries: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between one vector and multiple vectors.

        Args:
            x: Reference vector [dim]
            queries: Query vectors [n, dim]

        Returns:
            Array of similarity scores [n]
        """
        if self.metric == "cosine":
            # Batch cosine similarity
            norm_x = np.linalg.norm(x)
            norm_queries = np.linalg.norm(queries, axis=1)

            # Avoid division by zero
            norm_queries = np.where(norm_queries == 0, 1e-10, norm_queries)

            return np.dot(queries, x) / (norm_queries * norm_x)

        elif self.metric == "euclidean":
            # Batch Euclidean distance (converted to similarity)
            dists = np.linalg.norm(queries - x, axis=1)
            return 1.0 / (1.0 + dists)

        elif self.metric == "dot":
            # Batch dot product
            return np.dot(queries, x)

        else:
            raise ValueError(f"Unknown similarity metric: {self.metric}")

class MarketPattern:
    """Market pattern container with metadata."""

    def __init__(self,
                 data: np.ndarray,
                 embedding: np.ndarray,
                 pattern_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize market pattern.

        Args:
            data: Raw pattern data
            embedding: Vector embedding of pattern
            pattern_id: Unique pattern identifier
            timestamp: Pattern timestamp
            metadata: Additional pattern metadata
        """
        self.data = data
        self.embedding = embedding
        self.pattern_id = pattern_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = None
        self.similarity_scores = {}  # Cache for similarity scores

    def access(self) -> None:
        """Record pattern access."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pattern to dictionary.

        Returns:
            Dictionary representation of pattern
        """
        return {
            "pattern_id": self.pattern_id,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any], raw_data: Optional[np.ndarray] = None) -> 'MarketPattern':
        """
        Create pattern from dictionary.

        Args:
            data_dict: Dictionary representation
            raw_data: Optional raw pattern data

        Returns:
            MarketPattern instance
        """
        pattern = cls(
            data=raw_data if raw_data is not None else np.array([]),
            embedding=np.array(data_dict["embedding"]),
            pattern_id=data_dict["pattern_id"],
            timestamp=datetime.fromisoformat(data_dict["timestamp"]),
            metadata=data_dict["metadata"]
        )

        pattern.access_count = data_dict.get("access_count", 0)
        if data_dict.get("last_accessed"):
            pattern.last_accessed = datetime.fromisoformat(data_dict["last_accessed"])

        return pattern

class VectorMemory:
    """
    Advanced vector memory system for market patterns.
    """
    def __init__(self, capacity: int = 100000, dimension: int = 128, config: Dict[str, Any] = None):
        """
        Initialize vector memory system.

        Args:
            capacity: Maximum number of patterns to store
            dimension: Dimensionality of embeddings
            config: Optional configuration dictionary
        """
        # Support both direct arguments and config dict/object
        if config is not None:
            if hasattr(config, 'get') and callable(config.get):
                self.embedding_dim = config.get("embedding_dim", dimension)
                self.max_patterns = config.get("max_patterns", capacity)
                self.similarity_threshold = config.get("similarity_threshold", 0.85)
                self.db_type = config.get("db_type", "faiss")
                self.storage_path = Path(config.get("storage_path", "./data/vector_memory"))
                self.use_compression = config.get("use_compression", False)
                use_gpu = config.get("use_gpu", True)
            else:
                self.embedding_dim = getattr(config, "embedding_dim", dimension)
                self.max_patterns = getattr(config, "max_patterns", capacity)
                self.similarity_threshold = getattr(config, "similarity_threshold", 0.85)
                self.db_type = getattr(config, "db_type", "faiss")
                self.storage_path = Path(getattr(config, "storage_path", "./data/vector_memory"))
                self.use_compression = getattr(config, "use_compression", False)
                use_gpu = getattr(config, "use_gpu", True)
        else:
            self.embedding_dim = dimension
            self.max_patterns = capacity
            self.similarity_threshold = 0.85
            self.db_type = "faiss"
            self.storage_path = Path("./data/vector_memory")
            self.use_compression = False
            use_gpu = True
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.pattern_encoder = None
        self.similarity_calculator = PatternSimilarity(metric="cosine")
        self.scaler = StandardScaler()
        self.patterns: Dict[str, MarketPattern] = {}
        self.vector_index = None
        self.metadata_index: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            "total_patterns": 0,
            "queries_processed": 0,
            "avg_query_time_ms": 0,
            "last_maintenance": datetime.now(),
            "index_size_bytes": 0,
            "embedding_dimension": self.embedding_dim
        }

        # Initialize components
        self._initialize_components()

        logger.info(f"Vector memory initialized with {self.embedding_dim}-dimensional embeddings using {self.db_type} backend")

    def _initialize_components(self) -> None:
        """Initialize memory components."""
        try:
            # Initialize vector index
            if self.db_type == "faiss":
                self._init_faiss()
            elif self.db_type == "chroma":
                self._init_chroma()
            elif self.db_type == "lance" and LANCE_AVAILABLE:
                self._init_lance()
            else:
                logger.warning(f"Unknown or unavailable DB type: {self.db_type}, falling back to FAISS")
                self.db_type = "faiss"
                self._init_faiss()

            # Try to load saved patterns
            self._load_patterns()

        except Exception as e:
            logger.error(f"Error initializing vector memory: {e}")
            # Fall back to in-memory storage
            self._init_fallback()

    def _init_faiss(self) -> None:
        """Initialize FAISS vector index."""
        try:
            # Create FAISS index
            if self.use_compression:
                # Use compressed index for large collections
                self.vector_index = faiss.IndexIVFPQ(
                    faiss.IndexFlatL2(self.embedding_dim),  # Base index
                    self.embedding_dim,                     # Dimension
                    min(4096, self.max_patterns // 10),     # Number of centroids
                    8,                                      # Number of sub-quantizers
                    8                                       # Bits per sub-quantizer
                )
                self.vector_index.train(np.zeros((100, self.embedding_dim), dtype=np.float32))
            else:
                # Use flat index for exact search
                self.vector_index = faiss.IndexFlatL2(self.embedding_dim)

            logger.info(f"FAISS index initialized with dimension {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self._init_fallback()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB for vector storage."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=str(self.storage_path))

            # Create or get collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="nexus_patterns",
                embedding_function=None,  # We provide our own embeddings
                metadata={"dimension": self.embedding_dim}
            )

            logger.info("ChromaDB collection initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.db_type = "faiss"
            self._init_faiss()

    def _init_lance(self) -> None:
        """Initialize LanceDB for vector storage."""
        try:
            # Create LanceDB database
            self.lance_db = lancedb.connect(str(self.storage_path / "lancedb"))

            # Create or get table
            if "nexus_patterns" in self.lance_db.table_names():
                self.lance_table = self.lance_db.open_table("nexus_patterns")
            else:
                # Create schema for the table
                import pyarrow as pa
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
                    pa.field("timestamp", pa.string()),
                    pa.field("metadata", pa.string())
                ])

                # Create empty table with the schema
                self.lance_table = self.lance_db.create_table("nexus_patterns", schema=schema)

            logger.info("LanceDB table initialized")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            self.db_type = "faiss"
            self._init_faiss()

    def _init_fallback(self) -> None:
        """Initialize fallback in-memory storage."""
        logger.warning("Using fallback in-memory vector storage")
        self.db_type = "memory"
        self.vector_index = None
        self.patterns = {}

    def _save_patterns(self) -> None:
        """Save patterns to disk."""
        try:
            # Save patterns metadata
            patterns_data = {
                pattern_id: pattern.to_dict()
                for pattern_id, pattern in self.patterns.items()
            }

            with open(self.storage_path / "patterns_metadata.json", "w") as f:
                json.dump(patterns_data, f)

            # Save raw pattern data
            raw_data = {
                pattern_id: pattern.data.tolist() if isinstance(pattern.data, np.ndarray) else pattern.data
                for pattern_id, pattern in self.patterns.items()
            }

            with open(self.storage_path / "patterns_raw_data.json", "w") as f:
                json.dump(raw_data, f)

            # Save statistics
            with open(self.storage_path / "memory_stats.json", "w") as f:
                json.dump(self.stats, f)

            # If using FAISS, save the index
            if self.db_type == "faiss" and self.vector_index is not None:
                faiss.write_index(self.vector_index, str(self.storage_path / "faiss_index.bin"))

            logger.info(f"Saved {len(self.patterns)} patterns to disk")

        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

    def _load_patterns(self) -> None:
        """Load patterns from disk."""
        try:
            # Load patterns metadata
            metadata_path = self.storage_path / "patterns_metadata.json"
            if not metadata_path.exists():
                logger.info("No saved patterns found")
                return

            with open(metadata_path, "r") as f:
                patterns_data = json.load(f)

            # Load raw pattern data
            raw_data_path = self.storage_path / "patterns_raw_data.json"
            raw_data = {}
            if raw_data_path.exists():
                with open(raw_data_path, "r") as f:
                    raw_data = json.load(f)

            # Load patterns
            self.patterns = {}
            for pattern_id, pattern_dict in patterns_data.items():
                pattern_raw_data = np.array(raw_data.get(pattern_id, []))
                self.patterns[pattern_id] = MarketPattern.from_dict(pattern_dict, pattern_raw_data)

            # Load statistics
            stats_path = self.storage_path / "memory_stats.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.stats = json.load(f)

            # If using FAISS, load the index
            if self.db_type == "faiss":
                index_path = self.storage_path / "faiss_index.bin"
                if index_path.exists():
                    self.vector_index = faiss.read_index(str(index_path))

            logger.info(f"Loaded {len(self.patterns)} patterns from disk")

        except Exception as e:
            logger.error(f"Error loading patterns: {e}")

    def _update_vector_index(self, pattern: MarketPattern) -> None:
        """
        Update vector index with new pattern.

        Args:
            pattern: Pattern to add to index
        """
        try:
            if self.db_type == "faiss" and self.vector_index is not None:
                # Add vector to FAISS index
                self.vector_index.add(np.array([pattern.embedding], dtype=np.float32))

            elif self.db_type == "chroma":
                # Add document to ChromaDB
                self.chroma_collection.add(
                    ids=[pattern.pattern_id],
                    embeddings=[pattern.embedding.tolist()],
                    metadatas=[pattern.metadata]
                )

            elif self.db_type == "lance" and LANCE_AVAILABLE:
                # Add vector to LanceDB
                self.lance_table.add([{
                    "id": pattern.pattern_id,
                    "vector": pattern.embedding.tolist(),
                    "timestamp": pattern.timestamp.isoformat(),
                    "metadata": json.dumps(pattern.metadata)
                }])

            # Update metadata index for filtering
            for key, value in pattern.metadata.items():
                if key not in self.metadata_index:
                    self.metadata_index[key] = {}

                if value not in self.metadata_index[key]:
                    self.metadata_index[key][value] = set()

                self.metadata_index[key][value].add(pattern.pattern_id)

        except Exception as e:
            logger.error(f"Error updating vector index: {e}")

    def add_pattern(self,
                    data: Union[np.ndarray, pd.DataFrame],
                    embedding: Optional[np.ndarray] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add pattern to memory.

        Args:
            data: Raw pattern data
            embedding: Pre-computed embedding (optional)
            metadata: Pattern metadata

        Returns:
            Pattern ID
        """
        with perf_logger.measure("add_pattern"):
            try:
                # Convert DataFrame to numpy if needed
                if isinstance(data, pd.DataFrame):
                    data = data.values

                # Generate embedding if not provided
                if embedding is None:
                    embedding = self.encode_pattern(data)

                # Create metadata if not provided
                if metadata is None:
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "data_shape": data.shape if hasattr(data, "shape") else None,
                        "norm": float(np.linalg.norm(embedding))
                    }

                # Create pattern
                pattern = MarketPattern(
                    data=data,
                    embedding=embedding,
                    metadata=metadata
                )

                # Add to storage
                self.patterns[pattern.pattern_id] = pattern

                # Update vector index
                self._update_vector_index(pattern)

                # Update statistics
                self.stats["total_patterns"] = len(self.patterns)

                # Periodic maintenance
                if len(self.patterns) % 100 == 0:
                    asyncio.create_task(self._maintenance())

                return pattern.pattern_id

            except Exception as e:
                logger.error(f"Error adding pattern: {e}")
                return ""

    def encode_pattern(self, data: np.ndarray) -> np.ndarray:
        """
        Encode pattern to vector embedding.

        Args:
            data: Raw pattern data

        Returns:
            Vector embedding
        """
        try:
            # Use neural encoder if available
            if self.pattern_encoder is not None:
                with torch.no_grad():
                    data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
                    embedding = self.pattern_encoder(data_tensor).cpu().numpy()
                return embedding

            # Fallback: use PCA or raw features
            if data.size > self.embedding_dim:
                # Flatten data if needed
                flat_data = data.flatten() if data.ndim > 1 else data

                # Apply PCA if data dimension is larger than embedding dimension
                if not hasattr(self, 'pca'):
                    self.pca = PCA(n_components=self.embedding_dim)
                    self.pca.fit(flat_data.reshape(1, -1))

                embedding = self.pca.transform(flat_data.reshape(1, -1))[0]

                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

                return embedding
            else:
                # Pad if needed
                if data.size < self.embedding_dim:
                    embedding = np.zeros(self.embedding_dim)
                    embedding[:data.size] = data.flatten()
                else:
                    embedding = data.flatten()[:self.embedding_dim]

                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

                return embedding

        except Exception as e:
            logger.error(f"Error encoding pattern: {e}")
            return np.zeros(self.embedding_dim)

    def search_similar_patterns(self,
                               query: Union[np.ndarray, str],
                               k: int = 5,
                               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar patterns.

        Args:
            query: Query pattern or ID
            k: Number of results
            metadata_filter: Filter by metadata

        Returns:
            List of similar patterns with similarity scores
        """
        with perf_logger.measure("search_similar"):
            start_time = time.time()

            try:
                # If query is a pattern ID, get the pattern
                if isinstance(query, str) and query in self.patterns:
                    query_embedding = self.patterns[query].embedding
                elif isinstance(query, np.ndarray):
                    # If raw data, encode it
                    if query.shape[-1] != self.embedding_dim:
                        query_embedding = self.encode_pattern(query)
                    else:
                        query_embedding = query
                else:
                    logger.error(f"Invalid query type: {type(query)}")
                    return []

                # Apply metadata filter if provided
                candidate_ids = None
                if metadata_filter:
                    candidate_ids = self._filter_by_metadata(metadata_filter)
                    if not candidate_ids:
                        return []

                # Perform search based on database type
                if self.db_type == "faiss" and self.vector_index is not None:
                    # Search in FAISS
                    distances, indices = self.vector_index.search(
                        np.array([query_embedding], dtype=np.float32),
                        min(k, len(self.patterns))
                    )

                    # Map indices back to patterns
                    pattern_ids = list(self.patterns.keys())
                    results = []

                    for i, idx in enumerate(indices[0]):
                        if idx < 0 or idx >= len(pattern_ids):
                            continue

                        pattern_id = pattern_ids[idx]
                        pattern = self.patterns[pattern_id]

                        # Filter by metadata if needed
                        if candidate_ids and pattern_id not in candidate_ids:
                            continue

                        # Convert distance to similarity
                        similarity = 1.0 / (1.0 + distances[0][i])

                        results.append({
                            "pattern_id": pattern_id,
                            "similarity": float(similarity),
                            "metadata": pattern.metadata,
                            "timestamp": pattern.timestamp.isoformat()
                        })

                elif self.db_type == "chroma":
                    # Search in ChromaDB
                    search_results = self.chroma_collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=k,
                        where=metadata_filter
                    )

                    results = []
                    for i, pattern_id in enumerate(search_results["ids"][0]):
                        pattern = self.patterns.get(pattern_id)
                        if not pattern:
                            continue

                        similarity = float(search_results["distances"][0][i])

                        results.append({
                            "pattern_id": pattern_id,
                            "similarity": similarity,
                            "metadata": pattern.metadata,
                            "timestamp": pattern.timestamp.isoformat()
                        })

                elif self.db_type == "lance" and LANCE_AVAILABLE:
                    # Search in LanceDB
                    search_results = self.lance_table.search(query_embedding.tolist()).limit(k)
                    if metadata_filter:
                        # Apply filters
                        for key, value in metadata_filter.items():
                            search_results = search_results.where(f"json_extract(metadata, '$.{key}') = '{value}'")

                    # Get results
                    results_df = search_results.to_pandas()

                    results = []
                    for _, row in results_df.iterrows():
                        pattern_id = row["id"]
                        pattern = self.patterns.get(pattern_id)
                        if not pattern:
                            continue

                        similarity = float(row["_distance"])

                        results.append({
                            "pattern_id": pattern_id,
                            "similarity": similarity,
                            "metadata": pattern.metadata,
                            "timestamp": pattern.timestamp.isoformat()
                        })

                else:
                    # Fallback: brute force search
                    results = self._brute_force_search(query_embedding, k, candidate_ids)

                # Update statistics
                query_time = (time.time() - start_time) * 1000  # ms
                self.stats["queries_processed"] += 1
                self.stats["avg_query_time_ms"] = (
                    (self.stats["avg_query_time_ms"] * (self.stats["queries_processed"] - 1) + query_time) /
                    self.stats["queries_processed"]
                )

                return results

            except Exception as e:
                logger.error(f"Error searching similar patterns: {e}")
                return []

    def _brute_force_search(self, query_embedding: np.ndarray, k: int, candidate_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform brute force similarity search.

        Args:
            query_embedding: Query embedding
            k: Number of results
            candidate_ids: Optional set of pattern IDs to consider

        Returns:
            List of similar patterns with similarity scores
        """
        patterns_to_search = self.patterns
        if candidate_ids:
            patterns_to_search = {pid: self.patterns[pid] for pid in candidate_ids if pid in self.patterns}

        if not patterns_to_search:
            return []

        # Calculate similarities
        similarities = []
        for pattern_id, pattern in patterns_to_search.items():
            similarity = self.similarity_calculator.calculate(query_embedding, pattern.embedding)
            similarities.append((pattern_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for pattern_id, similarity in similarities[:k]:
            pattern = self.patterns[pattern_id]
            results.append({
                "pattern_id": pattern_id,
                "similarity": float(similarity),
                "metadata": pattern.metadata,
                "timestamp": pattern.timestamp.isoformat()
            })

        return results

    def _filter_by_metadata(self, metadata_filter: Dict[str, Any]) -> Set[str]:
        """
        Filter patterns by metadata.

        Args:
            metadata_filter: Metadata filter criteria

        Returns:
            Set of matching pattern IDs
        """
        if not metadata_filter:
            return set(self.patterns.keys())

        result_sets = []

        for key, value in metadata_filter.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                result_sets.append(self.metadata_index[key][value])
            else:
                # If any filter criterion doesn't match, return empty set
                return set()

        # Intersect all result sets
        if not result_sets:
            return set()

        result = result_sets[0]
        for s in result_sets[1:]:
            result &= s

        return result

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern data or None if not found
        """
        if pattern_id not in self.patterns:
            return None

        pattern = self.patterns[pattern_id]
        pattern.access()  # Update access stats

        return {
            "pattern_id": pattern_id,
            "data": pattern.data.tolist() if isinstance(pattern.data, np.ndarray) else pattern.data,
            "embedding": pattern.embedding.tolist(),
            "metadata": pattern.metadata,
            "timestamp": pattern.timestamp.isoformat(),
            "access_count": pattern.access_count,
            "last_accessed": pattern.last_accessed.isoformat() if pattern.last_accessed else None
        }

    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            True if deleted, False otherwise
        """
        if pattern_id not in self.patterns:
            return False

        try:
            # Remove from patterns
            pattern = self.patterns.pop(pattern_id)

            # Remove from metadata index
            for key, value in pattern.metadata.items():
                if key in self.metadata_index and value in self.metadata_index[key]:
                    self.metadata_index[key][value].discard(pattern_id)

            # Remove from vector index
            # Note: This is not efficient for FAISS - we'd need to rebuild the index
            # For production, we'd mark it as deleted and periodically rebuild

            # Update statistics
            self.stats["total_patterns"] = len(self.patterns)

            return True

        except Exception as e:
            logger.error(f"Error deleting pattern: {e}")
            return False

    async def _maintenance(self) -> None:
        """Perform periodic maintenance operations."""
        try:
            # Save patterns to disk
            self._save_patterns()

            # Update statistics
            self.stats["last_maintenance"] = datetime.now().isoformat()

            # Cleanup old patterns if needed
            if len(self.patterns) > self.max_patterns:
                self._cleanup_old_patterns()

            logger.info(f"Memory maintenance completed: {len(self.patterns)} patterns in storage")

        except Exception as e:
            logger.error(f"Error during maintenance: {e}")

    def _cleanup_old_patterns(self) -> None:
        """Remove old or less useful patterns if storage is full."""
        if len(self.patterns) <= self.max_patterns:
            return

        # Sort patterns by utility (access count and recency)
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1].access_count + (0 if x[1].last_accessed is None else (datetime.now() - x[1].last_accessed).total_seconds()),
        )

        # Keep only the most useful patterns
        patterns_to_remove = sorted_patterns[:len(self.patterns) - self.max_patterns]

        for pattern_id, _ in patterns_to_remove:
            self.delete_pattern(pattern_id)

        logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary of statistics
        """
        # Update index size
        if self.db_type == "faiss" and self.vector_index is not None:
            # Estimate FAISS index size
            self.stats["index_size_bytes"] = len(self.patterns) * self.embedding_dim * 4  # float32

        # Return stats
        return self.stats
