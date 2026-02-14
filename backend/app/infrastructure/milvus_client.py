"""
Milvus Vector Database Client — Persistent Storage for Genomic Embeddings.

Manages the "genomic_profiles" collection in Milvus, providing high-performance
vector similarity search over 48-dimensional STR profile embeddings.

Collection Schema:
    - profile_id (VARCHAR, 36): UUID v4, primary key.
    - node_id (VARCHAR, 64): Originating node/agency identifier.
    - timestamp (INT64): Unix epoch of profile creation.
    - embedding (FLOAT_VECTOR, dim=48): Normalized STR allele vector.

Index Strategy:
    IVF_FLAT with nlist=128 is used for the initial deployment. This provides
    exact recall on small-to-medium datasets (<1M vectors) with acceptable
    latency. For production scale (>10M vectors), switch to HNSW with
    ef_construction=256, M=32 for sub-5ms p99 latency.

Consistency:
    Strong consistency is enforced for forensic operations. Every insert is
    immediately visible to subsequent searches. This is critical for law
    enforcement scenarios where a profile ingested at time T must be
    matchable at time T+1.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.engine.vectorizer import VECTOR_DIM

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COLLECTION_NAME: str = "genomic_profiles"
METRIC_TYPE: str = "COSINE"
INDEX_TYPE: str = "IVF_FLAT"
INDEX_PARAMS: Dict[str, Any] = {"nlist": 128}
SEARCH_PARAMS: Dict[str, Any] = {"nprobe": 16}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class MilvusSearchHit(BaseModel):
    """A single search result from Milvus."""
    profile_id: str
    node_id: str
    timestamp: int
    distance: float = Field(..., ge=0.0, le=2.0, description="Cosine distance (0 = identical)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="1 - distance = similarity")


class MilvusSearchResult(BaseModel):
    """Structured result from a top-K similarity search."""
    query_id: str
    hits: List[MilvusSearchHit]
    total_candidates: int


class CollectionStats(BaseModel):
    """Metadata about the genomic_profiles collection."""
    collection_name: str = COLLECTION_NAME
    total_entities: int
    index_type: str = INDEX_TYPE
    metric_type: str = METRIC_TYPE
    vector_dim: int = VECTOR_DIM


# ═══════════════════════════════════════════════════════════════════════════════
# MILVUS CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class MilvusGenomicClient:
    """
    Manages the Milvus 'genomic_profiles' collection for vector storage and search.

    Provides:
        - Collection creation with enforced schema.
        - Single and batch profile insertion/upsertion.
        - Top-K similarity search with cosine distance.
        - Collection statistics and health checks.

    Connection lifecycle:
        Instantiate with host/port, call connect() to establish the session,
        and ensure_collection() to create or verify the collection schema.
        Call disconnect() on shutdown to release resources.
    """

    def __init__(self, host: str = "localhost", port: int = 19530) -> None:
        """
        Initialize the client with Milvus connection parameters.

        Args:
            host: Milvus server hostname or IP.
            port: Milvus gRPC port (default 19530).
        """
        self._host = host
        self._port = port
        self._connected = False
        self._collection: Any = None

    def connect(self) -> None:
        """
        Establish connection to the Milvus server.

        Uses pymilvus connections module for gRPC transport. In production,
        this should be wrapped with retry logic and circuit breaker patterns.

        Raises:
            ConnectionError: If Milvus server is unreachable.
        """
        try:
            from pymilvus import connections
            connections.connect(alias="default", host=self._host, port=str(self._port))
            self._connected = True
            logger.info(f"Connected to Milvus at {self._host}:{self._port}")
        except Exception as exc:
            self._connected = False
            raise ConnectionError(
                f"Failed to connect to Milvus at {self._host}:{self._port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Release the Milvus connection gracefully."""
        if self._connected:
            try:
                from pymilvus import connections
                connections.disconnect(alias="default")
                self._connected = False
                logger.info("Disconnected from Milvus")
            except Exception as exc:
                logger.warning(f"Error during Milvus disconnect: {exc}")

    def ensure_collection(self) -> None:
        """
        Create the genomic_profiles collection if it does not exist.

        Schema:
            - profile_id: VARCHAR(36), primary key. UUID v4 format.
            - node_id: VARCHAR(64). Originating forensic node identifier.
            - timestamp: INT64. Unix epoch of profile creation.
            - embedding: FLOAT_VECTOR(48). Normalized STR allele vector.

        Index:
            IVF_FLAT on the embedding field with COSINE metric. This index
            type partitions the vector space into nlist=128 clusters and
            searches nprobe=16 at query time, providing a balance between
            recall and latency for datasets under 1M vectors.
        """
        self._assert_connected()

        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if utility.has_collection(COLLECTION_NAME):
            self._collection = Collection(COLLECTION_NAME)
            self._collection.load()
            logger.info(f"Collection '{COLLECTION_NAME}' already exists, loaded.")
            return

        fields = [
            FieldSchema(
                name="profile_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=36,
                description="UUID v4 — unique profile identifier",
            ),
            FieldSchema(
                name="node_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Originating node/agency identifier",
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.INT64,
                description="Unix epoch of profile creation",
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
                description="48-dim normalized STR allele vector",
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="VANTAGE-STR genomic profile embeddings for forensic matching",
        )

        self._collection = Collection(
            name=COLLECTION_NAME,
            schema=schema,
            consistency_level="Strong",
        )

        # Build IVF_FLAT index on the embedding field
        self._collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": INDEX_TYPE,
                "metric_type": METRIC_TYPE,
                "params": INDEX_PARAMS,
            },
        )

        self._collection.load()
        logger.info(f"Collection '{COLLECTION_NAME}' created with IVF_FLAT index.")

    def insert_profile(
        self,
        profile_id: str,
        node_id: str,
        embedding: List[float],
        timestamp: int,
    ) -> str:
        """
        Insert a single genomic profile into the collection.

        If a profile with the same profile_id already exists, Milvus will
        reject the insert (primary key uniqueness). Use batch_upsert for
        update-or-insert semantics.

        Args:
            profile_id: UUID v4 string.
            node_id: Originating node identifier.
            embedding: 48-dimensional float vector.
            timestamp: Unix epoch timestamp.

        Returns:
            The profile_id of the inserted record.

        Raises:
            ValueError: If embedding dimension does not match VECTOR_DIM.
            RuntimeError: If collection is not initialized.
        """
        self._assert_collection()
        self._validate_embedding(embedding)

        data = [
            [profile_id],
            [node_id],
            [timestamp],
            [embedding],
        ]

        self._collection.insert(data)
        self._collection.flush()
        logger.info(f"Inserted profile {profile_id} from node {node_id}")
        return profile_id

    def batch_upsert(
        self,
        profiles: List[Dict[str, Any]],
    ) -> int:
        """
        Upsert multiple genomic profiles in a single batch operation.

        Each profile dict must contain keys: profile_id, node_id, embedding, timestamp.
        Upsert semantics: if a profile_id exists, it is updated; otherwise, inserted.

        Args:
            profiles: List of profile dicts with required keys.

        Returns:
            Number of profiles successfully upserted.

        Raises:
            ValueError: If any profile has invalid embedding dimension.
            RuntimeError: If collection is not initialized.
        """
        self._assert_collection()

        if not profiles:
            return 0

        profile_ids: List[str] = []
        node_ids: List[str] = []
        timestamps: List[int] = []
        embeddings: List[List[float]] = []

        for p in profiles:
            self._validate_embedding(p["embedding"])
            profile_ids.append(p["profile_id"])
            node_ids.append(p["node_id"])
            timestamps.append(p["timestamp"])
            embeddings.append(p["embedding"])

        data = [profile_ids, node_ids, timestamps, embeddings]
        self._collection.upsert(data)
        self._collection.flush()

        count = len(profiles)
        logger.info(f"Batch upserted {count} profiles")
        return count

    def search_top_k(
        self,
        query_vector: List[float],
        top_k: int = 10,
        node_filter: Optional[str] = None,
        query_id: str = "anonymous",
    ) -> MilvusSearchResult:
        """
        Search the collection for the top-K most similar genomic profiles.

        Uses cosine distance — lower distance = higher similarity.
        The confidence_score is computed as (1 - distance) for intuitive
        interpretation where 1.0 = perfect match.

        Args:
            query_vector: 48-dimensional query embedding.
            top_k: Number of closest matches to return (default 10).
            node_filter: Optional node_id filter expression (e.g., "node_id == 'FBI-US-TX'").
            query_id: Identifier for the query profile (for result tracking).

        Returns:
            MilvusSearchResult with ranked hits and confidence scores.

        Raises:
            ValueError: If query vector dimension does not match VECTOR_DIM.
            RuntimeError: If collection is not initialized.
        """
        self._assert_collection()
        self._validate_embedding(query_vector)

        search_kwargs: Dict[str, Any] = {
            "data": [query_vector],
            "anns_field": "embedding",
            "param": {"metric_type": METRIC_TYPE, "params": SEARCH_PARAMS},
            "limit": top_k,
            "output_fields": ["profile_id", "node_id", "timestamp"],
        }

        if node_filter:
            search_kwargs["expr"] = node_filter

        results = self._collection.search(**search_kwargs)

        hits: List[MilvusSearchHit] = []
        for result_set in results:
            for hit in result_set:
                distance = float(hit.distance)
                hits.append(MilvusSearchHit(
                    profile_id=hit.entity.get("profile_id"),
                    node_id=hit.entity.get("node_id"),
                    timestamp=hit.entity.get("timestamp"),
                    distance=distance,
                    confidence_score=round(max(0.0, 1.0 - distance), 8),
                ))

        return MilvusSearchResult(
            query_id=query_id,
            hits=hits,
            total_candidates=self._collection.num_entities,
        )

    def get_stats(self) -> CollectionStats:
        """
        Return current collection statistics.

        Useful for the frontend Global Network Status indicator to display
        the total number of indexed profiles.
        """
        self._assert_collection()
        return CollectionStats(total_entities=self._collection.num_entities)

    def drop_collection(self) -> None:
        """
        Drop the entire collection. DESTRUCTIVE — use only in testing.

        Removes all data and the index. Cannot be undone.
        """
        self._assert_connected()
        from pymilvus import utility

        if utility.has_collection(COLLECTION_NAME):
            from pymilvus import Collection
            Collection(COLLECTION_NAME).drop()
            self._collection = None
            logger.warning(f"Collection '{COLLECTION_NAME}' dropped.")

    # ───────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ───────────────────────────────────────────────────────────────────────────

    def _assert_connected(self) -> None:
        """Raise if not connected to Milvus."""
        if not self._connected:
            raise RuntimeError(
                "Not connected to Milvus. Call connect() first."
            )

    def _assert_collection(self) -> None:
        """Raise if collection is not initialized."""
        if self._collection is None:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not initialized. "
                "Call ensure_collection() first."
            )

    @staticmethod
    def _validate_embedding(embedding: List[float]) -> None:
        """
        Validate that an embedding vector has the correct dimension.

        The forensic vector space is strictly 48-dimensional (24 loci × 2 alleles).
        Vectors of incorrect dimension would corrupt the index and produce
        meaningless similarity results.

        Raises:
            ValueError: If embedding length != VECTOR_DIM.
        """
        if len(embedding) != VECTOR_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {VECTOR_DIM}, "
                f"got {len(embedding)}."
            )
