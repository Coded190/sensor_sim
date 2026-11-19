"""Local catalog retrieval for task queries.

Sentence-BERT is used only when the package imports and a checkpoint is
already on disk. This environment uses scikit-learn TF-IDF cosine similarity.
Scores are ``tfidf_cosine`` and are not semantic-similarity values.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sensor_sim.scene_gen.asset_package import canonical_json_bytes, read_index, verify_package

logger = logging.getLogger("sensor_sim.scene_gen.task_retrieve")

QUERY_TEMPLATE_VERSION = "deterministic_local_template_v1"
SYNONYMS = {
    "book": ("book",),
    "cup": ("cup", "mug"),
    "mug": ("cup", "mug"),
    "table": ("table",),
    "shelf": ("shelf",),
    "crate": ("crate", "container"),
    "container": ("crate", "container"),
}
VECTORIZER_CONFIG = {
    "lowercase": True,
    "token_pattern": r"[a-z0-9_]+",
    "ngram_range": [1, 1],
    "stop_words": None,
    "min_df": 1,
    "max_df": 1.0,
    "use_idf": True,
    "smooth_idf": True,
    "sublinear_tf": False,
    "norm": "l2",
    "dtype": "float64",
}


def template_hash() -> str:
    payload = {"version": QUERY_TEMPLATE_VERSION, "synonyms": {key: list(value) for key, value in sorted(SYNONYMS.items())}}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def rank_key(item: dict) -> tuple:
    return (-float(item["score"]), item["asset_id"], item["version"], item["package_sha256"])


def expand_query(query: str) -> str:
    tokens = []
    for word in query.lower().replace("_", " ").split():
        tokens.extend(SYNONYMS.get(word, (word,)))
    return " ".join(tokens)


def embedding_backend() -> dict:
    try:
        import sentence_transformers  # type: ignore
    except ImportError:
        sentence_transformers = None
    return {
        "sentence_transformers_importable": sentence_transformers is not None,
        "local_checkpoint": None,
        "selected": "tfidf_cosine",
        "vectorizer": VECTORIZER_CONFIG,
        "query_template": QUERY_TEMPLATE_VERSION,
        "query_template_sha256": template_hash(),
    }


def catalog_snapshot_hash(catalog: Path) -> str:
    index = read_index(Path(catalog) / "index.json")
    return hashlib.sha256(canonical_json_bytes(index)).hexdigest()


def load_entries(catalog: Path) -> list[dict]:
    catalog = Path(catalog)
    index = read_index(catalog / "index.json")
    entries = []
    for asset_id in sorted(index["assets"]):
        versions = index["assets"][asset_id]
        for version in sorted(versions):
            record = versions[version]
            manifest = verify_package(catalog / asset_id / version)
            if manifest["package_sha256"] != record["package_sha256"]:
                raise RuntimeError(f"index hash mismatch for {asset_id}")
            entries.append(
                {
                    "asset_id": asset_id,
                    "version": version,
                    "package_sha256": manifest["package_sha256"],
                    "license_status": manifest["provenance"]["license_status"],
                    "technical_status": record["status"] if record["status"] != "renderer_validated" else "technically_validated",
                    "renderer_status": record["status"],
                    "text": _text(manifest),
                    "manifest": manifest,
                }
            )
    entries.sort(key=lambda item: (item["asset_id"], item["version"], item["package_sha256"]))
    return entries


def retrieve(catalog: Path, query: str, *, top_k: int = 10, require_renderer: bool = False) -> dict:
    """Rank eligible catalog assets. An empty or zero query is a no-match."""
    entries = load_entries(catalog)
    expanded = expand_query(query)
    eligible = []
    rejected = []
    for entry in entries:
        reason = _reject_reason(entry, require_renderer=require_renderer)
        if reason:
            rejected.append({**_public(entry), "rejection": reason})
        else:
            eligible.append(entry)
    if not expanded.strip() or not eligible:
        return _result(query, expanded, [], rejected, top_k, eligible_count=len(eligible))
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[a-z0-9_]+",
        ngram_range=(1, 1),
        stop_words=None,
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        norm="l2",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform([item["text"] for item in eligible])
    query_vector = vectorizer.transform([expanded])
    if query_vector.nnz == 0 or float(np.linalg.norm(query_vector.toarray())) == 0.0:
        return _result(query, expanded, [], rejected, top_k, eligible_count=len(eligible))
    scores = cosine_similarity(query_vector, matrix)[0]
    ranked = []
    for entry, score in zip(eligible, scores):
        if float(score) <= 0.0:
            continue
        item = _public(entry)
        item["score"] = float(score)
        item["score_type"] = "tfidf_cosine"
        ranked.append(item)
    ranked.sort(key=rank_key)
    logger.info(
        "task_retrieve event=rank query=%s eligible=%d returned=%d backend=tfidf_cosine",
        query,
        len(eligible),
        min(top_k, len(ranked)),
    )
    return _result(query, expanded, ranked[:top_k], rejected, top_k, eligible_count=len(eligible), ranked_count=len(ranked))


def _text(manifest: dict) -> str:
    geometry = manifest.get("geometry") or {}
    parts = [manifest["asset_id"].replace("_", " "), str(geometry.get("profile", "")), str(geometry.get("category", ""))]
    aliases = geometry.get("aliases") or []
    parts.extend(str(alias) for alias in aliases)
    for component in geometry.get("components") or []:
        parts.append(str(component.get("id", "")).replace("_", " "))
    for surface in manifest.get("support_surfaces") or []:
        parts.append(str(surface.get("id", "")).replace("_", " "))
    interior = geometry.get("interior") or {}
    if interior:
        parts.append("container interior " + str(interior.get("id", "")))
    return " ".join(part for part in parts if part)


def _reject_reason(entry: dict, *, require_renderer: bool) -> str:
    if entry["license_status"] != "permitted":
        return "license"
    if entry["renderer_status"] not in {"candidate", "technically_validated", "renderer_validated"}:
        return "technical"
    if require_renderer and entry["renderer_status"] != "renderer_validated":
        return "renderer"
    return ""


def _public(entry: dict) -> dict:
    return {
        "asset_id": entry["asset_id"],
        "version": entry["version"],
        "package_sha256": entry["package_sha256"],
        "license_status": entry["license_status"],
        "technical_status": "technically_validated" if entry["renderer_status"] in {"technically_validated", "renderer_validated"} else entry["renderer_status"],
        "renderer_status": entry["renderer_status"],
        "visual_verification": "pending",
    }


def _result(query, expanded, ranked, rejected, top_k, *, eligible_count, ranked_count=0) -> dict:
    return {
        "query": query,
        "expanded_query": expanded,
        "backend": "tfidf_cosine",
        "top_k": int(top_k),
        "eligible_count": int(eligible_count),
        "ranked_count": int(ranked_count),
        "returned_count": len(ranked),
        "results": ranked,
        "rejected": rejected,
    }
