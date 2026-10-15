"""AI 축 공통 함수 — API 방식(배치)과 MCP 서버가 같은 함수를 공유한다.

- get_profile : BigQuery 뷰에서 가구 프로필 1건 조회 (4개 키)
- search_docs : 질문을 임베딩 → BigQuery VECTOR_SEARCH로 문서 조각 검색
- embed       : 문서/질문 임베딩 (문서와 질문은 반드시 같은 모델)
- search_hybrid : BM25(키워드) + 벡터 검색 순위를 RRF로 합침 (옵션, 기본은 search_docs)
- embed_query : 질문 임베딩. BigQuery query_embeddings 캐시에 있으면 그 벡터를 씀
                → 배포 서버(Streamlit Cloud)가 2.2GB 모델을 올리지 않고도 검색 가능 (캐시에 없을 때만 모델 로딩)
- 인증: Streamlit secrets의 [gcp_service_account]가 있으면 서비스 계정, 없으면 로컬 기본 인증(gcloud)
"""
import re
from functools import lru_cache

import numpy as np
from google.cloud import bigquery

PROJECT = "dunnhumby-analysis"
DS = f"{PROJECT}.dunnhumby"
# 임베딩 모델 (2026-10-15 검색 평가로 BGE-M3 채택: paraphrase hit@3 0.933 vs BM25 0.733 vs e5-small 0.600)
EMBED_MODEL = "BAAI/bge-m3"                      # 다국어(한국어 강함), 1024차원, 로컬 실행 (~2.2GB)
# e5 계열은 'query: ' / 'passage: ' 접두어 필요, BGE-M3는 접두어 없이 사용
PREFIX = {"intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "}}


def _credentials():
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]))
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _client():
    return bigquery.Client(project=PROJECT, credentials=_credentials())


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts, kind="passage"):
    """kind='passage'(문서) 또는 'query'(질문). 정규화된 벡터 리스트 반환"""
    pre = PREFIX.get(EMBED_MODEL, {}).get(kind, "")
    prefixed = [pre + t for t in texts]
    vecs = _model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=float)


@lru_cache(maxsize=1)
def _query_cache() -> dict:
    """{질문 문장: 벡터} — ai_06_query_cache.py로 미리 계산해 BigQuery에 올려 둔 것 (같은 모델만)"""
    try:
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("m", "STRING", EMBED_MODEL)])
        rows = _client().query(f"SELECT query, embedding FROM `{DS}.query_embeddings` WHERE model = @m",
                               job_config=cfg).result()
        return {r["query"]: np.asarray(r["embedding"], dtype=float) for r in rows}
    except Exception:
        return {}


def embed_query(query: str) -> np.ndarray:
    v = _query_cache().get(query)
    return v if v is not None else embed([query], kind="query")[0]


def get_profile(campaign_type: str, campaign: int, rule: str, household_key: int) -> dict | None:
    sql = f"""
        SELECT * FROM `{DS}.vw_ai_target_profile`
        WHERE campaign_type = @t AND campaign = @c AND rule = @r AND household_key = @h
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("t", "STRING", campaign_type),
        bigquery.ScalarQueryParameter("c", "INT64", int(campaign)),
        bigquery.ScalarQueryParameter("r", "STRING", rule),
        bigquery.ScalarQueryParameter("h", "INT64", int(household_key)),
    ])
    rows = list(_client().query(sql, job_config=cfg).result())
    return dict(rows[0]) if rows else None


def search_docs(query: str, k: int = 5, doc_types: list[str] | None = None) -> list[dict]:
    """질문과 가까운 문서 조각 k개. doc_types로 문서 종류 제한 가능 (예: ['규정·금지표현'])"""
    qvec = embed_query(query).tolist()
    where = "WHERE doc_type IN UNNEST(@types)" if doc_types else ""
    sql = f"""
        SELECT base.chunk_id, base.doc_id, base.doc_type, base.title, base.section, base.text,
               distance
        FROM VECTOR_SEARCH(
          (SELECT * FROM `{DS}.doc_chunks` {where}),
          'embedding',
          (SELECT @q AS embedding),
          top_k => {int(k)},
          distance_type => 'COSINE')
        ORDER BY distance
    """
    params = [bigquery.ArrayQueryParameter("q", "FLOAT64", qvec)]
    if doc_types:
        params.append(bigquery.ArrayQueryParameter("types", "STRING", doc_types))
    cfg = bigquery.QueryJobConfig(query_parameters=params)
    return [dict(r) for r in _client().query(sql, job_config=cfg).result()]


# ---------------------------------------------------------------------------
# 하이브리드 검색: BM25(문서 텍스트는 BigQuery doc_chunks에서 1회 읽어 메모리 색인) + VECTOR_SEARCH
def _tok(s):
    words = re.findall(r"[가-힣]+|[A-Za-z]+|\d+", s.lower())
    out = []
    for w in words:
        out.append(w)
        if re.match(r"[가-힣]", w) and len(w) > 1:
            out += [w[i:i + 2] for i in range(len(w) - 1)]
    return out


@lru_cache(maxsize=1)
def _bm25_index():
    from rank_bm25 import BM25Okapi
    df = _client().query(
        f"SELECT chunk_id, doc_id, doc_type, title, section, text FROM `{DS}.doc_chunks`").to_dataframe()
    return BM25Okapi([_tok(t) for t in df["text"]]), df


def search_hybrid(query: str, k: int = 5, doc_types: list[str] | None = None, pool: int = 20) -> list[dict]:
    bm25, df = _bm25_index()
    sub = df if not doc_types else df[df["doc_type"].isin(doc_types)]
    scores = bm25.get_scores(_tok(query))
    bm_rank = [df.loc[i, "chunk_id"] for i in np.argsort(-scores) if i in sub.index][:pool]
    vec_rank = [r["chunk_id"] for r in search_docs(query, k=pool, doc_types=doc_types)]
    score = {}
    for lst in (bm_rank, vec_rank):
        for r, c in enumerate(lst):
            score[c] = score.get(c, 0) + 1 / (60 + r + 1)
    top = [c for c, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]
    rows = df.set_index("chunk_id").loc[top].reset_index()
    rows["rrf"] = [score[c] for c in top]
    return rows.to_dict("records")
