"""AI 축 S5~S7: 브리프 스키마 · 프롬프트 · 검증 · 생성 (API 방식과 MCP 서버가 공유)

흐름: 프로필 조회(SQL) → 문서 검색(벡터) + 규정 고정 포함 → LLM(구조화 출력) → 코드 검증 → 실패 시 오류를 알려주고 재생성(최대 2회)
LLM은 LangChain으로 감싸 Claude / Gemini 교체 가능 (.env: ANTHROPIC_API_KEY, GOOGLE_API_KEY, CLAUDE_MODEL, GEMINI_MODEL)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Literal

from pydantic import BaseModel, Field

import ai_common as ac

STORE = "우리마트"                     # 가상 매장명
OPT_OUT = "무료수신거부 080-000-0000"   # 가상 번호
FIXED_DOC_IDS = ["ad_rules"]           # 안전 규정: 검색에 맡기지 않고 항상 포함
SEARCH_DOC_TYPES = ["문구가이드", "상품·쿠폰설명", "운영정책", "톤가이드", "세그먼트가이드"]
MAX_LMS_CHARS = 250


# ---------------------------------------------------------------------------
# 1. 출력 스키마 (다른 부서 전달용 JSON)
class WhyItem(BaseModel):
    factor: str = Field(description="선정 요인 이름 (한국어)")
    evidence: str = Field(description="데이터 근거 한 문장. 예: '사전 8주 장바구니당 $40.5 (전체 중앙값 $24.7)'")
    direction: Literal["+", "-"] = Field(description="예상 증분을 키웠으면 +")


class Product(BaseModel):
    product_id: int | None = Field(description="쿠폰 대상 상품 ID. 상품 이력이 없어 카테고리로 추천하면 null")
    category_en: str = Field(description="원본 영문 카테고리 (데이터 그대로)")
    category_ko: str = Field(description="한국어 카테고리명")
    reason: str = Field(description="추천 이유 (구매 이력 사실 기반)")


class CustomerMessage(BaseModel):
    channel: Literal["LMS", "SMS"] = "LMS"
    text: str = Field(description="고객에게 보낼 문구 전체 ((광고)로 시작, 매장명, 혜택, 기간, 수신거부 포함)")


class Brief(BaseModel):
    why_target: list[WhyItem] = Field(description="선정 이유 최대 3개 (SHAP 근거를 사람 말로)", max_length=3)
    recommended_products: list[Product] = Field(description="쿠폰 대상 중 추천 1~3개", min_length=1, max_length=3)
    talking_points: list[str] = Field(description="영업·마케팅 담당자용 응대 포인트 2~4개", min_length=2, max_length=4)
    customer_message: CustomerMessage
    sources: list[str] = Field(description="근거로 쓴 문서 조각 chunk_id 목록")
    caveats: str = Field(description="담당자가 알아야 할 주의점 한 문장")


# ---------------------------------------------------------------------------
# 2. 프롬프트
SYSTEM = f"""당신은 국내 마트 '{STORE}'의 CRM 분석가입니다. 분석팀이 고른 타깃 가구 1곳에 대해,
영업·마케팅 담당자에게 전달할 '타깃 고객 브리프'를 JSON으로 작성합니다.

[배경] 데이터는 미국 마트 데이터이며, 이 방법을 국내 마트에 이식하는 데모입니다. 상품 카테고리는 원본 영문과 한국어명을 함께 씁니다.
[목표] 쿠폰 사용률이 아니라 '고객이 원래 사지 않았을 상품을 한 번 더 담게 하는 것'입니다.

[작성 규칙]
1. 모든 사실은 아래 [가구 데이터]와 [참고 문서]에 있는 것만 씁니다. 없는 상품·할인 금액·기간을 지어내지 않습니다.
2. why_target: [가구 데이터]의 why_target(SHAP 상위 요인)을 사람 말로 바꾸고, 전체 중앙값 등 비교 기준을 붙입니다.
3. recommended_products: 이번 캠페인 쿠폰 대상 중에서만 고릅니다.
   - 우선순위: ① coupon_products_bought(이미 산 쿠폰 대상 상품) ② coupon_categories_bought(산 적 있는 쿠폰 대상 카테고리, product_id는 null)
     ③ 둘 다 없으면 top_categories와 문구 가이드의 '교차 제안 패턴'으로 연결되는 쿠폰 카테고리 (product_id null, reason에 '교차 제안' 명시)
   - 이미 매주 사는 기본 상품(우유·계란)만 단독 추천하지 않습니다.
4. customer_message: 반드시 '(광고)[{STORE}]'로 시작하고, 마지막 줄은 '{OPT_OUT}'로 끝냅니다. {MAX_LMS_CHARS}자 이내.
   - 혜택 금액은 쿠폰 설명서의 값만, 기간은 '쿠폰 기간: 캠페인 기간 내'로 씁니다.
   - 구매 횟수·금액·방문 빈도 같은 숫자, 가구 구성·소득·연령 추정, 이모지, 금지 표현을 쓰지 않습니다.
5. talking_points: 직원용이므로 데이터 근거(숫자)를 써도 됩니다. 단 고객에게 그대로 말하지 않도록 표현합니다.
6. sources: 실제로 근거로 쓴 문서 조각의 chunk_id를 적습니다.
7. 모든 문장은 한국어, 존댓말(~해요/~합니다)로 씁니다."""

USER_TMPL = """[가구 데이터]
{profile}

[광고 규정·금지 표현 — 반드시 준수]
{fixed}

[참고 문서 — 검색 결과]
{retrieved}
{feedback}"""


# ---------------------------------------------------------------------------
# 3. 검증 (규정 문서 5장 체크리스트와 동일)
PROHIBITED = [
    "최저가", "업계 최초", "1위", "최고", "유일", "무조건", "100%", "완벽", "절대",
    "안 사면 손해", "마지막 기회", "놓치면 후회", "품절 임박",
    "면역", "다이어트", "혈당", "뼈 건강", "질병", "피로 회복", "살 빠", "건강에 좋",
    "VIP", "자주 오시", "아이 있는", "자녀", "알뜰한", "큰손", "타사", "보다 저렴",
]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
NUMERIC_HISTORY = re.compile(r"(\d+\s*(번|회)\s*(구매|방문|오셨))|\$\s*\d")
WON = re.compile(r"\d[\d,]*\s*원")


def amounts(text: str) -> set[str]:
    """'1,000원', '1000 원' → '1000'"""
    return {re.sub(r"[^\d]", "", m) for m in WON.findall(text)}


_AMOUNT_PAT = re.compile(r"##\s*[^\n(]*\(([^)\n]+)\)[^\n]*\n(?:(?!\n##).)*?혜택\(가상\):[^\n]*?([\d,]+)\s*원", re.S)
_amount_cache: dict[int, dict[str, str]] = {}


def coupon_amount_map(campaign: int) -> dict[str, str]:
    """쿠폰 설명서(doc_chunks)에서 {영문 카테고리: 할인 금액} 표를 만든다. 예: {'CHEESE': '1000'}"""
    if campaign not in _amount_cache:
        from google.cloud import bigquery
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("d", "STRING", f"coupon_catalog_c{int(campaign)}")])
        rows = ac._client().query(
            f"SELECT text FROM `{ac.DS}.doc_chunks` WHERE doc_id = @d ORDER BY chunk_id", job_config=cfg).result()
        text = "\n".join(r["text"] for r in rows)
        _amount_cache[campaign] = {c.strip(): a.replace(",", "") for c, a in _AMOUNT_PAT.findall(text)}
    return _amount_cache[campaign]


def validate(brief: Brief, profile: dict, allowed_categories: set[str],
             amount_map: dict[str, str] | None = None) -> dict:
    msg = brief.customer_message.text.strip()
    found = [w for w in PROHIBITED if w in msg]
    bought_ids = {p["product_id"] for p in (profile.get("coupon_products_bought") or [])}
    unknown = [p.category_en for p in brief.recommended_products
               if (p.product_id is not None and p.product_id not in bought_ids)
               or p.category_en not in allowed_categories]
    # 문구의 금액은 '추천한 카테고리'의 설명서 금액이어야 함 (다른 카테고리 금액을 붙이는 실수 차단)
    ok_amounts = {amount_map.get(p.category_en) for p in brief.recommended_products} if amount_map else None
    bad_amounts = sorted(amounts(msg) - ok_amounts) if ok_amounts is not None else []
    checks = {
        "ad_label": msg.startswith("(광고)"),
        "sender": STORE in msg,
        "opt_out": "수신거부" in msg and "080" in msg,
        "length_ok": len(msg) <= MAX_LMS_CHARS,
        "no_prohibited": not found,
        "no_emoji": not EMOJI.search(msg),
        "no_history_numbers": not NUMERIC_HISTORY.search(msg),
        "known_products": not unknown,
        "known_amounts": not bad_amounts,       # 문구의 할인 금액 = 추천 카테고리의 설명서 금액
    }
    return {"passed": all(checks.values()), "checks": checks, "char_count": len(msg),
            "prohibited_terms_found": found, "unknown_products": unknown,
            "unknown_amounts": bad_amounts}


def feedback_text(v: dict) -> str:
    bad = [k for k, ok in v["checks"].items() if not ok]
    extra = []
    if v["prohibited_terms_found"]:
        extra.append(f"금지 표현: {v['prohibited_terms_found']}")
    if v["unknown_products"]:
        extra.append(f"쿠폰 대상이 아니거나 이력에 없는 상품: {v['unknown_products']}")
    if v.get("unknown_amounts"):
        extra.append(f"추천 카테고리의 설명서 금액이 아님: {v['unknown_amounts']} (각 카테고리의 쿠폰 설명서 금액만 사용)")
    return ("\n[이전 출력의 검증 실패 — 아래를 고쳐서 다시 작성]\n실패 항목: " + ", ".join(bad)
            + ("\n" + "\n".join(extra) if extra else "") + f"\n글자 수: {v['char_count']} (최대 {MAX_LMS_CHARS})")


# ---------------------------------------------------------------------------
# 4. 문맥 만들기
_fixed_cache: str | None = None


def fixed_context() -> str:
    global _fixed_cache
    if _fixed_cache is None:
        sql = f"SELECT chunk_id, text FROM `{ac.DS}.doc_chunks` WHERE doc_id IN UNNEST(@ids) ORDER BY chunk_id"
        from google.cloud import bigquery
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", FIXED_DOC_IDS)])
        rows = ac._client().query(sql, job_config=cfg).result()
        _fixed_cache = "\n\n".join(f"<{r['chunk_id']}>\n{r['text']}" for r in rows)
    return _fixed_cache


# 검색 질문 문장 (ai_06_query_cache.py가 같은 함수로 미리 임베딩 → 문장을 바꾸면 캐시도 다시 만들 것)
def category_query(c: str) -> str:
    return f"{c} 쿠폰 혜택과 함께 제안할 상품"


def rule_query(r: str) -> str:
    return f"{r} 세그먼트 고객 응대 포인트"


FIXED_QUERIES = ["고객 문구 구조와 길이, 좋은 문구 예시", "브랜드 말투와 하지 말 것"]


def retrieval_queries(p: dict) -> list[str]:
    cats = [c["commodity_desc"] for c in (p.get("coupon_categories_bought") or [])][:2] \
        or [c["commodity_desc"] for c in (p.get("top_categories") or [])][:2]
    return [category_query(c) for c in cats] + [rule_query(p["rule"])] + FIXED_QUERIES


def build_context(p: dict, k_per_query: int = 3) -> tuple[str, list[str]]:
    seen, parts = set(), []
    for q in retrieval_queries(p):
        for r in ac.search_docs(q, k=k_per_query, doc_types=SEARCH_DOC_TYPES):
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                parts.append(f"<{r['chunk_id']}>\n{r['text']}")
    return "\n\n".join(parts), sorted(seen)


def compact_profile(p: dict) -> str:
    keep = {k: p[k] for k in p if k not in ("target_rank",)}
    return json.dumps(keep, ensure_ascii=False, default=str, indent=1)


# ---------------------------------------------------------------------------
# 5. LLM (LangChain, 교체 가능)
def make_llm(provider: str = "claude"):
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"), temperature=0.3, max_tokens=2000)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kw = {}   # 속도 조절(선택): 생각(thinking) 토큰을 줄이면 빨라짐. 모델 세대에 따라 쓰는 이름이 다름
        if os.getenv("GEMINI_THINKING_LEVEL"):
            kw["thinking_level"] = os.getenv("GEMINI_THINKING_LEVEL")          # 예: low
        if os.getenv("GEMINI_THINKING_BUDGET"):
            kw["thinking_budget"] = int(os.getenv("GEMINI_THINKING_BUDGET"))   # 예: 0 또는 512
        llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.3, **kw)
    else:
        raise ValueError(provider)
    return llm.with_structured_output(Brief, include_raw=True)


def generate_brief(llm, key: tuple, allowed_categories: set[str], max_retry: int = 2) -> dict:
    """key = (campaign_type, campaign, rule, household_key). 결과 dict(브리프 + 검증 + 비용·시간)"""
    t0 = time.time()
    timing = {}
    p = ac.get_profile(*key)
    timing["profile"] = round(time.time() - t0, 1)
    base = {"campaign_type": key[0], "campaign": key[1], "rule": key[2], "household_key": key[3]}
    if p is None:
        return {**base, "error": "프로필 없음"}
    t1 = time.time()
    retrieved, retrieved_ids = build_context(p)
    fixed = fixed_context()
    amount_map = coupon_amount_map(p["campaign"])
    timing["retrieval"] = round(time.time() - t1, 1)
    t_llm = 0.0
    feedback, tokens_in, tokens_out, attempts = "", 0, 0, 0
    brief, v = None, None
    for attempts in range(1, max_retry + 2):
        user = USER_TMPL.format(profile=compact_profile(p), fixed=fixed,
                                retrieved=retrieved, feedback=feedback)
        t2 = time.time()
        out = llm.invoke([("system", SYSTEM), ("human", user)])
        t_llm += time.time() - t2
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        tokens_in += usage.get("input_tokens", 0); tokens_out += usage.get("output_tokens", 0)
        brief = out["parsed"]
        if brief is None:
            feedback = "\n[이전 출력이 JSON 스키마를 지키지 않았습니다. 스키마대로 다시 작성하세요]"
            continue
        v = validate(brief, p, allowed_categories, amount_map)
        if v["passed"]:
            break
        feedback = feedback_text(v)
    return {
        **base,
        "brief": brief.model_dump() if brief else None,
        "validation": v, "attempts": attempts, "passed_first_try": attempts == 1 and bool(v and v["passed"]),
        "retrieved_chunks": retrieved_ids,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "seconds": round(time.time() - t0, 1),
        "timing": {**timing, "llm": round(t_llm, 1)},
    }


# ---------------------------------------------------------------------------
# 6. 캐시 저장 (배치·Streamlit 공용) — 샌드박스라 load job APPEND, 같은 키는 최신 created_at이 유효
def save_cache(results: list[dict], provider: str) -> int:
    import pandas as pd
    from datetime import datetime
    from google.cloud import bigquery
    rows = [{"campaign_type": x["campaign_type"], "campaign": int(x["campaign"]), "rule": x["rule"],
             "household_key": int(x["household_key"]), "provider": provider,
             "passed": bool((x.get("validation") or {}).get("passed")),
             "brief_json": json.dumps(x.get("brief"), ensure_ascii=False),
             "result_json": json.dumps(x, ensure_ascii=False, default=str),     # 검증·근거·시간까지 전체
             "created_at": datetime.now().isoformat()} for x in results if x.get("brief")]
    if rows:
        cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND",
                                     schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION])
        ac._client().load_table_from_dataframe(pd.DataFrame(rows), f"{ac.DS}.ai_briefs", job_config=cfg).result()
    return len(rows)
