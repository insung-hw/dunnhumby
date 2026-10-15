# AI 축 S8: 타깃 고객 브리프 조회·생성 화면 (Streamlit)
# 실행 (code 폴더에서):  streamlit run app_streamlit.py
# - 선택 4단계: 캠페인 타입 → 캠페인 → 규칙 → 고객 ID (앞 선택에 따라 뒤 목록이 좁혀짐)
# - 이미 만든 브리프(BigQuery ai_briefs)는 바로 보여주고, 없으면 [브리프 생성]으로 그 자리에서 생성 → 캐시에 추가
# - 사전 설치: pip install -r requirements.txt
# - 배포(Streamlit Cloud): secrets에 [gcp_service_account], GOOGLE_API_KEY, APP_PASSWORD
#   APP_PASSWORD가 있으면 비밀번호를 넣은 사람만 생성 가능 (공개 링크로 API 비용이 나가지 않게). 없으면(로컬) 누구나 생성
import hmac
import io
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import ai_common as ac
import ai_brief as ab

st.set_page_config(page_title="타깃 고객 브리프", layout="wide")


# ---------------------------------------------------------------------------
# 데이터 (BigQuery) — 목록은 10분 캐시, 브리프는 생성 직후 캐시 비움
@st.cache_data(ttl=600)
def load_targets() -> pd.DataFrame:
    return ac._client().query(f"""
        SELECT campaign_type, campaign, rule, household_key, target_rank
        FROM `{ac.DS}.ai_target_list`
        ORDER BY campaign_type, campaign, rule, target_rank
    """).to_dataframe()


@st.cache_data(ttl=600)
def load_briefs(provider: str) -> pd.DataFrame:
    """같은 키를 여러 번 만들었으면 가장 최근 것만"""
    try:
        cols = {f.name for f in ac._client().get_table(f"{ac.DS}.ai_briefs").schema}
    except Exception:
        return pd.DataFrame(columns=["campaign_type", "campaign", "rule", "household_key"])
    result_col = "result_json" if "result_json" in cols else "CAST(NULL AS STRING) AS result_json"
    return ac._client().query(f"""
        SELECT campaign_type, campaign, rule, household_key, passed, brief_json, {result_col}, created_at
        FROM `{ac.DS}.ai_briefs`
        WHERE provider = '{provider}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY campaign_type, campaign, rule, household_key
                                   ORDER BY created_at DESC) = 1
    """).to_dataframe()


@st.cache_data(ttl=600)
def allowed_categories(campaign: int) -> set[str]:
    return set(ac._client().query(f"""
        SELECT DISTINCT p.commodity_desc FROM `{ac.DS}.coupon` c
        JOIN `{ac.DS}.product` p USING (product_id) WHERE c.campaign = {int(campaign)}
    """).to_dataframe()["commodity_desc"])


@st.cache_data(ttl=600)
def profile(campaign_type, campaign, rule, household_key):
    return ac.get_profile(campaign_type, int(campaign), rule, int(household_key))


@st.cache_data(ttl=600)
def chunk_texts(chunk_ids: tuple[str, ...]) -> dict[str, str]:
    from google.cloud import bigquery
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", list(chunk_ids))])
    rows = ac._client().query(f"SELECT chunk_id, text FROM `{ac.DS}.doc_chunks` WHERE chunk_id IN UNNEST(@ids)",
                              job_config=cfg).result()
    return {r["chunk_id"]: r["text"] for r in rows}


@st.cache_resource
def llm(provider: str):
    return ab.make_llm(provider)


def brief_rows(x: dict) -> tuple[dict, dict]:
    """엑셀 1행씩: (브리프 시트, 고객문구 시트) — ai_05의 엑셀과 같은 열"""
    b, v = x["brief"], x.get("validation") or {}
    base = {"campaign": x["campaign"], "rule": x["rule"], "household_key": x["household_key"]}
    r1 = {**base,
          "선정 이유": "\n".join(f"[{w['direction']}] {w['factor']}: {w['evidence']}" for w in b["why_target"]),
          "추천 상품": "\n".join(f"{p['category_ko']} ({p['category_en']}, id={p['product_id']}): {p['reason']}"
                             for p in b["recommended_products"]),
          "응대 포인트": "\n".join(f"- {t}" for t in b["talking_points"]),
          "주의": b["caveats"], "근거 문서": ", ".join(b["sources"])}
    r2 = {**base, "채널": b["customer_message"]["channel"], "문구": b["customer_message"]["text"],
          "검증 통과": v.get("passed"), "발송 상태": "담당자 확인 대기"}
    return r1, r2


def to_result(row, key, prof) -> dict:
    """캐시 1행 → 결과 dict. 예전 행(result_json 없음)은 검증을 다시 계산"""
    if isinstance(row.get("result_json"), str) and row["result_json"]:
        return json.loads(row["result_json"])
    brief = ab.Brief.model_validate(json.loads(row["brief_json"]))
    v = ab.validate(brief, prof, allowed_categories(key[1]), ab.coupon_amount_map(key[1]))
    return {"campaign_type": key[0], "campaign": key[1], "rule": key[2], "household_key": key[3],
            "brief": brief.model_dump(), "validation": v, "created_at": row.get("created_at"),
            "note": "예전 캐시 행: 검증은 화면에서 다시 계산, 검색 조각·시간 기록 없음"}


# ---------------------------------------------------------------------------
# 사이드바: 선택 4단계
with st.sidebar:
    st.header("대상 선택")
    provider = st.radio("LLM", ["gemini", "claude"], horizontal=True)
    only_done = st.checkbox("브리프가 있는 고객만 보기", value=True)
    APP_PASSWORD = os.getenv("APP_PASSWORD", "")
    if APP_PASSWORD:
        pw = st.text_input("생성 비밀번호 (관리자용)", type="password")
        can_generate = bool(pw) and hmac.compare_digest(pw, APP_PASSWORD)
        if not can_generate:
            st.caption("공개 데모: 저장된 브리프 조회만 가능합니다.")
    else:
        can_generate = True

    targets, briefs = load_targets(), load_briefs(provider)
    done_keys = set(zip(briefs["campaign_type"], briefs["campaign"], briefs["rule"], briefs["household_key"]))
    targets["done"] = [k in done_keys for k in zip(targets.campaign_type, targets.campaign,
                                                    targets.rule, targets.household_key)]
    pool = targets[targets["done"]] if only_done else targets
    if pool.empty:
        st.warning("조건에 맞는 고객이 없습니다. '브리프가 있는 고객만 보기'를 끄면 전체 목록이 보입니다.")
        st.stop()

    ctype = st.selectbox("① 캠페인 타입", sorted(pool.campaign_type.unique()))
    p1 = pool[pool.campaign_type == ctype]
    camp = st.selectbox("② 캠페인", sorted(p1.campaign.unique()))
    p2 = p1[p1.campaign == camp]
    rule = st.selectbox("③ 타깃 규칙", sorted(p2.rule.unique()))
    p3 = p2[p2.rule == rule].sort_values("target_rank")
    labels = {int(r.household_key): f"{int(r.household_key)}  (순위 {int(r.target_rank)}{' · 생성됨' if r.done else ''})"
              for r in p3.itertuples()}
    hh = st.selectbox("④ 고객 ID (household_key)", list(labels), format_func=labels.get)

    st.caption(f"이 캠페인·규칙: 대상 {len(targets[(targets.campaign == camp) & (targets.rule == rule)])}가구 / "
               f"브리프 {int(targets[(targets.campaign == camp) & (targets.rule == rule)].done.sum())}건")
    st.divider()
    st.caption("데이터: dunnhumby The Complete Journey (미국) · 문서·매장명·할인 금액은 데모용 가상 값 · "
               "LLM에는 household_key와 집계 지표만 전달 (개인정보 없음)")

key = (ctype, int(camp), rule, int(hh))
prof = profile(*key)
st.title("타깃 고객 브리프")
st.caption(f"캠페인 {camp} ({ctype}) · 규칙 {rule} · 고객 {hh} · LLM {provider}")

if prof is None:
    st.error("프로필이 없습니다 (vw_ai_target_profile에 이 키가 없음).")
    st.stop()

# ---------------------------------------------------------------------------
# 브리프: 캐시에 있으면 표시, 없으면 생성
hit = briefs[(briefs.campaign_type == ctype) & (briefs.campaign == camp) &
             (briefs.rule == rule) & (briefs.household_key == hh)]
c1, c2 = st.columns([1, 4])
regen = False
if hit.empty:
    gen = c1.button("브리프 생성", type="primary", disabled=not can_generate)
    c2.info("아직 브리프가 없습니다. 생성에 약 30초~1분 걸립니다 (처음 1회는 임베딩 모델 로딩 시간 추가).")
else:
    gen = False
    regen = c1.button("다시 생성", disabled=not can_generate)

if gen or regen:
    with st.spinner("프로필 조회 → 문서 검색 → 생성 → 검증 중..."):
        try:
            res = ab.generate_brief(llm(provider), key, allowed_categories(camp))
            res["provider"] = provider
            if res.get("brief"):
                ab.save_cache([res], provider)
                load_briefs.clear()
            st.session_state["last"] = (key, res)
        except Exception as e:
            st.error(f"생성 실패: {e!r}"[:500])
            st.stop()
    st.rerun()

last = st.session_state.get("last")
if hit.empty and not (last and last[0] == key):
    st.stop()
if not hit.empty:
    x = to_result(hit.iloc[0].to_dict(), key, prof)
else:
    x = last[1]
    if x.get("brief") is None:
        st.error("생성은 됐지만 스키마를 지키지 않아 브리프가 비었습니다. 다시 생성해 주세요.")
        st.stop()
b, v = x["brief"], x.get("validation") or {}

# 상단 요약
m = st.columns(5)
m[0].metric("검증", "통과" if v.get("passed") else "실패")
m[1].metric("시도 횟수", x.get("attempts", "-"))
m[2].metric("문구 글자 수", f"{v.get('char_count', '-')} / {ab.MAX_LMS_CHARS}")
m[3].metric("소요 시간(초)", x.get("seconds", "-"))
m[4].metric("토큰 입력/출력", f"{x.get('tokens_in', '-')} / {x.get('tokens_out', '-')}")
if x.get("note"):
    st.caption(x["note"])

tab1, tab2, tab3, tab4 = st.tabs(["직원용 브리프", "고객 문구", "근거·검증", "프로필 원본"])

with tab1:
    st.subheader("선정 이유")
    for w in b["why_target"]:
        st.markdown(f"- **{w['factor']}** ({w['direction']}) — {w['evidence']}")
    st.subheader("추천 상품 (캠페인 쿠폰 대상)")
    st.dataframe(pd.DataFrame(b["recommended_products"])[["category_ko", "category_en", "product_id", "reason"]]
                 .rename(columns={"category_ko": "카테고리", "category_en": "원본 카테고리",
                                  "product_id": "상품 ID", "reason": "추천 이유"}),
                 hide_index=True, width="stretch")
    st.subheader("응대 포인트")
    for t in b["talking_points"]:
        st.markdown(f"- {t}")
    st.warning(b["caveats"])

with tab2:
    msg = b["customer_message"]
    st.caption(f"채널: {msg['channel']} · 발송 전 담당자 확인 필요")
    st.text_area("문구 (복사용)", msg["text"], height=220)

with tab3:
    st.subheader("코드 검증")
    names = {"ad_label": "(광고) 표기로 시작", "sender": "매장명 포함", "opt_out": "무료수신거부 번호",
             "length_ok": f"{ab.MAX_LMS_CHARS}자 이내", "no_prohibited": "금지 표현 없음", "no_emoji": "이모지 없음",
             "no_history_numbers": "구매 이력 숫자 없음", "known_products": "쿠폰 대상·이력 상품만",
             "known_amounts": "할인 금액 = 추천 카테고리의 설명서 금액"}
    st.dataframe(pd.DataFrame([{"항목": names.get(k, k), "결과": "통과" if ok else "실패"}
                               for k, ok in (v.get("checks") or {}).items()]), hide_index=True)
    for k, label in [("prohibited_terms_found", "금지 표현"), ("unknown_products", "확인 안 된 상품"),
                     ("unknown_amounts", "맞지 않는 금액")]:
        if v.get(k):
            st.error(f"{label}: {v[k]}")
    if x.get("timing"):
        st.caption(f"단계별 시간(초): {x['timing']}")
    st.subheader("근거 문서 조각")
    st.caption("LLM이 근거로 적은 조각 (광고 규정 문서는 검색 없이 항상 포함)")
    texts = chunk_texts(tuple(b["sources"])) if b["sources"] else {}
    for cid in b["sources"]:
        with st.expander(cid):
            st.text(texts.get(cid, "(doc_chunks에 없는 조각 ID — 확인 필요)"))
    if x.get("retrieved_chunks"):
        st.caption("검색으로 가져온 조각: " + ", ".join(x["retrieved_chunks"]))

with tab4:
    st.caption("LLM에 전달한 가구 데이터 (vw_ai_target_profile)")
    st.json(json.loads(json.dumps(prof, default=str)))

# ---------------------------------------------------------------------------
# 내려받기: 이 고객 JSON / 이 캠페인·규칙의 전체 브리프 엑셀
st.divider()
d1, d2 = st.columns(2)
d1.download_button("이 고객 JSON 내려받기", json.dumps(x, ensure_ascii=False, indent=1, default=str),
                   file_name=f"brief_{camp}_{rule}_{hh}_{provider}.json", mime="application/json")

sel = briefs[(briefs.campaign == camp) & (briefs.rule == rule)]
if len(sel):
    rows1, rows2 = [], []
    for r in sel.to_dict("records"):
        k = (r["campaign_type"], int(r["campaign"]), r["rule"], int(r["household_key"]))
        xr = to_result(r, k, profile(*k))
        a, c = brief_rows(xr)
        rows1.append(a); rows2.append(c)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame(rows1).to_excel(w, sheet_name="브리프", index=False)
        pd.DataFrame(rows2).to_excel(w, sheet_name="고객문구", index=False)
    d2.download_button(f"캠페인 {camp} · {rule} 전체 엑셀 ({len(sel)}건)", buf.getvalue(),
                       file_name=f"briefs_{camp}_{rule}_{provider}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
