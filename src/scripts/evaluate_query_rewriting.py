"""수동 리라이팅의 효과를 실제 청크에서 비교하는 소규모 탐색 평가.

LLM/API는 호출하지 않습니다. 사람이 작성한 검색 표현의 효과만 측정합니다.
대표 근거 한 개만 라벨링했으므로 다른 관련 문서가 오답으로 집계될 수 있습니다.
따라서 Recall/Precision은 이 제한된 라벨 기준이며 공식 검색 성적이 아닙니다.
"""
import json
from pathlib import Path
from time import perf_counter

from src.scripts.try_real_search import load_documents
from .try_real_search import load_documents
from src.retriever import BM25Retriever
from src.retrieval_evaluation import chunk_key, evaluate_ranking


# 검색 결과를 보고 표현을 반복 조정하지 않고 고정한 탐색용 질문 쌍입니다.
# 질문의 사실관계는 유지하며 동의어를 검색 표현으로 바꿉니다.
# 법 조문 이름을 그대로 사용하는 사례도 있어 실제 LLM보다 유리할 수 있습니다.
CASES = [
    ("집 계약이 말없이 연장됐는데 세입자가 중간에 나갈 수 있어?", "주택 임대차 묵시적 갱신 계약 해지 임차인 통지 효력", "001248", "6_2"),
    ("집 계약 끝났는데 보증금 못 받았어. 법원에 등기를 신청할 수 있어?", "임대차 종료 보증금 미반환 임차권등기명령 신청", "001248", "3_3"),
    ("집 계약 연장해달라는 요구는 몇 번 할 수 있어?", "주택 임대차 계약갱신 요구 행사 횟수", "001248", "6_3"),
    ("집주인이 월세를 올려달라는데 인상 한도가 있어?", "주택 임대차 차임 증액청구 상한", "001248", "7_0"),
    ("집주인은 빌려준 집을 사용할 수 있는 상태로 유지해야 해?", "임대인 목적물 사용 수익 상태 유지 의무", "001706", "623_0"),
    ("내 잘못 없이 빌린 집 일부를 못 쓰게 됐어. 월세를 줄일 수 있어?", "임차물 일부 사용 수익 불가 임차인 과실 없이 차임 감액청구", "001706", "627_0"),
    ("건물 세입자가 월세를 밀리면 집주인이 계약을 끝낼 수 있어?", "건물 임대차 차임연체 임대인 계약 해지", "001706", "640_0"),
    ("인터넷으로 산 물건 취소는 언제까지 가능해?", "통신판매 재화 구매 계약 청약철회 기간", "009318", "17_0"),
    ("주택 임대차 묵시적 갱신 계약 해지 통지 효력", "주택 임대차 묵시적 갱신 계약 해지 통지 효력", "001248", "6_2"),
    ("임대인 목적물 사용 수익 상태 유지 의무", "임대인 목적물 사용 수익 상태 유지 의무", "001706", "623_0"),
]


def main():
    documents = load_documents()
    searcher = BM25Retriever(documents)
    rows = []
    for original, rewritten, law, article in CASES:
        # 조문 번호만으로는 다른 법률과 충돌하므로 법률 ID까지 함께 확인합니다.
        targets = [d for d in documents if d.point_id.startswith(f"law:{law}_")
                   and f"_art_{article}_chunk_0000" in d.point_id]
        if len(targets) != 1:
            raise ValueError(f"대표 근거를 하나로 특정할 수 없습니다: {law}/{article}")
        target = targets[0]
        row = {"original": original, "rewritten": rewritten,
               "control": original == rewritten,
               "relevant_ids": [chunk_key(target)], "evidence": target.text}
        for name, query in (("original_result", original), ("rewritten_result", rewritten)):
            start = perf_counter()
            hits = searcher.search_documents(query, top_k=5)
            ids = [chunk_key(hit) for hit in hits]
            row[name] = {**evaluate_ranking(ids, row["relevant_ids"], k=5),
                         "seconds": perf_counter() - start,
                         "retrieved_ids": ids,
                         "retrieved_excerpts": [hit.text[:250] for hit in hits]}
        rows.append(row)
    # 전체 평균과 실제로 표현을 바꾼 8개 질문의 평균을 따로 제공합니다.
    summary = {}
    for group, selected in (("all", rows), ("rewritten_only", [r for r in rows if not r["control"]])):
        summary[group] = {name: {metric: sum(r[name][metric] for r in selected) / len(selected)
                                for metric in ("recall", "precision", "hit", "rr", "seconds")}
                          for name in ("original_result", "rewritten_result")}
    report = {"method": "BM25, manual rewriting, no LLM", "document_count": len(documents),
              "k": 5, "limitations": "Representative evidence labels only; author-created exploratory cases, not a held-out benchmark. Rewrite generation latency is not measured.",
              "summary": summary, "cases": rows}
    output = Path("docs/query-rewriting-evaluation.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        print(row["original"], "RR:", row["original_result"]["rr"], "->", row["rewritten_result"]["rr"])
    print(f"상세 결과: {output}")


if __name__ == "__main__":
    main()
