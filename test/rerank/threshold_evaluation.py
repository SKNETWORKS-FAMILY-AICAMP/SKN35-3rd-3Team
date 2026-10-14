"""STEP 4/5 only. Standard library; never calls retrieval or model inference."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def metrics(pairs):
    counts = dict.fromkeys(("TP", "TN", "FP", "FN"), 0)
    for truth, prediction in pairs:
        counts[("T" if truth == prediction else "F") + ("P" if prediction else "N")] += 1
    tp, tn, fp, fn = (counts[k] for k in ("TP", "TN", "FP", "FN"))
    def div(a, b):
        return a / b if b else None
    return {**counts, "n": tp + tn + fp + fn,
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn),
            "accuracy": div(tp + tn, tp + tn + fp + fn),
            "mcc": div(tp * tn - fp * fn, math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
            "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
            "error_rate": div(fp + fn, tp + tn + fp + fn)}


def load_inputs(cases_path, scores_path):
    dataset = read_json(cases_path)
    case_list = dataset["cases"] if isinstance(dataset, dict) else dataset
    cases = {c["case_id"]: c for c in case_list}
    if len(cases) != len(case_list):
        raise ValueError("Duplicate case_id in dataset")
    groups = defaultdict(set)
    queries = defaultdict(set)
    for c in case_list:
        if c.get("evaluation_split") not in ("development", "validation"):
            raise ValueError("Missing/invalid dataset evaluation_split")
        groups[c.get("question_group_id", c["case_id"])].add(c["evaluation_split"])
        if c.get("query"):
            queries[" ".join(c["query"].split()).casefold()].add(c["evaluation_split"])
        source = c.get("source_case_id")
        if source in cases and cases[source]["evaluation_split"] != c["evaluation_split"]:
            raise ValueError("Source question crosses splits: " + c["case_id"])
    if any(len(s) > 1 for s in [*groups.values(), *queries.values()]):
        raise ValueError("Question group or identical query crosses splits")
    payload = read_json(scores_path)
    if isinstance(payload, list):
        rows = payload
    else:
        rows = next((payload[k] for k in ("documents", "scores", "results", "records")
                     if isinstance(payload.get(k), list)), None)
    if not rows:
        raise ValueError("Expected nonempty flat score list or documents/scores/results/records list")
    seen = set()
    required = {"case_id", "document_id", "relevance_grade", "rerank_score",
                "evaluation_split", "candidate_k", "rrf_rank", "rerank_rank"}
    for r in rows:
        if required - r.keys():
            raise ValueError(f"Missing score fields: {required - r.keys()}")
        c = cases.get(r["case_id"])
        if c is None or c["evaluation_split"] != r["evaluation_split"]:
            raise ValueError("Unknown case or split mismatch")
        if type(r["relevance_grade"]) is not int or r["relevance_grade"] not in (0, 1, 2):
            raise ValueError("Invalid relevance grade")
        if isinstance(r["rerank_score"], bool) or not isinstance(r["rerank_score"], (int, float)) or not math.isfinite(r["rerank_score"]):
            raise ValueError("Score must be finite numeric raw score")
        if type(r["candidate_k"]) is not int or r["candidate_k"] < 1:
            raise ValueError("candidate_k must be positive integer")
        for rank in ("rrf_rank", "rerank_rank"):
            if type(r[rank]) is not int or not 1 <= r[rank] <= r["candidate_k"]:
                raise ValueError("Invalid rank")
        key = (r["candidate_k"], r["case_id"], r["document_id"])
        if key in seen:
            raise ValueError("Duplicate scored document")
        seen.add(key)
        labels = {x["document_id"]: x["relevance_grade"] for x in c.get("relevance_labels", [])}
        if r["document_id"] in labels and labels[r["document_id"]] != r["relevance_grade"]:
            raise ValueError("Score label disagrees with dataset")
    for k in {r["candidate_k"] for r in rows}:
        subset = [r for r in rows if r["candidate_k"] == k]
        if {r["case_id"] for r in subset} != set(cases):
            raise ValueError(f"candidate_k={k}: missing cases; cannot silently count missing scores as blocked")
        for cid in cases:
            rs = [r for r in subset if r["case_id"] == cid]
            if len(rs) > k or any(len({r[rank] for r in rs}) != len(rs) for rank in ("rrf_rank", "rerank_rank")):
                raise ValueError("Duplicate ranks or too many candidates")
    return dataset, cases, rows


def evaluate(rows, cases, threshold, direction):
    def passed(r):
        return r["rerank_score"] >= threshold if direction == "higher" else r["rerank_score"] <= threshold
    document_errors, question_errors = [], []
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
        if (r["relevance_grade"] > 0) != passed(r):
            document_errors.append({**r, "error": "FP" if passed(r) else "FN"})
    pool_pairs, gold_pairs = [], []
    strata = defaultdict(list)
    for cid, c in cases.items():
        rs = by_case[cid]
        pool_truth = any(r["relevance_grade"] > 0 for r in rs)
        prediction = any(passed(r) for r in rs)
        # Explicit uncertainty is preserved; never converted to unanswerable.
        answerability = c.get("answerability")
        gold = (True if answerability == "answerable" else False if answerability == "unanswerable"
                else None if answerability == "uncertain" else
                bool(c.get("relevant_ids") or any(x["relevance_grade"] > 0 for x in c.get("relevance_labels", []))))
        if gold is False and pool_truth:
            raise ValueError("Unanswerable case has relevant candidates: " + cid)
        category = "unknown_answerability" if gold is None else "no_gold_answer" if not gold else "gold_in_pool" if pool_truth else "gold_missing_from_pool"
        pool_pairs.append((pool_truth, prediction))
        strata[category].append((pool_truth, prediction))
        if gold is not None:
            gold_pairs.append((gold, prediction))
        if prediction != pool_truth or (gold is not None and prediction != gold):
            question_errors.append({"case_id": cid, "query": c.get("query"), "category": category,
                                    "answer_proceed": prediction, "pool_truth": pool_truth, "gold_truth": gold,
                                    "pool_error": None if prediction == pool_truth else "FP" if prediction else "FN",
                                    "gold_error": None if gold is None or prediction == gold else "FP" if prediction else "FN",
                                    "passed_document_ids": [r["document_id"] for r in rs if passed(r)],
                                    "relevant_document_ids": [r["document_id"] for r in rs if r["relevance_grade"] > 0]})
    return {"document": metrics([(r["relevance_grade"] > 0, passed(r)) for r in rows]),
            "question_pool": metrics(pool_pairs), "question_gold_known_only": metrics(gold_pairs),
            "question_strata": {s: metrics(p) for s, p in strata.items()},
            "document_errors": document_errors, "question_errors": question_errors}


def selection_key(result):
    # Predeclared rule, applied independently for each candidate_k.
    return (result["document"]["mcc"] if result["document"]["mcc"] is not None else -2,
            result["document"]["f1"] or 0, -result["question_pool"]["error_rate"],
            -(result["document"]["fpr"] or 0))


def run(cases_path, scores_path, output):
    dataset, cases, rows = load_inputs(cases_path, scores_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    results, comparisons = {}, []
    for k in sorted({r["candidate_k"] for r in rows}):
        dev = [r for r in rows if r["candidate_k"] == k and r["evaluation_split"] == "development"]
        val = [r for r in rows if r["candidate_k"] == k and r["evaluation_split"] == "validation"]
        dc = {i:c for i,c in cases.items() if c["evaluation_split"] == "development"}
        vc = {i:c for i,c in cases.items() if c["evaluation_split"] == "validation"}
        if not dev or not val or len({r["relevance_grade"] > 0 for r in dev}) != 2:
            raise ValueError("Both splits and both development document classes required")
        scores = sorted({r["rerank_score"] for r in dev})
        best, best_key, direction_best = None, None, {}
        for direction in ("higher", "lower"):
            thresholds = scores + [math.nextafter(scores[-1], math.inf) if direction == "higher" else math.nextafter(scores[0], -math.inf)]
            for threshold in sorted(thresholds):
                if not math.isfinite(threshold):
                    raise ValueError("Cannot construct finite boundary threshold")
                result = evaluate(dev, dc, threshold, direction)
                key = selection_key(result)
                row = {"candidate_k": k, "direction": direction, "threshold": threshold}
                for unit in ("document", "question_pool", "question_gold_known_only"):
                    row.update({unit + "_" + m: v for m,v in result[unit].items()})
                comparisons.append(row)
                if direction not in direction_best or key > direction_best[direction]:
                    direction_best[direction] = key
                if best_key is None or key > best_key:
                    best_key, best = key, (threshold, direction, result)
        threshold, direction, development = best
        # No validation scores or labels participate in threshold/direction selection.
        validation = evaluate(val, vc, threshold, direction)
        results[str(k)] = {"threshold": threshold, "direction": direction,
                           "selection_split": "development", "validation_threshold_frozen": True,
                           "development_raw_score_range": [min(scores), max(scores)],
                           "development_direction_comparison": direction_best,
                           "development": development, "validation": validation}
    with (out / "threshold_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    # Separate document classification from question routing for both STEP 4/5.
    split_rows = []
    for k, selected in results.items():
        for split, step in (("development", 4), ("validation", 5)):
            for unit in ("document", "question_pool", "question_gold_known_only"):
                split_rows.append({"step": step, "split": split, "candidate_k": int(k),
                                   "threshold": selected["threshold"], "direction": selected["direction"],
                                   "unit": unit, **selected[split][unit]})
    with (out / "split_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    report = {"selection_rule": "Development document MCC max; ties: document F1 max, question_pool error min, document FPR min. Exact ties prefer higher direction then smallest threshold. Separate threshold per candidate_k.",
              "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (cases_path, scores_path)},
              "leakage_audit": "Checked case splits, source/group/exact-query separation; validation never used in selection. Prior model-training/manual tuning exposure cannot be verified from these files.",
              "label_policy": dataset.get("label_policy") if isinstance(dataset, dict) else None,
              "results": results}
    (out / "evaluation_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# RAG STEP 4·5 평가", "", report["selection_rule"], "", report["leakage_audit"], "",
             "미정의 지표는 null/빈 CSV 셀. Grade 0=무관, 1·2=관련. 점수는 원본 그대로 사용.",
             "question_pool은 후보 풀 관련 문서 존재 여부가 정답이며 전체 질문을 분모로 사용한다. question_gold_known_only는 데이터셋 answerability 기준이며 uncertain을 제외한 분모를 명시한다.",
             "정답은 있으나 후보 풀에 없는 질문은 별도 strata로 보고한다. 관련 문서 없이 무관 문서만 통과한 답변도 gold 기준 TP가 될 수 있으므로 두 질문 지표와 문서 오류를 함께 읽어야 한다.", ""]
    for k, r in results.items():
        lines += [f"## candidate_k={k}", f"선택 임계값: {r['threshold']!r}; 통과 방향: {r['direction']}; Dev 원점수 범위: {r['development_raw_score_range']}", ""]
        for split in ("development", "validation"):
            lines += [f"### STEP {4 if split == 'development' else 5}: {split}", "",
                      "| 평가 단위 | n | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | 오분기율/오류율 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for unit in ("document", "question_pool", "question_gold_known_only"):
                values = r[split][unit]
                fields = ('n', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'fpr', 'fnr', 'error_rate')
                cells = ['N/A' if values[m] is None else str(values[m]) if m in fields[:5] else f'{values[m]:.6f}' for m in fields]
                lines.append('| ' + unit + ' | ' + ' | '.join(cells) + ' |')
            lines.append("")
            for unit in ("document", "question_pool", "question_gold_known_only", "question_strata"):
                lines.append(f"- {unit}: {json.dumps(r[split][unit], ensure_ascii=False)}")
            lines.append(f"문서 오류 {len(r[split]['document_errors'])}건, 질문 오류 {len(r[split]['question_errors'])}건. 전체 사례 및 통과/관련 문서 ID는 evaluation_results.json 참조.")
            errors = r[split]['question_errors']
            for category in ('no_gold_answer', 'gold_missing_from_pool', 'gold_in_pool', 'unknown_answerability'):
                selected = [e for e in errors if e['category'] == category]
                if selected:
                    lines.append(f"- 질문 실패 유형 {category}: {len(selected)}건; 예시 case_id: {', '.join(e['case_id'] for e in selected[:5])}")
            for error in ('FP', 'FN'):
                selected = [e for e in r[split]['document_errors'] if e['error'] == error]
                if selected:
                    example = selected[0]
                    lines.append(f"- 문서 {error} {len(selected)}건; 예시 {example['case_id']} / {example['document_id']} / grade={example['relevance_grade']} / score={example['rerank_score']}. " + ('무관 문서가 임계값을 통과함.' if error == 'FP' else '관련 문서가 임계값에서 차단됨.'))
        lines.append("Dev → Validation 지표 차이(Validation - Dev):")
        for unit in ("document", "question_pool", "question_gold_known_only"):
            diff = {m: (r['validation'][unit][m] - r['development'][unit][m]) if r['validation'][unit][m] is not None and r['development'][unit][m] is not None else None for m in ('f1', 'mcc', 'fpr', 'fnr', 'error_rate')}
            lines.append(f"- {unit}: {json.dumps(diff)}")
        n = r["validation"]["question_pool"]["n"]
        lines += [f"Validation 질문 {n}개: 질문 한 개가 오분기율을 {1/n:.1%}p 바꾼다. 소표본 결과를 일반화하지 않는다.", ""]
    lines += ["라벨은 데이터셋의 검토 수준에 종속된다. 문서 공유 자체는 누수로 단정하지 않으며, 의미적 중복 및 이전 Validation 노출 여부는 별도 검토가 필요하다."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    docs = Path(__file__).resolve().parent / "docs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=root / "rag_evaluation_cases_list.json")
    parser.add_argument("--scores", default=docs / "rerank_scores.json")
    parser.add_argument("--output", default=docs)
    args = parser.parse_args()
    if not Path(args.scores).is_file():
        parser.exit(2, f"WAITING: score file not found: {args.scores}. No real evaluation performed.\n")
    try:
        run(args.cases, args.scores, args.output)
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"INPUT ERROR: {exc}\n")
    print(f"STEP 4/5 results written to {args.output}")


if __name__ == "__main__":
    main()
