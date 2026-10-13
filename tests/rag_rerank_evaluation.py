"""실제 RRF 후보의 리랭크 전후만 평가하고 Markdown 보고서 하나를 저장합니다."""
import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from src.const.config import RAG_EVALUATION_CASES_PATH, RAG_RERANK_REPORT_PATH
from src.const.model import OPENAI_LLM_MODEL, RERANK_MODEL, create_llm, create_rerank_model
from src.multi_query import create_queries
from src.prompt.multi_query_prompt import MULTI_QUERY_PROMPT
from src.prompt.query_rewriter_prompt import QUERY_REWRITE_PROMPT
from src.query_rewriter import query_rewriter
from src.rerank import rerank
from src.retrieval_evaluation import chunk_key, evaluate_ranking
from src.rrf import reciprocal_rank_fusion
from src.vector_db.bootstrap import create_team_retriever
from src.vectorstore import VectorSearchHit


# 데이터·후보·프롬프트를 동일한 규칙으로 해시합니다.
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode('utf-8')).hexdigest()


# 실제 DB의 본문·벡터 버전을 읽기 전용으로 확인합니다.
def snapshot_collection(store):
    records, offset = [], None
    while True:
        points, offset = store.client.scroll(collection_name=store.collection_name,
                                            limit=256, offset=offset,
                                            with_payload=True, with_vectors=True)
        for point in points:
            payload = dict(point.payload or {})
            if not isinstance(payload.get('text'), str) or not payload['text'].strip():
                raise ValueError(f'실제 본문이 없는 Point입니다: {point.id}')
            records.append({'point_id': str(point.id), 'payload': payload,
                            'vector_sha256': digest(point.vector)})
        if offset is None:
            break
    if not records:
        raise ValueError('Qdrant가 비어 있습니다. 더미 데이터로 대체하지 않습니다.')
    records.sort(key=lambda record: record['point_id'])
    return {'collection': store.collection_name, 'sha256': digest(records), 'records': records}


# 검수 상태는 API 호출 전에, 실제 정답·버전·모델은 DB 조회 후 검증합니다.
def validate_dataset(dataset, snapshot=None, embedding=None):
    if not isinstance(dataset, dict) or dataset.get('schema_version') != 1:
        raise ValueError('근거가 포함된 검수 데이터가 필요합니다.')
    if snapshot is not None and (dataset.get('collection') != snapshot['collection']
                                 or dataset.get('snapshot_sha256') != snapshot['sha256']):
        raise ValueError('컬렉션/data version이 검수 시점과 다릅니다.')
    cases = dataset.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('평가 질문이 없습니다.')
    seen = set()
    for case in cases:
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('case_id는 중복 없는 문자열이어야 합니다.')
        seen.add(case_id)
        if not isinstance(case.get('query'), str) or not case['query'].strip():
            raise ValueError(f'{case_id}: 질문이 비어 있습니다.')
        if case.get('review_status') != 'reviewed' or not isinstance(case.get('reviewed_by'), str) or not case['reviewed_by'].strip():
            raise ValueError(f'{case_id}: 검수 완료 라벨이 필요합니다. AI/사람 검수 여부와 검수자를 기록하세요.')
        if not isinstance(case.get('relevant_ids'), list):
            raise ValueError(f'{case_id}: 정답 ID 목록이 필요합니다.')
        evaluate_ranking([], case['relevant_ids'], k=10)
    if snapshot is not None:
        if dataset.get('embedding') != embedding:
            raise ValueError('검수 시점과 임베딩 설정이 다릅니다.')
        indexed = {}
        for record in snapshot['records']:
            payload = record['payload']
            if any(payload.get('embedding_' + key) != value for key, value in embedding.items()):
                raise ValueError(f'저장된 임베딩 모델·설정이 다릅니다: {record["point_id"]}')
            key = f"{payload.get('document_id')}::{payload.get('chunk_id')}"
            if key in indexed:
                raise ValueError(f'중복 청크 ID입니다: {key}')
            indexed[key] = indexed[record['point_id']] = payload
        for case in cases:
            evidence = {item['id']: item for item in case.get('evidence', [])}
            filters = case.get('filters') or {}
            if not isinstance(filters, dict):
                raise ValueError('filters는 객체여야 합니다.')
            for key in case['relevant_ids']:
                payload, item = indexed.get(key), evidence.get(key, {})
                if (payload is None or item.get('review_status') != 'reviewed'
                        or not isinstance(item.get('rationale'), str) or not item['rationale'].strip()
                        or item.get('text') != payload['text']):
                    raise ValueError(f'실제 정답·검수 근거가 불일치합니다: {key}')
                if payload.get('split_from_chunk_id') or any(payload.get(field) != value for field, value in filters.items()):
                    raise ValueError(f'필터/논리 청크 매핑을 재검수해야 합니다: {key}')
    return cases


# 기존 지표 계산 함수를 재사용하여 @5와 @10을 계산합니다.
def metrics(ids, relevant):
    at5, at10 = evaluate_ranking(ids, relevant, k=5), evaluate_ranking(ids, relevant, k=10)
    return {'Recall@5': at5['recall'], 'Recall@10': at10['recall'], 'Precision@5': at5['precision'],
            'MRR@10': at10['rr'], 'nDCG@10': at10['ndcg'], 'Hit Rate@5': at5['hit']}


# 비교·시간·개선/악화 사례와 재현에 필요한 상세 정보를 Markdown 하나에 담습니다.
def write_report(response, path):
    lines = ['# 실제 리랭크 전후 평가', '',
             f"컬렉션: `{response['settings']['collection']}`, 모델: `{response['settings']['rerank_model']}`",
             f"후보 {response['candidate_k']}개, 최종 {response['final_k']}개. 실제 질문 {len(response['cases'])}개.",
             f"라벨 검수: `{response.get('label_review', {})}`",
             f"초기화·로딩·준비 실행(ms): `{response['initialization_ms']}`", '']
    if response['summary']:
        columns = list(response['summary']['rrf'])
        lines += ['| 조건 | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
        for condition, values in response['summary'].items():
            lines.append('| ' + condition + ' | ' + ' | '.join(f'{values[key]:.4f}' for key in columns) + ' |')
        lines += ['', '## RRF 대비 리랭크 변화', '', '| 지표 | 절대 변화 | 상대 변화(%) |', '|---|---:|---:|']
        for key in response['cases'][0]['rrf']['metrics']:
            before, after = response['summary']['rrf'][key], response['summary']['rerank'][key]
            relative = f'{(after / before - 1) * 100:+.2f}' if before else 'N/A (baseline=0)'
            lines.append(f'| {key} | {after - before:+.4f} | {relative} |')
    else:
        lines += ['실제 실행·시간 확인만 수행했습니다. 정답 라벨을 사용하거나 품질 점수를 생성하지 않았습니다.']
    lines += ['', '## 질문별 결과와 실패 관찰', '']
    failures = Counter()
    for case in response['cases']:
        if response['summary']:
            delta = case['rerank']['metrics']['nDCG@10'] - case['rrf']['metrics']['nDCG@10']
            category = '개선' if delta > 0 else '악화' if delta < 0 else '변화 없음'
            lines.append(f"- {case['case_id']}: {case['query']} / {category}, ΔnDCG@10={delta:+.4f}")
            if case['candidate_recall'] < 1:
                failures['정답 일부가 후보에 없음; 리랭크로 복구 불가'] += 1
            elif case['rerank']['metrics']['Recall@10'] < 1:
                failures['후보의 정답이 리랭크 상위 10위 밖'] += 1
        lines += [f"- 전: `{case['rrf']['ids']}` / 후: `{case['rerank']['ids']}`",
                  f"  순수 처리 {case['processing_ms']:.3f}ms, predict {case['inference_ms']:.3f}ms"]
    lines += ['', f'실패 관찰: {dict(failures)}', '',
              '같은 실제 후보·질의·라벨에서 리랭크만 ON/OFF 비교합니다. 실행 성공과 품질 개선은 별개입니다.',
              '모델 로딩·준비 실행은 제외합니다. predict는 순수 처리 시간의 일부이므로 합산하지 않습니다.',
              'P95는 질문별 1회 측정값의 선형 보간입니다. 앞단 시간은 두 조건이 공유하며 재사용 질의의 생성 시간은 최초 API 측정값입니다.',
              '높은 기존 순위·후보 누락·불완전한 라벨은 효과를 제한할 수 있으나 원인을 수치만으로 확정하지 않습니다.',
              '', '## 재현용 실제 설정·질의·후보·결과', '', '```json',
              json.dumps(response, ensure_ascii=False, indent=2), '```']
    response_text = '\n'.join(lines) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response_text, encoding='utf-8')
    return response_text


# 기존 검색·RRF에서 후보를 한 번 얻고, 로드된 실제 모델로 리랭크 전후만 평가합니다.
def evaluate_rerank(dataset, candidate_k, final_k, output, smoke=False, query_report=None):
    cases = ([{'case_id': 'smoke', 'query': dataset['cases'][0]['query'], 'filters': None}]
             if smoke else validate_dataset(dataset))
    started = perf_counter()
    searcher = create_team_retriever()
    initialization = {'retriever_ms': (perf_counter() - started) * 1000}
    try:
        settings = {'collection': searcher.vector_store.settings.collection.name,
                    'embedding': asdict(searcher.vector_store.settings.embedding),
                    'llm_model': OPENAI_LLM_MODEL, 'rerank_model': RERANK_MODEL, 'temperature': 0,
                    'search_mode': searcher.vector_store.settings.collection.search_mode,
                    'hybrid_prefetch_k': max(candidate_k, searcher.vector_store.settings.collection.hybrid_prefetch_limit),
                    'prompt_sha256': digest([QUERY_REWRITE_PROMPT, MULTI_QUERY_PROMPT])}
        snapshot = None if smoke else snapshot_collection(searcher.vector_store)
        if not smoke:
            validate_dataset(dataset, snapshot, settings['embedding'])
        context = {'settings': settings, 'dataset_sha256': digest(dataset),
                   'snapshot_sha256': None if smoke else snapshot['sha256']}
        cached = {}
        if query_report:
            # 재현 정보도 Markdown에서 읽으므로 별도 JSON 캐시를 만들지 않습니다.
            previous = json.loads(query_report.read_text(encoding='utf-8').split('```json\n', 1)[1].rsplit('\n```', 1)[0])
            if previous['context'] != context:
                raise ValueError('질의 보고서의 데이터 버전·설정이 다릅니다.')
            cached = {case['case_id']: case for case in previous['cases']}
        llm, model = create_llm(temperature=0), None
        if searcher.sparse_embedder is not None:
            started = perf_counter()
            searcher.sparse_embedder.embed_query(cases[0]['query'])
            initialization['sparse_load_and_first_embed_ms'] = (perf_counter() - started) * 1000
        rows = []
        for case in cases:
            generated = cached.get(case['case_id'])
            if generated:
                if generated['query'] != case['query']:
                    raise ValueError('재사용 질문이 다릅니다.')
                rewritten, queries, generation_ms = generated['rewritten'], generated['queries'], generated['generation_ms']
                if (not isinstance(queries, list) or not 1 <= len(queries) <= 3 or queries[0] != rewritten
                        or any(not isinstance(query, str) or not query.strip() for query in queries)
                        or not np.isfinite(generation_ms) or generation_ms < 0):
                    raise ValueError('재사용 질의·시간이 잘못되었습니다.')
            else:
                started = perf_counter()
                rewritten = query_rewriter(case['query'], llm)
                queries = create_queries(rewritten, llm)
                generation_ms = (perf_counter() - started) * 1000
            started = perf_counter()
            rankings = {f'query_{index}': searcher.search_documents(query, top_k=candidate_k, filters=case.get('filters'))
                        for index, query in enumerate(queries)}
            candidates = list(reciprocal_rank_fusion(rankings, top_k=candidate_k))
            retrieval_rrf_ms = (perf_counter() - started) * 1000
            documents = [hit.to_document() for hit in candidates]
            pool_hash, timing = digest(documents), {}
            if model is None and documents:
                started = perf_counter()
                model = create_rerank_model()
                initialization['model_load_ms'] = (perf_counter() - started) * 1000
                started = perf_counter()
                rerank({'origin_query': case['query'], 'documents': documents[:1]}, top_k=1, model=model, verbose=False)
                initialization['warmup_ms'] = (perf_counter() - started) * 1000
            after = rerank({'origin_query': case['query'], 'documents': documents}, top_k=final_k,
                           model=model, timings=timing, verbose=False)['documents']
            original = {chunk_key(hit): (hit.point_id, hit.text) for hit in candidates}
            after_hits = [VectorSearchHit(doc['point_id'], doc['score'], doc['text'], doc['metadata']) for doc in after]
            after_ids = [chunk_key(hit) for hit in after_hits]
            if (digest(documents) != pool_hash or len(original) != len(candidates)
                    or len(set(after_ids)) != len(after_ids) or len(after) != min(final_k, len(candidates))
                    or any(original.get(chunk_key(hit)) != (hit.point_id, hit.text) for hit in after_hits)):
                raise ValueError('리랭크 전후 실제 후보 ID·본문이 불일치합니다.')
            if smoke and not candidates:
                raise ValueError('실제 후보가 없어 리랭크 실행을 확인하지 못했습니다.')
            ids = [chunk_key(hit) for hit in candidates]
            row = {'case_id': case['case_id'], 'query': case['query'], 'rewritten': rewritten, 'queries': queries,
                   'generation_ms': generation_ms, 'retrieval_rrf_ms': retrieval_rrf_ms,
                   'candidate_pool_sha256': pool_hash, 'candidate_documents': documents,
                   'processing_ms': timing['processing_seconds'] * 1000,
                   'inference_ms': timing['inference_seconds'] * 1000,
                   'rrf': {'ids': ids[:final_k]}, 'rerank': {'ids': after_ids}}
            if not smoke:
                row['candidate_recall'] = evaluate_ranking(ids, case['relevant_ids'], k=candidate_k)['recall']
                for condition in ('rrf', 'rerank'):
                    row[condition]['metrics'] = metrics(row[condition]['ids'], case['relevant_ids'])
            rows.append(row)
        if not smoke and snapshot_collection(searcher.vector_store)['sha256'] != snapshot['sha256']:
            raise ValueError('실험 중 DB 버전이 변경되어 보고서 생성을 중단합니다.')
        summary = {}
        if not smoke:
            for condition in ('rrf', 'rerank'):
                summary[condition] = {key: float(np.mean([row[condition]['metrics'][key] for row in rows]))
                                      for key in rows[0][condition]['metrics']}
                # 동일한 앞단 시간에 리랭크 처리 시간만 추가합니다.
                times = [row['generation_ms'] + row['retrieval_rrf_ms'] +
                         (row['processing_ms'] if condition == 'rerank' else 0) for row in rows]
                summary[condition].update(mean_latency_ms=float(np.mean(times)), p95_latency_ms=float(np.percentile(times, 95)))
            for name in ('processing_ms', 'inference_ms'):
                values = [row[name] for row in rows]
                summary['rerank']['mean_' + name] = float(np.mean(values))
                summary['rerank']['p95_' + name] = float(np.percentile(values, 95))
                summary['rrf']['mean_' + name] = summary['rrf']['p95_' + name] = 0.0
        response = {'evaluation_type': 'smoke_without_quality_scores' if smoke else 'rerank_only',
                    'context': context, 'settings': settings, 'candidate_k': candidate_k, 'final_k': final_k,
                    'initialization_ms': initialization, 'summary': summary, 'cases': rows,
                    'label_review': dataset.get('review', {})}
        report_text = write_report(response, output)
        print(report_text.split('## 재현용', 1)[0])
        print(f'보고서: {output}')
        return response
    finally:
        searcher.vector_store.client.close()


# 경로는 config 기본값을 사용하며 터미널 옵션으로 필요한 값만 변경합니다.
def main():
    parser = argparse.ArgumentParser(description='실제 동일 RRF 후보의 리랭크 전후 평가; Markdown만 저장')
    parser.add_argument('--evaluation-file', type=Path, default=RAG_EVALUATION_CASES_PATH)
    parser.add_argument('--output', type=Path, default=RAG_RERANK_REPORT_PATH)
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--final-k', type=int, default=10)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--query-report', type=Path, help='생성 질의를 재사용할 이전 Markdown 보고서')
    args = parser.parse_args()
    if not 10 <= args.final_k <= args.candidate_k <= 100:
        parser.error('10 <= final-k <= candidate-k <= 100이어야 합니다.')
    if args.output.suffix != '.md' or not args.output.name.startswith('rag_'):
        parser.error('출력은 rag_로 시작하는 Markdown 파일이어야 합니다.')
    dataset = json.loads(args.evaluation_file.read_text(encoding='utf-8-sig'))
    response = evaluate_rerank(dataset, args.candidate_k, args.final_k, args.output, args.smoke_test, args.query_report)
    return response


if __name__ == '__main__':
    main()
