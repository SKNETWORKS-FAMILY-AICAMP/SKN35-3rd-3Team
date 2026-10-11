"""소비자 자료의 역할·표·수식·출처를 보존한 청크 생성."""
import json
from pathlib import Path

if __package__:
    from .chunk_metadata import normalize_chunk, METADATA_VERSION
    from .preprocess_consumer_sources import OUTPUT_DIR as INPUT_DIR
    from .collect_consumer_sources import ROOT, digest, write_json
else:
    from chunk_metadata import normalize_chunk, METADATA_VERSION
    from preprocess_consumer_sources import OUTPUT_DIR as INPUT_DIR
    from collect_consumer_sources import ROOT, digest, write_json

OUTPUT_DIR=ROOT/'data/chunk/consumer_sources'


def search_passages(text, size=700, overlap=120):
    """검색 점수용 본문 창. 표 셀을 해석하지 않으며 원문 청크를 대체하지 않는다."""
    passages=[]
    start=0
    while start<len(text):
        end=min(start+size,len(text))
        if end<len(text):
            line=text.rfind('\n',start+size//2,end)
            if line>=0:
                end=line+1
        passages.append({'start':start,'end':end,'text':text[start:end]})
        if end==len(text):
            break
        start=max(start+1,end-overlap)
    return passages


def make_chunks(doc,size=1200,overlap=120):
    if type(size) is not int or type(overlap) is not int or size<=0 or not 0<=overlap<size:
        raise ValueError('유효하지 않은 청크 크기')
    if not doc['embedding_candidate'] or doc['needs_manual_review']:
        return []
    chunks=[]
    facts=next((u['text'] for u in doc['units'] if u['role']=='facts'),'')
    for source in doc['units']:
        text=source['text']
        start,index=0,0
        while start<len(text):
            end=len(text) if source.get('atomic') else min(start+size,len(text))
            if end<len(text):
                line=text.rfind('\n',start+size//2,end)
                if line>=0:
                    end=line+1
            context=(facts if source['role'] in ['party_claims','reasoning','decision'] else '')
            header=f"[{doc['title']} / {doc['document_type']} / {source['role']} / {source.get('source_locator',source['unit_id'])}]"
            if doc.get('terms_revision_date'):
                header+=f" [개정일: {doc['terms_revision_date']}]"
            caution='\n[원문 주의사항] '+ ' '.join(source.get('warnings',[])) if source.get('source_anomaly') else ''
            chunks.append({'schema_version':1,'chunk_id':f"{doc['source_id']}_{source['unit_id']}_{index:04d}",
                'source_id':doc['source_id'],'unit_id':source['unit_id'],'title':doc['title'],
                'document_type':doc['document_type'],'domain':source.get('domain',doc['domain']),
                'role':source['role'],'source_url':doc['source_url'],'raw_file':doc['raw_file'],
                'raw_sha256':doc['raw_sha256'],'effective_date':doc['effective_date'],
                'updated_date':doc['updated_date'],'decision_date':doc['decision_date'],
                'terms_revision_date':doc.get('terms_revision_date'),
                'version_role':doc.get('version_role'),
                'derived_pdf_file':doc.get('derived_pdf_file'),
                'derived_pdf_sha256':doc.get('derived_pdf_sha256'),
                'page_numbers':source.get('page_numbers'),
                'article_number':source.get('article_number'),
                'article_title':source.get('article_title'),
                'source_anomaly':source.get('source_anomaly'),
                'calculation_restricted_clause':source.get('calculation_restricted_clause'),
                'start':start,'end':end,'offset_basis':'unit.text','text':text[start:end],
                'source_locator':source.get('source_locator'),'parent_context':context,
                **({'appendix_number':source['appendix_number'], 'appendix_title':source['appendix_title'],
                    'source_offset_basis':source['offset_basis'], 'source_start':source['start'], 'source_end':source['end']}
                   if source['role']=='appendix_table' else {}),
                'context_header':header,'embedding_text':header+caution+('\n[사건개요 문맥]\n'+context if context else '')+'\n'+text[start:end],
                'atomic':source.get('atomic',False),'oversize_atomic_unit':source.get('atomic',False) and len(text)>size,
                'embedding_candidate':True,'warnings':doc['warnings']+source.get('warnings',[])})
            index+=1
            if end==len(text):
                break
            start=max(start+1,end-overlap)
    for chunk in chunks:
        chunk['search_passages']=search_passages(chunk['text'])
        # HWPX의 주소·병합 정보 JSON은 그대로 보존하고 검색에는 확인된 셀 문자만 추가 사용한다.
        marker='[표 셀: address는 행·열, span은 병합 범위]\n'
        cursor=0
        while marker in chunk['text'][cursor:]:
            start=chunk['text'].index(marker,cursor)+len(marker)
            cells,length=json.JSONDecoder().raw_decode(chunk['text'][start:])
            end=start+length
            if not isinstance(cells,list) or not all(isinstance(c,dict) and isinstance(c.get('text'),str) for c in cells):
                raise ValueError('검색 표현을 생성할 표 셀 형식 불일치')
            chunk['search_passages'].append({'start':start,'end':end,'text':chunk['text'][start:end],
                'search_text':'\n'.join(c['text'] for c in cells),
                'derivation':'table_cell_texts_in_source_order_no_merge_inference'})
            cursor=end
        chunk['search_representation_version']=1
        chunk['search_passages_usage']='본문 위치 기반 BM25 점수용. 셀·조건의 독립 근거가 아니며 반환·임베딩은 전체 원문 청크 사용.'
        chunk['evidence_role']='party_claim' if chunk['role']=='party_claims' else ('case_reasoning' if chunk['role']=='reasoning' else chunk['role'])
    return [normalize_chunk(c, doc) for c in chunks]


def run():
    report=json.loads((INPUT_DIR/'consumer_preprocessing_report.json').read_text(encoding='utf-8'))
    if report['failures']:
        raise ValueError('정제 실패를 먼저 확인하세요.')
    chunks=[]
    inputs=[]
    for entry in report['saved']:
        path=INPUT_DIR/entry['file']
        data=path.read_bytes()
        doc=json.loads(data)
        if digest((ROOT/doc['raw_file']).read_bytes())!=doc['raw_sha256']:
            raise ValueError('정제 후 원본 변경')
        if doc.get('derived_pdf_file') and digest((ROOT/doc['derived_pdf_file']).read_bytes())!=doc['derived_pdf_sha256']:
            raise ValueError('정제 후 변환 PDF 변경')
        chunks.extend(make_chunks(doc))
        inputs.append({'file':entry['file'],'processed_sha256':digest(data)})
    if len({c['chunk_id'] for c in chunks})!=len(chunks):
        raise ValueError('청크 ID 중복')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in chunks)
    path=OUTPUT_DIR/'chunks.jsonl'
    temporary=path.with_suffix('.jsonl.tmp')
    temporary.write_text(content,encoding='utf-8')
    temporary.replace(path)
    result={'metadata_version':METADATA_VERSION,'document_count':len({c['source_id'] for c in chunks}),'chunk_count':len(chunks),
            'size':1200,'overlap':120,'chunks_sha256':digest(path.read_bytes()),'inputs':inputs,
            'oversize_atomic_count':sum(c['oversize_atomic_unit'] for c in chunks),
            'usage':'embedding_text 사용. 표·약관 조문은 원자 단위 유지. 실제 임베딩 전 모델 토큰 제한 확인.',
            'excluded':report['excluded'],'duplicate_aliases':report['duplicate_aliases']}
    write_json(OUTPUT_DIR/'consumer_chunking_report.json',result)
    print(f"문서 {result['document_count']} / 청크 {len(chunks)}")
    return result


if __name__=='__main__':
    run()
