"""필라테스 사례의 계약·해지 시점별 공식 법령 대조. 과거 원문은 기본 검색에서 제외.

uv run python -m src.scripts.review_pilates_law_versions --collect
uv run python -m src.scripts.review_pilates_law_versions
두 번째 명령은 네트워크 없이 수집본으로 비교 기록을 재생성한다.
"""
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import sys
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    CHUNK_LAWS_DATA_PATH,
    PROCESSED_CONSUMER_SOURCES_DATA_PATH,
    PROJECT_PATH,
    RAW_LAWS_DATA_PATH,
    RAW_LAW_HISTORY_DATA_PATH,
)

ROOT = PROJECT_PATH

if __package__:
    from .collect_consumer_sources import HTMLTree, write_json
else:
    from collect_consumer_sources import HTMLTree, write_json

RAW = RAW_LAW_HISTORY_DATA_PATH
REPORT = PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'pilates_law_version_comparison_pmh.json'

@dataclass(frozen=True)
class ReviewPaths:
    """시점 대조의 원본·정제·청크·보고서 위치. 기본값은 config.py 경로다."""
    project_root: Path = ROOT
    history_dir: Path = RAW
    current_laws_dir: Path = RAW_LAWS_DATA_PATH
    consumer_dir: Path = PROCESSED_CONSUMER_SOURCES_DATA_PATH
    law_chunk_path: Path = CHUNK_LAWS_DATA_PATH / "chunks.jsonl"
    report_path: Path = REPORT

    def __post_init__(self):
        for name in ("project_root", "history_dir", "current_laws_dir", "consumer_dir", "law_chunk_path", "report_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        for name in ("history_dir", "current_laws_dir", "consumer_dir", "law_chunk_path", "report_path"):
            if not getattr(self, name).is_relative_to(self.project_root):
                raise ValueError("출처 기록의 경로는 project_root 안에 있어야 합니다.")


LAW = '방문판매 등에 관한 법률'
DECREE = LAW+' 시행령'
RULE = LAW+' 시행규칙'
TERMS = '약관의 규제에 관한 법률'
VERSIONS = {
    '237455': (LAW, '20221208'), '249019': (LAW, '20230321'), '252407': (LAW, '20230711'),
    '246957': (DECREE, '20221227'), '248655': (DECREE, '20230307'),
    '223955': (RULE, '20201204'), '225063': (TERMS, '20211230'), '260021': (TERMS, '20240807'),
}
HISTORY_IDS = {LAW:'000354', DECREE:'003572', RULE:'007196', TERMS:'adhesion_terms'}
CASE_EVENTS = {
    'kca_case_1003739114': [('contract','2023-04','month','case_facts','2023. 4.'),
                         ('termination','2023-07-14','day','case_reasoning','2023. 7. 14.')],
    'kca_case_1003640779': [('contract','2023-02-06','day','case_facts','2023. 2. 6.'),
                         ('termination_request','2023-05-13','day','case_facts','2023. 5. 13.')],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked(relative, expected=None, *, project_root=ROOT):
    project_root = Path(project_root).resolve()
    path=(project_root/relative).resolve()
    if not path.is_relative_to(project_root) or not path.is_file():
        raise ValueError('대조 입력 경로 오류/누락')
    data=path.read_bytes()
    if expected is not None and digest(data)!=expected:
        raise ValueError('대조 입력 지문 불일치: '+str(relative))
    return data


def history_rows(text, name):
    rows=[]
    tree=HTMLTree(text)
    for node in tree.root.find(lambda n:n.tag=='input'):
        if (node.attrs.get('id') or node.attrs.get('name') or '').lower()=='oc' and node.attrs.get('value') not in {'','REDACTED'}:
            raise ValueError('연혁 입력의 인증값 제거 미완료')
    for tr in tree.root.find(lambda n:n.tag=='tr'):
        cells=[n for n in tr.children if getattr(n,'tag',None)=='td']
        if len(cells)<9:
            continue
        values=[' '.join(n.text().split()) for n in cells]
        if values[1]!=name:
            continue
        links=tr.find(lambda n:n.tag=='a')
        ids={m.group(1) for n in links for m in re.finditer(r'(?:MST=|lsiSeq=)(\d+)',n.attrs.get('href',''))}
        if len(ids)!=1:
            raise ValueError('연혁 버전 링크 불명확')
        rows.append({'master_id':ids.pop(),'name':name,
                     'effective_date':datetime.strptime(values[7],'%Y.%m.%d').strftime('%Y%m%d'),
                     'promulgation_date':datetime.strptime(values[6],'%Y.%m.%d').strftime('%Y%m%d'),'number':values[5]})
    if not rows or len(rows)!=len({r['master_id'] for r in rows}):
        raise ValueError('정확한 법령명 연혁 목록 누락/중복')
    return rows


def select_at(rows, date):
    candidates=[r for r in rows if r['effective_date']<=date.replace('-','')]
    if not candidates:
        raise ValueError('해당 시점 이전 법령 없음')
    latest=max(r['effective_date'] for r in candidates)
    selected=[r for r in candidates if r['effective_date']==latest]
    if len(selected)!=1:
        raise ValueError('동일 시행일 버전 선택 모호')
    return selected[0]


def validate_body(text, name, effective):
    root=ET.fromstring(text)
    if root.tag!='법령' or root.findtext('기본정보/법령명_한글')!=name or root.findtext('기본정보/시행일자')!=effective:
        raise ValueError('요청 버전과 본문 법령명/시행일 불일치')
    if not root.findall('./조문/조문단위'):
        raise ValueError('조문 없는 응답')
    return root


def sanitize_response(text, oc):
    text=re.sub(r'([?&](?:amp;)?OC=)[^&\s<>\"\']+',r'\1REDACTED',text,flags=re.I)
    def hidden_input(match):
        tag=match.group()
        if re.search(r'\b(?:id|name)\s*=\s*[\"\']OC[\"\']',tag,re.I):
            tag=re.sub(r'(\bvalue\s*=\s*[\"\'])[^\"\']*',r'\1REDACTED',tag,flags=re.I)
        return tag
    text=re.sub(r'<input\b[^>]*>',hidden_input,text,flags=re.I)
    if oc and oc in text:
        raise ValueError('인증값 제거 미완료: 원문 저장 중단')
    return text


def request(session, oc, endpoint, **params):
    try:
        response=session.get('https://www.law.go.kr/DRF/'+endpoint,params={'OC':oc,**params},timeout=(10,30))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError('법령 대조 수집 실패: '+type(exc).__name__) from None
    text=response.content.decode('utf-8-sig')
    return sanitize_response(text,oc)


def preserve(path, data):
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError('기존 대조 원문 자동 덮어쓰기 금지')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


def collect(*, paths=None, env_path=None, oc=None, client=None):
    """지정 위치에 연혁·구법 원본을 수집하고 saved/skipped 보고서를 반환한다.

    실제 API 요청과 저장을 수행한다. 전달된 client는 닫지 않는다.
    오류는 호출부로 전달하며 기존 원본을 덮어쓰지 않는다.
    """
    paths = paths if paths is not None else ReviewPaths()
    if oc is None:
        load_dotenv(Path(env_path) if env_path is not None else paths.project_root / '.env')
        oc = os.getenv('LAW_API_OC', '')
    if not isinstance(oc, str) or not oc.strip():
        raise ValueError('LAW_API_OC 필요')
    oc = oc.strip()
    result = {'saved': [], 'skipped': []}
    with (requests.Session() if client is None else nullcontext(client)) as session:
        for name,identifier in HISTORY_IDS.items():
            path=paths.history_dir/f'history_{identifier}_through_2023.html'
            if not path.exists():
                text=request(session,oc,'lawSearch.do',target='lsHistory',type='HTML',query=name,
                             display=100,page=1,efYd='20100101~20231231',sort='efdes')
                rows=history_rows(text,name)
                if len(rows)>=100:
                    raise ValueError('연혁 페이지 추가 조회 필요')
                preserve(path,text.encode('utf-8'))
                result['saved'].append(path.relative_to(paths.project_root).as_posix())
            else:
                history_rows(path.read_text(encoding='utf-8'), name)
                result['skipped'].append(path.relative_to(paths.project_root).as_posix())
        for master,(name,effective) in VERSIONS.items():
            path=paths.history_dir/f'{master}_{effective}.json'
            if path.exists():
                validate_body(read(path)['response_xml'],name,effective)
                result['skipped'].append(path.relative_to(paths.project_root).as_posix())
                continue
            text=request(session,oc,'lawService.do',target='law',type='XML',MST=master)
            root=validate_body(text,name,effective)
            data={'schema_version':1,'master_id':master,'law_name':name,'effective_date':effective,
                  'source_url':f'https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={master}',
                  'collected_at':datetime.now(timezone.utc).isoformat(),'version_scope':'case_date_comparison_only',
                  'embedding_candidate':False,'law_id':root.findtext('기본정보/법령ID'),'response_xml':text}
            write_json(path,data)
            result['saved'].append(path.relative_to(paths.project_root).as_posix())
    return result


def article(root, number, subclause=None):
    matches=[n for n in root.findall('./조문/조문단위') if n.findtext('조문번호')==number
             and n.findtext('조문여부')=='조문' and (n.findtext('조문가지번호') or '0') in {'0','00'}]
    if len(matches)!=1:
        raise ValueError('대조 조문 누락/중복: '+number)
    node=matches[0]
    if subclause:
        clauses=[h for h in node.iter('호') if (h.findtext('호번호') or '').rstrip('.')==subclause]
        if len(clauses)!=1:
            raise ValueError('대조 호 누락/중복')
        return clauses[0].findtext('호내용') or ''
    # 번호·메타데이터의 중복 출력 없이 조문/항/호/목 본문과 개정 표기를 보존한다.
    return '\n'.join(e.text or '' for e in node.iter() if e.tag in {'조문내용','항내용','호내용','목내용'})


def build_report(*, paths=None, law_chunks=None, documents=None):
    """원본과 사례 근거를 검증해 대조 보고서를 반환한다. 저장·API 호출은 없다.

    law_chunks와 documents(사례·고시 ID를 키로 가진 딕셔너리)를 전달하면
    해당 정제 자료를 재사용한다. 원본 파일 지문 검증은 유지한다.
    """
    paths = paths if paths is not None else ReviewPaths()
    inputs={}
    def track(path):
        relative=path.relative_to(paths.project_root).as_posix()
        data=checked(relative, project_root=paths.project_root)
        inputs[relative]=digest(data)
        return data
    roots={}
    refs={}
    histories={}
    for name,identifier in HISTORY_IDS.items():
        p=paths.history_dir/f'history_{identifier}_through_2023.html'
        histories[name]=history_rows(track(p).decode('utf-8'),name)
    for master,(name,effective) in VERSIONS.items():
        p=paths.history_dir/f'{master}_{effective}.json';data=json.loads(track(p))
        if data['master_id']!=master or data['embedding_candidate']:
            raise ValueError('과거 원문 식별/기본 검색 제외 불일치')
        roots[master]=validate_body(data['response_xml'],name,effective)
        refs[master]={'raw_file':p.relative_to(paths.project_root).as_posix(),'raw_sha256':inputs[p.relative_to(paths.project_root).as_posix()],
                      'law_name':name,'master_id':master,'effective_date':effective,'source_url':data['source_url']}
        if master!='260021' and not any(r['master_id']==master and r['effective_date']==effective for r in histories[name]):
            raise ValueError('원문 버전의 공식 연혁 목록 연결 누락')
    current={}
    for master in ['290239','286997','284567']:
        matches=list(paths.current_laws_dir.glob(f'*_{master}_*.json'))
        if len(matches)!=1:raise ValueError('수집본 대조 버전 누락/중복')
        p=matches[0];d=json.loads(track(p));root=ET.fromstring(d['response_xml'])
        name=root.findtext('기본정보/법령명_한글');effective=root.findtext('기본정보/시행일자')
        current[name]=(validate_body(d['response_xml'],name,effective),master)
        refs[master]={'raw_file':p.relative_to(paths.project_root).as_posix(),'raw_sha256':inputs[p.relative_to(paths.project_root).as_posix()],
                      'law_name':name,'master_id':master,'effective_date':effective,'source_url':d['source_url']}
    current[TERMS]=(roots['260021'],'260021')
    comparisons=[]
    for master,(name,_) in VERSIONS.items():
        if master=='260021':continue
        numbers=['2','9','31','32','52'] if name==LAW else ['13','40','41'] if name==DECREE else ['22'] if name==RULE else ['9']
        for number in numbers:
            for subclause in ([None,'10'] if name==LAW and number=='2' else [None]):
                old=article(roots[master],number,subclause);new=article(current[name][0],number,subclause)
                equal=re.sub(r'\s+','',old)==re.sub(r'\s+','',new)
                comparisons.append({'historical_master_id':master,'collected_master_id':current[name][1],
                    'law_name':name,'article_number':number,'subclause':subclause,'historical_text':old,'collected_text':new,
                    'historical_text_sha256':digest(old.encode()),'collected_text_sha256':digest(new.encode()),
                    'status':'same_except_whitespace' if equal else 'changed_non_target_definition',
                    'scope_note':'제2조 전체 변경을 계속거래 정의 변경으로 해석하지 않음' if not equal else '비교한 본문에 한정; 법령 전체 동일/개별 적용 확정 아님'})
                if not equal and not (master=='237455' and name==LAW and number=='2' and subclause is None):
                    raise ValueError('예상하지 못한 조문 변경: 추가 내용 검토 필요')
    cases=[]
    for sid,events in CASE_EVENTS.items():
        doc=documents[sid] if documents is not None else read(paths.consumer_dir / f'{sid}.json')
        track(paths.project_root/doc['raw_file'])
        if digest(checked(doc['raw_file'], project_root=paths.project_root))!=doc['raw_sha256'] or doc['decision_date'] is not None:
            raise ValueError('사례 원본 지문/미확인 조정일 상태 변경: 재검토 필요')
        units={u['unit_id']:u for u in doc['units']}
        timings=[]
        for kind,value,precision,uid,anchor in events:
            text=units[uid]['text'];start=text.find(anchor)
            if start<0:raise ValueError('사례 날짜 근거 누락')
            dates=[value+'-01',value+'-30'] if precision=='month' else [value,value]
            selections={}
            for name,rows in histories.items():
                selected=[select_at(rows,d)['master_id'] for d in dates]
                if len(set(selected))!=1:raise ValueError('월 단위 사건 날짜에 개정 경계 존재')
                selections[name]=selected[0]
                if selected[0] not in roots:raise ValueError('선정 구법 본문 미수집')
            timings.append({'event':kind,'value':value,'precision':precision,'evidence_unit_id':uid,
                            'evidence_start':start,'evidence_end':start+len(anchor),'evidence_text':anchor,'law_versions':selections})
        requested={LAW:{'2','31','32','52'},DECREE:{'13','40','41'},RULE:{'22'}}
        if law_chunks is None:
            law_chunks=[json.loads(line) for line in paths.law_chunk_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        linked=[c['chunk_id'] for c in law_chunks if c.get('law_name') in requested
                and str(c.get('article_number')) in requested[c['law_name']]]
        found={(c.get('law_name'),str(c.get('article_number'))) for c in law_chunks if c['chunk_id'] in linked}
        if found!={(name,n) for name,nums in requested.items() for n in nums}:
            raise ValueError('수집본 관련 조문 청크 연결 누락')
        for chunk in law_chunks:
            if chunk['chunk_id'] in linked and chunk['source_path']!=refs[current[chunk['law_name']][1]]['raw_file']:
                raise ValueError('대조 수집본과 연결 청크의 법령 버전 불일치')
        term_context='[시점 대조 참고 원문: 약관의 규제에 관한 법률 제9조 / 시행 2021-12-30 / '+refs['225063']['source_url']+']\n'+article(roots['225063'],'9')
        cases.append({'source_id':sid,'case_raw_file':doc['raw_file'],'case_raw_sha256':doc['raw_sha256'],
            'case_source_url':doc['source_url'],'decision_date':None,'events':timings,
            'unit_text_sha256':{u['unit_id']:digest(u['text'].encode()) for u in doc['units']},
            'comparison_status':'relevant_provisions_compared_at_documented_events',
            'required_context_chunk_ids':linked,'supporting_law_context':term_context,
            'warnings':['계약·해지 시점의 관련 조문 대조. 조정일은 확인되지 않아 null 유지.',
                        '폐강·양도 사례의 귀책사유·환급액·연 15% 지급 조건을 일반 계약에 자동 적용하지 않음.',
                        '2023년 조문과 수집본의 일치는 법령 전체 동일·최신 개정 전수 검증·개별 계약 적용 확정이 아님.']})
    notice=documents['continual_contract_refund'] if documents is not None else read(paths.consumer_dir / 'continual_contract_refund.json')
    notice_xml=ET.fromstring(track(paths.project_root/notice['raw_file']))
    meta=notice['administrative_metadata']
    if meta.get('시행일자')!='20191119' or not any('2019' in (e.text or '') and '9' in (e.text or '') for e in notice_xml.iter() if '발령번호' in e.tag):
        raise ValueError('사례 인용 2019-9호와 수집 고시 버전 불일치')
    return {'schema_version':1,'reviewed_on':'2026-10-11','scope':'수집된 두 필라테스 조정사례의 계약·해지 시점과 관련 조문 비교. 조정일/사건 적용 법률 확정은 범위 밖.',
            'comparison_policy':'원문 본문을 보존하고 공백만 제거하여 동일성 비교. 제2조 전체와 제10호를 별도로 비교.',
            'inputs':inputs,'law_sources':refs,'comparisons':comparisons,'cases':cases,
            'refund_notice':{'source_id':notice['source_id'],'raw_file':notice['raw_file'],'raw_sha256':notice['raw_sha256'],
                             'effective_date':'20191119','notice_number':'2019-9','status':'same_version_cited_by_case'},
            'search_policy':'law_history는 시점 대조 참고 원문. 기본 법령 청크에 자동 편입하지 않고 사례 메타데이터·원문 링크에 연결.'}


def verified_case(doc, *, paths=None, report=None, law_chunks=None, documents=None):
    """검증된 사례 메타데이터를 반환한다. report는 재사용할 보고서 딕셔너리다.

    기록의 정확한 지문을 보존하기 위해 보고서 파일을 읽고 전달된 report와도
    비교한다. 현재 원본·조문·사례 근거 검증을 생략하지 않는다.
    """
    paths = paths if paths is not None else ReviewPaths()
    report_bytes = checked(paths.report_path, project_root=paths.project_root)
    stored = json.loads(report_bytes)
    if report is None:
        report = stored
    elif report != stored:
        raise ValueError('전달한 보고서와 저장된 대조 기록 불일치')
    if report != build_report(paths=paths, law_chunks=law_chunks, documents=documents):
        raise ValueError('법령 시점 대조 기록과 현재 입력 불일치: 재대조 필요')
    matches = [case for case in report['cases'] if case['source_id'] == doc['source_id']]
    if len(matches) != 1:
        raise ValueError('사례 대조 기록 누락 또는 중복')
    case = matches[0]
    if (doc['raw_file'],doc['raw_sha256'],doc['decision_date'])!=(case['case_raw_file'],case['case_raw_sha256'],None):
        raise ValueError('사례 출처/미확인 조정일 변경: 재검토 필요')
    if case['unit_text_sha256']!={u['unit_id']:digest(u['text'].encode()) for u in doc['units']}:
        raise ValueError('사례 본문 변경: 시점 재검토 필요')
    return {**case,'review_record_file':paths.report_path.relative_to(paths.project_root).as_posix(),
            'review_record_sha256':digest(report_bytes),'law_sources':report['law_sources'],
            'refund_notice':report['refund_notice']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--history-dir', type=Path, default=RAW)
    parser.add_argument('--laws-dir', type=Path, default=RAW_LAWS_DATA_PATH)
    parser.add_argument('--consumer-dir', type=Path, default=PROCESSED_CONSUMER_SOURCES_DATA_PATH)
    parser.add_argument('--law-chunks', type=Path, default=CHUNK_LAWS_DATA_PATH / 'chunks.jsonl')
    parser.add_argument('--report-file', type=Path, default=REPORT)
    parser.add_argument('--env-file', type=Path, default=None)
    args = parser.parse_args(argv)
    paths = ReviewPaths(history_dir=args.history_dir, current_laws_dir=args.laws_dir,
                        consumer_dir=args.consumer_dir, law_chunk_path=args.law_chunks,
                        report_path=args.report_file)
    try:
        if args.collect:
            collect(paths=paths, env_path=args.env_file)
        report = build_report(paths=paths)
        if (paths.report_path.is_relative_to(paths.history_dir)
                or paths.report_path.is_relative_to(paths.current_laws_dir)
                or paths.report_path == paths.law_chunk_path
                or paths.report_path in [paths.consumer_dir / f'{sid}.json' for sid in (*CASE_EVENTS, 'continual_contract_refund')]):
            raise ValueError('보고서가 원본·청크 자료를 덮어쓸 수 없습니다.')
        write_json(paths.report_path, report)
    except (ValueError, OSError, KeyError, ET.ParseError, RuntimeError):
        print('법령 대조 실패: 입력 자료와 설정을 확인하세요.', file=sys.stderr)
        return 1
    print(json.dumps({'cases':len(report['cases']),'law_versions':len(report['law_sources']),
                     'comparisons':len(report['comparisons']),
                     'changed_rows':sum(r['status']!='same_except_whitespace' for r in report['comparisons'])},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
