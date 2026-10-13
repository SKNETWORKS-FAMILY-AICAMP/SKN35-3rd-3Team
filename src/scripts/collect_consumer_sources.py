"""공식 소비자 자료를 내용 해시 기반 파일로 보존한다. HTTPS 검증 유지.

uv run python src/scripts/collect_consumer_sources.py
"""
import argparse
import hashlib
import json
import math
import sys
import os
import re
import ssl
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    PROCESSED_CONSUMER_SOURCES_DATA_PATH,
    PROJECT_PATH,
    RAW_CONSUMER_SOURCES_DATA_PATH,
)

ROOT = PROJECT_PATH
RAW_DIR = RAW_CONSUMER_SOURCES_DATA_PATH
MANIFEST = PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'source_manifest.json'
HOSTS = {'www.kca.go.kr', 'www.ftc.go.kr', 'www.law.go.kr', 'law.go.kr', 'news.seoul.go.kr', 'labor.moel.go.kr', '1350.moel.go.kr', 'www.easylaw.go.kr'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_url(url):
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query=urlencode([(k,v) for k,v in parse_qsl(parsed.query) if k.lower() != 'oc'])))


class Node:
    def __init__(self, tag='', attrs=None):
        self.tag, self.attrs, self.children = tag, dict(attrs or []), []

    def find(self, predicate):
        result = [self] if predicate(self) else []
        for child in self.children:
            if isinstance(child, Node):
                result.extend(child.find(predicate))
        return result

    def text(self):
        if self.tag in {'script','style','noscript'}:
            return ''
        parts = [c.text() if isinstance(c,Node) else c for c in self.children]
        sep = '\n' if self.tag in {'table','tr','div','p','ul','li','section','article','h1','h2','h3','h4','br'} else ' '
        return sep.join(parts)


class HTMLTree(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node('root')
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'br','img','input','hr','meta','link','source','wbr','area','base','embed','param','track','col'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1, 0, -1):
            if self.stack[index].tag == tag:
                self.stack = self.stack[:index]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def decode_html(data):
    head = data[:4000].decode('ascii', errors='ignore')
    match = re.search(r'charset\s*=\s*["\']?([\w-]+)', head, re.I)
    return data.decode(match.group(1) if match else 'utf-8', errors='strict')


def session():
    """KCA가 보내지 않는 중간 인증서만 추가. 기존 CA·호스트 검증을 유지한다."""
    result = requests.Session()
    result.headers['User-Agent'] = 'LegalRAG-Research/1.0 (public document collection)'
    return result


def repair_kca_tls(client):
    import certifi
    import warnings
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization.pkcs7 import load_der_pkcs7_certificates
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    response = requests.get('http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt', timeout=(10,30))
    response.raise_for_status()
    context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(response.content))
    chain = requests.get('http://crt.sectigo.com/SectigoPublicServerAuthenticationRootR46.p7c', timeout=(10,30))
    chain.raise_for_status()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        certificates = load_der_pkcs7_certificates(chain.content)
    # 자체 서명 루트는 추가하지 않는다. 기존 신뢰 CA로 연결되는 교차 서명만 추가한다.
    for certificate in certificates:
        if certificate.subject != certificate.issuer:
            context.load_verify_locations(cadata=certificate.public_bytes(serialization.Encoding.PEM).decode())

    class Adapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs['ssl_context'] = context
            return super().init_poolmanager(*args, **kwargs)

    client.mount('https://www.kca.go.kr/', Adapter())


def fetch(client, url, params=None):
    if urlparse(url).scheme != 'https' or urlparse(url).hostname not in HOSTS:
        raise ValueError('허용된 공식 HTTPS 출처가 아닙니다.')
    try:
        response = client.get(url, params=params, timeout=(10,30))
    except requests.exceptions.SSLError:
        if urlparse(url).hostname != 'www.kca.go.kr':
            raise RuntimeError('HTTPS 인증서 검증 실패') from None
        repair_kca_tls(client)
        response = client.get(url, params=params, timeout=(10,30))
    response.raise_for_status()
    if urlparse(response.url).hostname not in HOSTS or urlparse(response.url).scheme != 'https':
        raise ValueError('공식 출처 외부로 이동했습니다.')
    if urlparse(response.url).hostname == 'www.easylaw.go.kr':
        tree=HTMLTree(decode_html(response.content)).root
        if len(tree.find(lambda n:'ovDivbox' in n.attrs.get('class','').split())) != 1:
            raise ValueError('생활법령 본문 누락 또는 접근 차단 화면')
    return response


def preserve(data, source, suffix, output_dir=RAW_DIR, *, project_root=ROOT):
    """기존 raw는 덮어쓰지 않는다. 동일 응답은 동일 파일을 재사용한다."""
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root).resolve()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', source['source_id']):
        raise ValueError('잘못된 출처 ID')
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(data)
    path = output_dir / f"{source['source_id']}_{fingerprint[:16]}.{suffix}"
    if path.exists():
        if digest(path.read_bytes()) != fingerprint:
            raise ValueError('기존 원본 해시 불일치')
    else:
        path.write_bytes(data)
    return {**source, 'raw_file': path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else str(path),
            'raw_sha256': fingerprint, 'collected_at': datetime.now(timezone.utc).isoformat(),
            'format': suffix}


def collect_admin(client, source, *, output_dir=RAW_DIR, project_root=ROOT,
                  env_path=None, oc=None):
    """행정규칙과 첨부 원본을 지정한 위치에 보존하고 수집 기록 목록을 반환한다."""
    if oc is None:
        load_dotenv(Path(env_path) if env_path is not None else Path(project_root) / '.env')
        oc = os.environ['LAW_API_OC']
    if not isinstance(oc, str) or not oc.strip():
        raise ValueError("LAW_API_OC 인증값이 필요합니다.")
    oc = oc.strip()
    response = fetch(client, 'https://www.law.go.kr/DRF/lawService.do',
                     {'OC':oc, 'target':'admrul', 'type':'XML',
                      **({'ID':source['administrative_id']} if source.get('administrative_id') else {'LM':source['title']})})
    text = response.content.decode('utf-8-sig')
    text = re.sub(r'([?&](?:amp;)?OC=)[^&\s<>"\']+', r'\1REDACTED', text, flags=re.I)
    root = ET.fromstring(text)
    if root.find('.//행정규칙명') is None:
        raise ValueError('정상 행정규칙 응답이 아닙니다.')
    info=root.find('행정규칙기본정보')
    if info is None or re.sub(r'\s','',info.findtext('행정규칙명','')) != re.sub(r'\s','',source['title']):
        raise ValueError('요청한 행정규칙명과 응답이 다릅니다.')
    if source.get('administrative_id') and info.findtext('행정규칙일련번호')!=source['administrative_id']:
        raise ValueError('행정규칙 고정 버전 불일치')
    effective=info.findtext('시행일자','')
    if not re.fullmatch(r'\d{8}',effective) or datetime.strptime(effective,'%Y%m%d').date()>datetime.now().date():
        raise ValueError('행정규칙 시행일 확인 필요 또는 시행예정 자료')
    if source.get('expected_effective_date') and effective != source['expected_effective_date']:
        raise ValueError('고정한 적용 연도의 시행일과 응답이 다릅니다.')
    record = preserve(text.encode('utf-8'), source, 'xml', output_dir, project_root=project_root)
    attachments = []
    for index, unit in enumerate(root.findall('.//별표단위')):
        link = unit.findtext('별표서식PDF파일링크')
        if not link:
            continue
        url = urljoin('https://www.law.go.kr', link)
        pdf = fetch(client, url)
        if not pdf.content.startswith(b'%PDF'):
            raise ValueError('별표 PDF 응답이 아닙니다.')
        item = {**source, 'source_id':source['source_id']+f'_appendix_{index}',
                'title':unit.findtext('별표제목') or '별표', 'source_url':safe_url(url),
                'parent_source_id':source['source_id'], 'document_type':'administrative_rule_appendix'}
        attachments.append(preserve(pdf.content, item, 'pdf', output_dir, project_root=project_root))
    if source['source_id']=='minimum_wage_2026':
        pairs=[];name=None
        for n in root.find('첨부파일'):
            if n.tag=='첨부파일명':name=(n.text or '').strip()
            elif n.tag=='첨부파일링크' and name and name.startswith(source['title']) and name.endswith('.pdf'):
                pairs.append((name,n.text))
        if len(pairs)!=1:
            raise ValueError('최저임금 고시 본문 PDF를 확정할 수 없습니다.')
        url=urljoin('https://www.law.go.kr',pairs[0][1]).replace('http://','https://',1)
        response=fetch(client,url)
        if not response.content.startswith(b'%PDF'):
            raise ValueError('최저임금 고시 본문 PDF 응답이 아닙니다.')
        attachment=preserve(response.content,{**source,'source_id':source['source_id']+'_attachment_body',
            'title':pairs[0][0],'source_url':safe_url(url),'parent_source_id':source['source_id'],
            'document_type':'administrative_rule_appendix'},'pdf', output_dir, project_root=project_root)
        record['body_pdf_file']=attachment['raw_file']
        record['body_pdf_sha256']=attachment['raw_sha256']
        record['body_pdf_url']=attachment['source_url']
        attachments.append(attachment)
    return [record]+attachments


def discover(source, data):
    tree = HTMLTree(decode_html(data)).root
    found = []
    for node in tree.find(lambda n:n.tag=='a'):
        title = ' '.join(node.text().split())
        if source['source_id'].startswith('kca_discovery'):
            match = re.search(r'fn_view_bbd\("(\d+)",\s*"(\d+)"\)', node.attrs.get('onclick',''))
            if not match or not title:
                continue
            relevant = ((source['domain']=='fitness_refund' and any(k in title for k in ['헬스','필라테스','PT']) and any(k in title for k in ['환급','해지']))
                        or (source['domain']=='parcel_loss' and '택배' in title and any(k in title for k in ['분실','배상','파손']))
                        or (source['domain']=='shopping_return' and any(k in title for k in ['반품','청약철회'])))
            if relevant:
                found.append({**source,'source_id':'kca_case_'+match[1], 'title':title,
                    'document_type':'mediation_case','source_url':f'https://www.kca.go.kr/odr/bj/br/osBjDecisionExamDetW.do?brdId={match[2]}&dataStts=Y&seq={match[1]}'})
        elif source['source_id'].startswith('ftc_terms_discovery'):
            href = node.attrs.get('href','')
            if 'selectBbsNttView' in href and any(k in title for k in ['필라테스','체력단련장','택배']):
                sn = dict(parse_qsl(urlparse(href).query)).get('nttSn')
                found.append({**source,'source_id':'ftc_terms_'+sn,'title':title,'document_type':'standard_terms_notice',
                              'source_url':urljoin(source['source_url'],href)})
    return found[:6]


def attachments(source, data):
    if source['document_type'] != 'standard_terms_notice':
        return []
    tree = HTMLTree(decode_html(data)).root
    found = []
    for node in tree.find(lambda n:n.tag=='a'):
        href=node.attrs.get('href','')
        if 'download' in href.lower() or 'fileDown' in href:
            url=urljoin(source['source_url'],href)
            if urlparse(url).hostname=='www.ftc.go.kr':
                found.append({**source,'source_id':source['source_id']+'_file_'+digest(url.encode())[:8],
                    'source_url':url,'document_type':'standard_terms','parent_source_id':source['source_id']})
    return found


def run(source_ids=None, *, manifest_path=None, output_dir=None, report_path=None,
        project_root=ROOT, env_path=None, oc=None, client=None, delay=0.3,
        progress=None):
    """원본을 수집·보존하고 누적 수집 보고서 딕셔너리를 반환한다.

    실제 HTTP 요청과 파일 저장을 수행한다. client를 전달하면 재사용하며 닫지 않는다.
    자체 생성한 세션은 종료 시 닫는다. progress에는 진행 ID를 받을 함수를 전달한다.
    자료별 오류는 failures에 인증값 없이 기록한다. 입력·보고서 저장 오류는 전달한다.
    """
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
        raise ValueError("delay는 유한한 0 이상의 숫자여야 합니다.")
    if progress is not None and not callable(progress):
        raise TypeError("progress는 호출 가능한 함수여야 합니다.")
    if source_ids is not None:
        if isinstance(source_ids, (str, bytes)):
            raise TypeError("source_ids는 문자열 목록이어야 합니다.")
        source_ids = list(source_ids)
        if not source_ids or any(not isinstance(identifier, str) or not identifier for identifier in source_ids):
            raise ValueError("source_ids에는 비어 있지 않은 출처 ID가 필요합니다.")
    manifest_path = Path(manifest_path if manifest_path is not None else MANIFEST).resolve()
    output_dir = Path(output_dir if output_dir is not None else RAW_DIR).resolve()
    project_root = Path(project_root).resolve()
    report_path = Path(report_path).resolve() if report_path is not None else manifest_path.parent / 'consumer_collection_report.json'
    if manifest_path.is_relative_to(output_dir) or report_path.is_relative_to(output_dir) or report_path == manifest_path:
        raise ValueError("manifest와 현재 보고서는 원본 저장 폴더 밖에 별도로 지정해야 합니다.")
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    report = {'saved': [], 'failures': [], 'started_at':datetime.now(timezone.utc).isoformat()}
    queue = [s for s in manifest['sources'] if source_ids is None or s['source_id'] in source_ids]
    if source_ids and set(source_ids) != {s['source_id'] for s in queue}:
        raise ValueError('manifest에 없는 수집 ID')
    owns_client = client is None
    if owns_client:
        client = session()
    seen = set()
    try:
        for source in queue:
            if source['source_id'] in seen:
                continue
            seen.add(source['source_id'])
            if progress is not None:
                progress(source['source_id'])
            try:
                if source['document_type'] == 'administrative_rule':
                    report['saved'].extend(collect_admin(client, source, output_dir=output_dir, project_root=project_root, env_path=env_path, oc=oc))
                else:
                    response = fetch(client, source['source_url'])
                    data = response.content
                    suffix = ('pdf' if data.startswith(b'%PDF') else 'hwpx' if data.startswith(b'PK')
                              else 'hwp' if data.startswith(b'\xd0\xcf\x11\xe0') else 'html')
                    report['saved'].append(preserve(data, {**source,'resolved_url':safe_url(response.url)}, suffix, output_dir, project_root=project_root))
                    if suffix == 'html':
                        queue.extend(discover(source,data) if source['document_type']=='discovery' else attachments(source,data))
            except (requests.RequestException, ValueError, OSError, KeyError, ET.ParseError, RuntimeError) as exc:
                # 예외 문자열에는 인증값/요청 URL이 포함될 수 있으므로 기록하지 않는다.
                report['failures'].append({'source_id':source['source_id'], 'error_type':type(exc).__name__})
            time.sleep(delay)
    finally:
        if owns_client:
            client.close()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(output_dir / f'collection_report_{stamp}.json', report)
    current_path=report_path
    if source_ids and current_path.exists():
        previous=json.loads(current_path.read_text(encoding='utf-8'))
        replaced={s['source_id'] for s in report['saved']}
        report['saved']=[s for s in previous['saved'] if s['source_id'] not in replaced]+report['saved']
        report['failures']=[s for s in previous['failures'] if s['source_id'] not in set(source_ids)]+report['failures']
    write_json(current_path, report)
    manifest['sources'] = list({s['source_id']:s for s in manifest['sources']+queue}.values())
    write_json(manifest_path,manifest)
    return report


def main(argv=None):
    """명령행 실행, 진행 출력과 종료 코드 처리."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id', action='append', help='특정 자료만 수집하고 기존 기록 유지')
    parser.add_argument('--manifest-file', type=Path, default=MANIFEST)
    parser.add_argument('--output-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--report-file', type=Path, default=None)
    parser.add_argument('--env-file', type=Path, default=None)
    parser.add_argument('--delay', type=float, default=0.3)
    args = parser.parse_args(argv)
    try:
        report = run(args.source_id, manifest_path=args.manifest_file, output_dir=args.output_dir,
                     report_path=args.report_file, env_path=args.env_file, delay=args.delay,
                     progress=lambda identifier: print('[수집] ' + identifier, flush=True))
    except (ValueError, TypeError, KeyError, OSError):
        print('수집 설정 또는 입력 파일을 확인하세요.', file=sys.stderr)
        return 1
    print(f"수집 {len(report['saved'])} / 실패 {len(report['failures'])}")
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
