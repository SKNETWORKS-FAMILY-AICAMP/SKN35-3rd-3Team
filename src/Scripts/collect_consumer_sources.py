"""공식 소비자 자료를 내용 해시 기반 파일로 보존한다. HTTPS 검증 유지.

uv run python src/Scripts/collect_consumer_sources.py
"""
import argparse
import hashlib
import json
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

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data/raw/consumer_sources'
MANIFEST = ROOT / 'data/processed/consumer_sources/source_manifest.json'
HOSTS = {'www.kca.go.kr', 'www.ftc.go.kr', 'www.law.go.kr', 'law.go.kr', 'news.seoul.go.kr'}


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
    return response


def preserve(data, source, suffix, output_dir=RAW_DIR):
    """기존 raw는 덮어쓰지 않는다. 동일 응답은 동일 파일을 재사용한다."""
    output_dir = Path(output_dir)
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
    return {**source, 'raw_file': path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            'raw_sha256': fingerprint, 'collected_at': datetime.now(timezone.utc).isoformat(),
            'format': suffix}


def collect_admin(client, source):
    load_dotenv(ROOT / '.env')
    oc = os.environ['LAW_API_OC']
    response = fetch(client, 'https://www.law.go.kr/DRF/lawService.do',
                     {'OC':oc, 'target':'admrul', 'type':'XML', 'LM':source['title']})
    text = response.content.decode('utf-8-sig')
    text = re.sub(r'([?&](?:amp;)?OC=)[^&\s<>"\']+', r'\1REDACTED', text, flags=re.I)
    root = ET.fromstring(text)
    if root.find('.//행정규칙명') is None:
        raise ValueError('정상 행정규칙 응답이 아닙니다.')
    record = preserve(text.encode('utf-8'), source, 'xml')
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
        attachments.append(preserve(pdf.content, item, 'pdf'))
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


def run():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    report = {'saved': [], 'failures': [], 'started_at':datetime.now(timezone.utc).isoformat()}
    client = session()
    queue = list(manifest['sources'])
    seen = set()
    for source in queue:
        if source['source_id'] in seen:
            continue
        seen.add(source['source_id'])
        print('[수집] '+source['source_id'], flush=True)
        try:
            if source['document_type'] == 'administrative_rule':
                report['saved'].extend(collect_admin(client, source))
            else:
                response = fetch(client, source['source_url'])
                data = response.content
                suffix = ('pdf' if data.startswith(b'%PDF') else 'hwpx' if data.startswith(b'PK')
                          else 'hwp' if data.startswith(b'\xd0\xcf\x11\xe0') else 'html')
                report['saved'].append(preserve(data, {**source,'resolved_url':safe_url(response.url)}, suffix))
                if suffix == 'html':
                    queue.extend(discover(source,data) if source['document_type']=='discovery' else attachments(source,data))
        except (requests.RequestException, ValueError, OSError, KeyError, ET.ParseError, RuntimeError) as exc:
            # 예외 문자열에는 인증값/요청 URL이 포함될 수 있으므로 기록하지 않는다.
            report['failures'].append({'source_id':source['source_id'], 'error_type':type(exc).__name__})
        time.sleep(0.3)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(RAW_DIR / f'collection_report_{stamp}.json', report)
    write_json(MANIFEST.parent / 'consumer_collection_report.json', report)
    manifest['sources'] = list({s['source_id']:s for s in queue}.values())
    write_json(MANIFEST,manifest)
    print(f"수집 {len(report['saved'])} / 실패 {len(report['failures'])}")
    return report


if __name__ == '__main__':
    raise SystemExit(bool(run()['failures']))
