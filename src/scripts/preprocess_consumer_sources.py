"""공식 소비자 자료를 유형·역할·시점·출처를 보존하며 정제한다."""
import argparse
import json
from dataclasses import dataclass
from collections.abc import Callable, Sequence
import re
import struct
import zlib
from collections import Counter
from difflib import SequenceMatcher, unified_diff
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    CHUNK_LAWS_DATA_PATH,
    PROCESSED_CONSUMER_SOURCES_DATA_PATH,
    PROJECT_PATH,
    RAW_CONSUMER_SOURCES_DATA_PATH,
)

ROOT = PROJECT_PATH

if __package__:
    from .collect_consumer_sources import HTMLTree, decode_html, digest, write_json
else:
    from collect_consumer_sources import HTMLTree, decode_html, digest, write_json

RAW_DIR = RAW_CONSUMER_SOURCES_DATA_PATH
OUTPUT_DIR = PROCESSED_CONSUMER_SOURCES_DATA_PATH
MANIFEST = OUTPUT_DIR / "source_manifest.json"
APPENDIX_REVIEW_FILE = OUTPUT_DIR/'consumer_appendix_review_pmh.json'


@dataclass(frozen=True)
class ProcessingContext:
    """참조 자료와 출력 위치를 분리하고 원본·검토 기록·법령 청크를 명시적으로 전달한다."""
    project_root: Path = ROOT
    processed_dir: Path = OUTPUT_DIR
    raw_dir: Path = RAW_DIR
    appendix_review_path: Path = APPENDIX_REVIEW_FILE
    manifest_path: Path = MANIFEST
    law_chunk_path: Path = CHUNK_LAWS_DATA_PATH / "chunks.jsonl"
    law_records: Sequence[dict] | None = None
    case_verifier: Callable[[dict], dict] | None = None

    def __post_init__(self):
        for name in ("project_root", "processed_dir", "raw_dir", "appendix_review_path", "manifest_path", "law_chunk_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        if self.case_verifier is not None and not callable(self.case_verifier):
            raise TypeError("case_verifier는 호출 가능한 함수여야 합니다.")


def _law_records(context):
    if context.law_records is not None:
        return context.law_records
    return [json.loads(line) for line in context.law_chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]


PARCEL_VERSIONS = {
    'ftc_terms_11185_file_a321590d': ('ftc_parcel_terms_14849', '2020-06-05', 26, 'preferred_collected_version'),
    'ftc_terms_11125_file_e27d5f98': ('ftc_parcel_terms_14786', '2007-12-28', 23, 'historical'),
}


def hwp_paragraph_text(data):
    """검증용: 수집한 HWP 5/CFB v3의 문단 텍스트만 읽는다. 렌더러가 아니다.

    HWP 레코드/컨트롤 형식: https://tech.hancom.com/python-hwp-parsing-2/
    그림·수식·필드 결과는 해석하지 않으며 해당 개체가 있으면 호출자가 검토한다.
    """
    if data[:8] != bytes.fromhex('d0cf11e0a1b11ae1') or len(data) < 512:
        raise ValueError('HWP compound header 불일치')
    if struct.unpack_from('<H', data, 26)[0] != 3 or struct.unpack_from('<I', data, 72)[0]:
        raise ValueError('검증용 HWP reader가 지원하지 않는 compound 형식')
    sector_size = 1 << struct.unpack_from('<H', data, 30)[0]
    if sector_size != 512:
        raise ValueError('지원하지 않는 sector 크기')

    def sector(n):
        result = data[(n+1)*sector_size:(n+2)*sector_size]
        if len(result) != sector_size:
            raise ValueError('HWP sector 범위 초과')
        return result

    fat = []
    for n in struct.unpack_from('<109I', data, 76):
        if n < 0xfffffffa:
            fat.extend(struct.unpack('<128I', sector(n)))

    def chain(n, table, reader):
        parts, seen = [], set()
        while n < 0xfffffffa:
            if n in seen or n >= len(table):
                raise ValueError('HWP chain 순환 또는 범위 초과')
            seen.add(n)
            parts.append(reader(n))
            n = table[n]
        return b''.join(parts)

    directory = chain(struct.unpack_from('<I', data, 48)[0], fat, sector)
    entries = []
    for i in range(0, len(directory), 128):
        entry = directory[i:i+128]
        length = struct.unpack_from('<H', entry, 64)[0]
        if length:
            if length > 64 or length % 2:
                raise ValueError('HWP directory name 불일치')
            entries.append((entry[:length-2].decode('utf-16le'), entry[66],
                            struct.unpack_from('<I', entry, 116)[0],
                            struct.unpack_from('<Q', entry, 120)[0]))
    roots = [e for e in entries if e[1] == 5]
    if len(roots) != 1:
        raise ValueError('HWP root 불일치')
    root = roots[0]
    mini = chain(root[2], fat, sector)[:root[3]]
    mini_fat_bytes = chain(struct.unpack_from('<I', data, 60)[0], fat, sector)
    mini_fat = struct.unpack('<'+str(len(mini_fat_bytes)//4)+'I', mini_fat_bytes)
    mini_size = 1 << struct.unpack_from('<H', data, 32)[0]

    def stream(entry):
        content = (chain(entry[2], mini_fat, lambda n:mini[n*mini_size:(n+1)*mini_size])
                   if entry[3] < 4096 else chain(entry[2], fat, sector))
        if len(content) < entry[3]:
            raise ValueError('HWP stream 누락')
        return content[:entry[3]]

    headers = [e for e in entries if e[0] == 'FileHeader']
    if len(headers) != 1:
        raise ValueError('HWP FileHeader 누락')
    header = stream(headers[0])
    if not header.startswith(b'HWP Document File'):
        raise ValueError('HWP 5 signature 불일치')
    flags = struct.unpack_from('<I', header, 36)[0]
    if flags & 6:
        raise ValueError('암호화/배포용 HWP는 검증 지원하지 않음')
    sections = sorted((e for e in entries if e[1] == 2 and re.fullmatch(r'Section\d+', e[0])),
                      key=lambda e:int(e[0][7:]))
    if [e[0] for e in sections] != [f'Section{i}' for i in range(len(sections))] or not sections:
        raise ValueError('HWP section 불완전')
    paragraphs, objects = [], []
    for entry in sections:
        body = stream(entry)
        if flags & 1:
            body = zlib.decompress(body, -15)
        offset = 0
        while offset < len(body):
            value = struct.unpack_from('<I', body, offset)[0]
            offset += 4
            tag, size = value & 1023, value >> 20
            if size == 4095:
                size = struct.unpack_from('<I', body, offset)[0]
                offset += 4
            record = body[offset:offset+size]
            offset += size
            if len(record) != size:
                raise ValueError('HWP record 잘림')
            if tag != 67:
                continue
            text, cursor = [], 0
            while cursor < len(record):
                code = struct.unpack_from('<H', record, cursor)[0]
                if code < 32:
                    width = 2 if code in [0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31] else 16
                    if code == 11:
                        objects.append(record[cursor:cursor+width].hex())
                    text.append({9:'\t', 10:'\n', 13:'\n', 24:'-', 30:' ', 31:' '}.get(code, ''))
                    cursor += width
                else:
                    text.append(record[cursor:cursor+2].decode('utf-16le'))
                    cursor += 2
            paragraphs.append(''.join(text))
    return clean('\n'.join(paragraphs)), objects


def normalized_text(text):
    return re.sub(r'\s+', '', text)


def extract_parcel_pdf(source, data, *, context=None):
    context = context if context is not None else ProcessingContext()
    import pdfplumber
    basename, revision, article_count, version_role = PARCEL_VERSIONS[source['source_id']]
    pdf_path = context.processed_dir/'converted'/(basename+'.pdf')
    metadata = json.loads(pdf_path.with_suffix('.conversion.json').read_text(encoding='utf-8'))
    if (Path(metadata['raw_source'].replace('\\', '/')) != Path(source['raw_file'])
            or metadata['raw_sha256'] != source['raw_sha256']
            or metadata['pdf_sha256'] != digest(pdf_path.read_bytes())
            or metadata['document_revision_date'] != revision):
        raise ValueError('HWP/PDF 변환 provenance 불일치')
    pages, offsets, cursor = [], [], 0
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            text = clean(page.extract_text() or '')
            if not text:
                raise ValueError('택배 약관 PDF 빈 본문 페이지')
            pages.append(text)
            offsets.append((number, cursor, cursor+len(text)))
            cursor += len(text)+1
    if len(pages) != metadata['pdf_page_count']:
        raise ValueError('PDF page count 불일치')
    text = '\n'.join(pages)
    raw_text, objects = hwp_paragraph_text(data)
    matches = list(re.finditer(r'(?m)^제\s*(\d+)조\s*\(([^)]+)\)', text))
    if [int(m[1]) for m in matches] != list(range(1, article_count+1)):
        raise ValueError('택배 약관 조문 번호 누락/중복')
    raw_start = re.search(r'(?m)^제\s*1조\s*\(', raw_text)
    if not raw_start:
        raise ValueError('HWP 첫 조문 누락')
    # PDF/HWP의 줄바꿈·읽기 순서는 달라도 조문 본문 문자가 모두 보존되어야 한다.
    pdf_body, hwp_body = text[matches[0].start():], raw_text[raw_start.start():]
    if Counter(normalized_text(pdf_body)) != Counter(normalized_text(hwp_body)):
        raise ValueError('택배 약관 PDF와 HWP 조문 본문 문자 불일치')
    raw_matches = list(re.finditer(r'(?m)^제\s*(\d+)조\s*\(([^)]+)\)', raw_text))
    if [int(m[1]) for m in raw_matches] != list(range(1, article_count+1)):
        raise ValueError('HWP 조문 번호 누락/중복')
    for i, (pdf_match, raw_match) in enumerate(zip(matches, raw_matches)):
        pdf_end = matches[i+1].start() if i+1 < len(matches) else len(text)
        raw_end = raw_matches[i+1].start() if i+1 < len(raw_matches) else len(raw_text)
        if Counter(normalized_text(text[pdf_match.start():pdf_end])) != Counter(normalized_text(raw_text[raw_match.start():raw_end])):
            raise ValueError(f'HWP/PDF 제{pdf_match[1]}조 문자 불일치')
    date = re.search(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*개정', text)
    if not date or '-'.join([date[1],date[2].zfill(2),date[3].zfill(2)]) != revision:
        raise ValueError('택배 약관 개정일 불일치')
    boundaries = [0]+[m.start() for m in matches]+[len(text)]
    units = []
    for i in range(len(boundaries)-1):
        start, end = boundaries[i:i+2]
        article = matches[i-1] if i else None
        page_numbers = [n for n, a, b in offsets if a < end and b > start]
        units.append(unit('preamble' if not article else 'terms_article_'+article[1],
            'terms_preamble' if not article else 'standard_terms_article', text[start:end],
            atomic=True, start=start, end=end, offset_basis='pdf_extracted_text',
            article_number=int(article[1]) if article else None,
            article_title=article[2] if article else None,
            page_numbers=page_numbers,
            source_locator=f'PDF pp. {page_numbers[0]}–{page_numbers[-1]}'
                + (f' / 제{article[1]}조({article[2]})' if article else ' / 머리말')))
    anomalies = []
    for item in units:
        if re.search(r'운송장기재운임액의\s*200\s*$', item['text'], re.M):
            warning = '원문 제22조 제2항 제3호 나목의 "운송장기재운임액의 200"에 단위가 없음. 200%로 추정하거나 이 문장으로 배상액을 계산하지 말 것.'
            item['warnings'] = [warning]
            item['source_anomaly'] = 'missing_rate_unit_in_original'
            item['calculation_restricted_clause'] = '제22조 제2항 제3호 나목'
            anomalies.append({'unit_id':item['unit_id'], 'warning':warning,
                              'confirmed_in':'raw HWP paragraph text and converted PDF'})
    return {'units':units, 'extracted_text':text, 'hwp_validation_text':raw_text,
            'derived_pdf_file':pdf_path.relative_to(context.project_root).as_posix(),
            'derived_pdf_sha256':metadata['pdf_sha256'], 'pdf_page_count':len(pages),
            'terms_revision_date':revision, 'version_role':version_role,
            'conversion_validation':{'body_character_counts_match':True,
                'article_character_counts_match':True,
                'body_text_order_matches':normalized_text(pdf_body)==normalized_text(hwp_body),
                'object_controls':objects, 'scope':'본문 문단 문자 대조; 그림·수식은 별도 검토'},
            'historical_reference':version_role == 'historical', 'source_anomalies':anomalies}


def compare_parcel_versions(documents):
    versions = {d['terms_revision_date']:d for d in documents if d['source_id'] in PARCEL_VERSIONS}
    old, new = versions['2007-12-28'], versions['2020-06-05']
    def key(title):
        # 수화인/수하인은 원문을 보존하면서 제목 대응에만 같은 키로 처리한다.
        return normalized_text(title).replace('수하인', '수화인')
    def articles(doc):
        result = {key(u['article_title']):u for u in doc['units'] if u.get('article_number')}
        if len(result) != len(doc['units'])-1:
            raise ValueError('대조 조문 제목 중복')
        return result
    before, after = articles(old), articles(new)
    records = []
    for title in sorted(before.keys() | after.keys()):
        a, b = before.get(title), after.get(title)
        a_body = a['text'].split(')', 1)[1] if a else ''
        b_body = b['text'].split(')', 1)[1] if b else ''
        status = ('added' if not a else 'removed' if not b else
                  'unchanged' if normalized_text(a_body) == normalized_text(b_body) else 'changed')
        records.append({'article_title':(b or a)['article_title'], 'status':status,
            'old_article_number':a['article_number'] if a else None,
            'new_article_number':b['article_number'] if b else None,
            'old_unit_id':a['unit_id'] if a else None, 'new_unit_id':b['unit_id'] if b else None,
            'old_pages':a['page_numbers'] if a else [], 'new_pages':b['page_numbers'] if b else [],
            'old_text':a['text'] if a else None, 'new_text':b['text'] if b else None,
            'text_similarity':SequenceMatcher(None, normalized_text(a_body), normalized_text(b_body), autojunk=False).ratio() if a and b else None,
            'diff':list(unified_diff(a['text'].splitlines() if a else [], b['text'].splitlines() if b else [],
                                    fromfile='2007-12-28', tofile='2020-06-05', lineterm=''))})
    return {'comparison_type':'article_title_aligned_text_diff',
        'old_source_id':old['source_id'], 'new_source_id':new['source_id'],
        'old_revision_date':old['terms_revision_date'], 'new_revision_date':new['terms_revision_date'],
        'old_article_count':len(before), 'new_article_count':len(after),
        'counts':dict(Counter(r['status'] for r in records)), 'articles':records,
        'source_anomalies':new['source_anomalies'],
        'limitations':['문자 차이 대조이며 개정의 법률적 효과를 판단하지 않음.',
                       '2020년본은 수집한 두 버전 중 우선 사용본. 최신 현행 여부를 별도로 단정하지 않음.']}


def clean(text):
    return '\n'.join(line.strip() for line in text.replace('\r','\n').splitlines() if line.strip())


def unit(identifier, role, text, **extra):
    return {'unit_id':identifier, 'role':role, 'text':text, **extra}


def extract_case(data):
    tree = HTMLTree(decode_html(data)).root
    titles = tree.find(lambda n:n.tag=='h4')
    if len(titles) != 1:
        raise ValueError('사례 제목이 없거나 모호합니다.')
    labels = {'사건개요':'facts', '당사자주장':'party_claims', '판단':'reasoning',
              '결정사항':'decision', '조정결정사항':'decision', '관련법률':'cited_law'}
    units, dates = [], {}
    for row in tree.find(lambda n:n.tag=='tr'):
        th = row.find(lambda n:n.tag=='th')
        td = row.find(lambda n:n.tag=='td')
        if len(th) != 1 or len(td) != 1:
            continue
        label, text = clean(th[0].text()).replace(' ',''), clean(td[0].text())
        if label in labels and text:
            units.append(unit('case_'+labels[label], labels[label], text, source_label=label,
                              source_locator='HTML table row th='+label))
        if label in ['수정일','등록일','결정일','조정결정일']:
            match = re.search(r'\d{4}-\d{2}-\d{2}',text)
            if match:
                dates[label] = match.group()
    if not {'facts','reasoning'}.issubset({u['role'] for u in units}):
        raise ValueError('사건개요 또는 판단 본문 누락')
    return clean(titles[0].text()), units, dates


def extract_admin(data):
    root = ET.fromstring(data)
    info = root.find('행정규칙기본정보')
    if info is None or info.findtext('행정규칙명') != '소비자분쟁해결기준':
        raise ValueError('소비자분쟁해결기준 원문이 아닙니다.')
    units = [unit(f'article_{i}', 'general_rule', n.text or '',
                  source_locator=f'조문내용[{i+1}]') for i,n in enumerate(root.findall('조문내용'))]
    appendix = root.find('.//별표단위[별표번호="0002"]')
    if appendix is None:
        raise ValueError('품목별 해결기준 별표 누락')
    text = appendix.findtext('별표내용') or ''
    headings = list(re.finditer(r'(?m)^[ \t]*<[^>\n]+>',text))
    targets = {'체육시설업':'fitness_refund','택배 및 퀵서비스업':'parcel_loss','인터넷쇼핑몰업':'shopping_return'}
    domains = set()
    for i, heading in enumerate(headings):
        for key, domain in targets.items():
            if key not in heading.group():
                continue
            start, end = heading.start(), headings[i+1].start() if i+1<len(headings) else len(text)
            units.append(unit('industry_'+domain, 'industry_table', text[start:end],
                domain=domain, atomic=True, start=start, end=end, offset_basis='별표단위[별표번호=0002].별표내용',
                source_locator='별표 Ⅱ 품목별 해결기준 / '+key,
                warnings=['공식 XML의 문자 표 전체 보존. 셀 값·비고를 추정하지 않음. PDF 별표와 대조 필요']))
            domains.add(domain)
    if domains != set(targets.values()):
        raise ValueError('필수 업종 별표 구간 누락')
    # 이 품목들은 <업종>이 아니라 문자 표 안의 원형 번호 제목으로 구분된다.
    product_headings=list(re.finditer(r'(?m)^┃[ \t]*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳][^│\n]*┃[ \t]*$',text))
    for identifier,key,title in [('jewelry','귀금속','귀금속·보석'),
                                  ('electronics_office','전자제품','전자제품·사무용기기')]:
        selected=[i for i,h in enumerate(product_headings) if key in h.group()]
        if not selected or selected != list(range(selected[0],selected[-1]+1)):
            raise ValueError(f'{title} 표 제목 누락 또는 비연속 구간')
        first,last=selected[0],selected[-1]
        # 위·아래 표 테두리, 연속 페이지 표, 비고를 모두 포함한다.
        start=text.rfind('┏',0,product_headings[first].start())
        end=(text.rfind('┏',product_headings[last].end(),product_headings[last+1].start())
             if last+1<len(product_headings) else len(text))
        if start<0 or end<=product_headings[last].end() or '┛' not in text[start:end]:
            raise ValueError(f'{title} 표 경계 불완전')
        units.append(unit('product_'+identifier,'product_table',text[start:end],domain='consumer',
            atomic=True,product_scope=title,appendix_number='0002',start=start,end=end,
            offset_basis='별표단위[별표번호=0002].별표내용',source_locator='별표 Ⅱ 품목별 해결기준 / '+title,
            warnings=['선정 품목에 한정된 기준. 다른 품목·개인 간 거래에 일괄 적용하지 않음.',
                      '공식 XML 문자 표의 조건·비고·연속 표 전체 보존. 문자 이상·병합 관계는 공식 별표 대조 필요.']))
    # 문자 표의 병합 열·공통 조건을 추정하거나 중간에서 잘라내지 않는다.
    for number, title in [('0003', '품목별 품질보증기간 및 부품보유기간'),
                          ('0004', '품목별 내용연수표')]:
        matches = root.findall(f'.//별표단위[별표번호="{number}"]')
        if len(matches) != 1:
            raise ValueError(f'별표 {number} 누락 또는 중복')
        appendix = matches[0]
        body = appendix.findtext('별표내용') or ''
        if appendix.findtext('별표제목') != title or not body.strip():
            raise ValueError(f'별표 {number} 제목 불일치 또는 본문 누락')
        units.append(unit('appendix_'+number, 'appendix_table', body,
            domain='consumer', atomic=True, appendix_number=number,
            appendix_title=title, start=0, end=len(body),
            offset_basis=f'별표단위[별표번호={number}].별표내용',
            source_locator=f'별표 {number} / {title}',
            warnings=['공식 XML 문자 표 전체 보존. 품목·기간·비고·공통 조건을 분리하거나 셀 관계를 추정하지 않음.',
                      'XML의 물음표 등 문자 이상 가능성 및 표의 병합 관계는 공식 별표 파일과 대조 필요. 수치 자동 계산 근거로 사용하기 전 확인.',
                      '표 전체를 하나의 청크로 보존하므로 임베딩 모델 토큰 제한 확인 필요.']))
    meta = {n.tag:n.text for n in info}
    return units, meta


def reviewed_pdf_table(page, layout):
    """검토된 좌표로만 표를 읽는다. 빈 값·병합 관계를 텍스트 순서로 추정하지 않는다."""
    xs,ys=layout['column_boundaries'],layout['row_boundaries']
    if len(xs)<3 or len(ys)<3 or xs!=sorted(set(xs)) or ys!=sorted(set(ys)):
        raise ValueError('검토 PDF 표의 행·열 경계 불일치')
    if layout['table_bbox'] != [xs[0],ys[0],xs[-1],ys[-1]]:
        raise ValueError('검토 PDF 표 범위 불일치')
    physical_tables=page.find_tables()
    if len(physical_tables)!=1 or [[round(v,3) for v in cell] for cell in physical_tables[0].cells]!=layout['physical_cells']:
        raise ValueError('검토 PDF의 실물 표 격자 변경')
    def region_text(bbox, outside=False):
        def selected(obj):
            if obj.get('object_type')!='char':
                return False
            x,y=(obj['x0']+obj['x1'])/2,(obj['top']+obj['bottom'])/2
            inside=bbox[0]<=x<bbox[2] and bbox[1]<=y<bbox[3]
            return not inside if outside else inside
        return page.filter(selected).extract_text() or ''
    grid={};cells=[]
    for raw in layout['logical_cells']:
        row,col=raw['row_index'],raw['column_index']
        height,width=raw['row_span'],raw['column_span']
        if (any(type(n) is not int for n in [row,col,height,width]) or row<0 or col<0 or height<1 or width<1
                or row+height>=len(ys) or col+width>=len(xs)):
            raise ValueError('검토 PDF 셀 범위 불일치')
        bbox=[xs[col],ys[row],xs[col+width],ys[row+height]]
        if raw['bbox']!=bbox:
            raise ValueError('검토 PDF 셀 좌표·병합 범위 불일치')
        text=region_text(bbox)
        if text!=raw['reviewed_text'] or digest(text.encode('utf-8'))!=raw['text_sha256']:
            raise ValueError('검토 후 PDF 셀 본문 변경 또는 좌표 불일치')
        index=len(cells)
        for r in range(row,row+height):
            for c in range(col,col+width):
                if (r,c) in grid:
                    raise ValueError('검토 PDF 병합 셀 중복')
                grid[r,c]=index
        cells.append({k:v for k,v in raw.items() if k not in {'reviewed_text','text_sha256'}} | {'text':text})
    expected={(r,c) for r in range(len(ys)-1) for c in range(len(xs)-1)}
    if set(grid)!=expected:
        raise ValueError('검토 PDF 표 셀 누락')
    outside=region_text(layout['table_bbox'],outside=True)
    if outside!=layout['outside_text'] or digest(outside.encode('utf-8'))!=layout['outside_text_sha256']:
        raise ValueError('검토 PDF 공통기준·표 외 본문 변경')
    original=Counter(normalized_text(''.join(c['text'] for c in page.chars)))
    extracted=Counter(normalized_text(outside+''.join(c['text'] for c in cells)))
    if original!=extracted:
        raise ValueError('검토 PDF 페이지 문자 누락 또는 중복')
    by_id={c['cell_id']:i for i,c in enumerate(cells)}
    if len(by_id)!=len(cells):
        raise ValueError('검토 PDF 셀 ID 중복')
    rows=[]
    for r in range(len(ys)-1):
        context_ids=layout.get('row_context_cell_ids',{}).get(str(r),[])
        if any(i not in by_id for i in context_ids):
            raise ValueError('검토 PDF 공통 조건 셀 누락')
        rows.append({'row_index':r,'column_cell_indices':[grid[r,c] for c in range(len(xs)-1)],
                     'context_cell_indices':[by_id[i] for i in context_ids]})
    return {'page_number':layout['page_number'],'bbox':layout['table_bbox'],
            'physical_cells':layout['physical_cells'],'column_boundaries':xs,'row_boundaries':ys,
            'cells':cells,'rows':rows,'outside_text':outside,'character_inventory_match':True,
            'derivation':'reviewed_pdf_coordinates_and_spans'}


def apply_reviewed_consumer_appendices(source, units, review=None, *, context=None):
    context = context if context is not None else ProcessingContext()
    import pdfplumber
    record_bytes=context.appendix_review_path.read_bytes()
    review=json.loads(record_bytes) if review is None else review
    if (review['parent_source_id']!=source['source_id'] or review['parent_raw_file']!=source['raw_file']
            or review['parent_raw_sha256']!=source['raw_sha256']):
        raise ValueError('별표 검토 기록의 상위 XML 원본 지문 불일치')
    xml=ET.fromstring((context.project_root/source['raw_file']).read_bytes())
    if review['effective_date']!=xml.findtext('.//시행일자'):
        raise ValueError('별표 PDF 검토와 상위 고시 시행본 불일치')
    if {a['appendix_number'] for a in review['appendices']}!={'0003','0004'} or len(review['appendices'])!=2:
        raise ValueError('검토 별표 목록 누락 또는 중복')
    result=[]
    for original in units:
        if original['unit_id'] not in {'appendix_0003','appendix_0004'}:
            result.append(original)
            continue
        number=original['appendix_number']
        spec=next(a for a in review['appendices'] if a['appendix_number']==number)
        node=xml.find(f'.//별표단위[별표번호="{number}"]')
        link=node.findtext('별표서식PDF파일링크')
        if original['appendix_title']!=spec['title'] or spec['pdf_source_url']!='https://www.law.go.kr'+link:
            raise ValueError('별표 PDF 제목·원문 XML 첨부 링크 불일치')
        path=(context.project_root/spec['pdf_file']).resolve()
        if not path.is_relative_to(context.raw_dir.resolve()) or not path.is_file():
            raise ValueError('검토 별표 PDF 경로 불일치')
        if digest(path.read_bytes())!=spec['pdf_sha256']:
            raise ValueError('검토 별표 PDF 원본 변경')
        tables=[]
        with pdfplumber.open(path) as pdf:
            if [p['page_number'] for p in spec['pages']]!=list(range(1,len(pdf.pages)+1)):
                raise ValueError('검토 별표 PDF 페이지 누락 또는 중복')
            for page,layout in zip(pdf.pages,spec['pages']):
                tables.append(reviewed_pdf_table(page,layout))
        if normalized_text(spec['title']) not in normalized_text(''.join(t['outside_text'] for t in tables)):
            raise ValueError('검토 PDF 별표 제목 누락')
        pieces=[]
        labels=['품목','품질보증기간','부품보유기간'] if number=='0003' else ['품목','내용연수']
        for table in tables:
            page_number=table['page_number']
            pieces.append(f'[공식 PDF 페이지 {page_number} 공통기준]\n'+table['outside_text'])
            for cell in table['cells']:
                label=f"[페이지 {page_number} / 행 {cell['row_index']} / 열 {labels[cell['column_index']]} / 셀 {cell['cell_id']} / 행 병합 {cell['row_span']}]"
                cell['rendered_start']=len('\n'.join(pieces))+(1 if pieces else 0)
                block=label+'\n'+cell['text']
                pieces.append(block)
                cell['rendered_end']=cell['rendered_start']+len(block)
        text='\n'.join(pieces)
        original_text=original['text']
        result.append({**original,'text':text,'start':0,'end':len(text),
            'offset_basis':'reviewed_pdf_table_text','source_locator':f'공식 PDF 별표 {number} / 검토 좌표·병합 범위 / 전체 {len(tables)}페이지',
            'source_xml_text':original_text,'source_xml_text_sha256':digest(original_text.encode('utf-8')),
            'source_xml_offset_basis':original['offset_basis'],'source_xml_start':original['start'],'source_xml_end':original['end'],
            'pdf_file':spec['pdf_file'],'pdf_sha256':spec['pdf_sha256'],'pdf_source_url':spec['pdf_source_url'],
            'pdf_source_id':spec['pdf_source_id'],'pdf_tables':tables,'page_numbers':[t['page_number'] for t in tables],
            'review_record_file':context.appendix_review_path.relative_to(context.project_root).as_posix(),'review_record_sha256':digest(record_bytes),
            **({'required_context_chunk_ids':['consumer_dispute_standard_appendix_0003_0000']} if number=='0004' else {}),
            'source_comparison':{'status':'reviewed_against_collected_official_pdf','reviewed_on':review['reviewed_on'],
                'source_effective_date':review['effective_date'],'xml_question_mark_count':original_text.count('?'),
                'pdf_question_mark_count':sum(c['text'].count('?') for t in tables for c in t['cells']),
                'xml_preserved':True,'scope':review['scope'],'layout_policy':review['layout_policy']},
            'warnings':['수집된 공식 PDF의 문자·좌표·병합 관계를 대조한 별표. XML 원문은 별도로 보존하고 물음표를 일괄 치환하지 않음.',
                        '논리 행은 시각 대조한 좌표 기준이며 빈 셀은 미기재로 보존. 품질보증기간·부품보유기간·내용연수는 서로 다른 기준.',
                        '표 전체를 하나의 청크로 보존. 개별 계약·제품에 대한 적용과 최신 개정 여부 및 임베딩 모델 토큰 제한은 별도 확인.']})
    return result


def refund_table_rows(cells):
    """환급 표의 실제 주소·병합 범위만 펼친다. 수식이나 법률 조건을 계산하지 않는다."""
    grid={}
    for index,cell in enumerate(cells):
        try:
            row,col=int(cell['address']['rowAddr']),int(cell['address']['colAddr'])
            height,width=int(cell['span']['rowSpan']),int(cell['span']['colSpan'])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('환급 표 셀 주소·병합 범위 누락') from exc
        if row<0 or col<0 or height<=0 or width<=0 or row+height>1000 or col+width>3:
            raise ValueError('환급 표 셀 범위 불일치')
        for r in range(row,row+height):
            for c in range(col,col+width):
                if (r,c) in grid:
                    raise ValueError('환급 표 셀 범위 중복')
                grid[r,c]=index
    if not grid or any((0,c) not in grid for c in range(3)):
        raise ValueError('환급 표 제목 행 누락')
    if '환급' not in cells[grid[0,2]]['text']:
        raise ValueError('환급 기준 열 확인 불가')
    common=[i for i,c in enumerate(cells) if int(c['span']['colSpan'])==3]
    rows=[]
    for r in sorted({r for r,_ in grid if r>0}):
        if any((r,c) not in grid for c in range(3)):
            raise ValueError('환급 표 조건·기준 셀 누락')
        indices=[grid[r,c] for c in range(3)]
        if len(set(indices))==1:
            continue
        if len(set(indices))!=3:
            raise ValueError('지원하지 않는 환급 표 행 구조')
        rows.append({'row_index':r,'column_cell_indices':indices,
                     'condition_texts':[cells[i]['text'] for i in indices[:2]],
                     'refund_text':cells[indices[2]]['text'],
                     'common_cell_indices':common,'common_texts':[cells[i]['text'] for i in common]})
    if not rows:
        raise ValueError('환급 표 데이터 행 누락')
    return rows


def attach_refund_tables(units):
    marker='[표 셀: address는 행·열, span은 병합 범위]\n'
    for u in units:
        cursor=0;tables=[]
        while marker in u['text'][cursor:]:
            start=u['text'].index(marker,cursor)+len(marker)
            cells,length=json.JSONDecoder().raw_decode(u['text'][start:]);end=start+length
            # 환급 표만 대상으로 하며 다른 표는 원래 구조를 그대로 보존한다.
            headers=[c['text'] for c in cells if c.get('address',{}).get('rowAddr')=='0']
            if any('환급' in h for h in headers):
                tables.append({'start':start,'end':end,'offset_basis':'unit.text',
                               'cells':cells,'rows':refund_table_rows(cells),
                               'derivation':'hwpx_cell_addresses_and_spans',
                               'context_before':u['text'][:start],'context_after':u['text'][end:]})
            cursor=end
        if tables:
            u['refund_tables']=tables
    return units


def extract_hwpx(data_path):
    def cell_text(element):
        pieces=[]
        for child in element.iter():
            if child.tag.endswith('}t'):
                pieces.append(child.text or '')
            elif child.tag.endswith('}script'):
                # 수식을 해석·계산하지 않고 HWP 수식 원문을 보존한다.
                pieces.append('〔수식 원문: '+(child.text or '')+'〕')
        return clean('\n'.join(pieces))

    def render(element):
        if element.tag.endswith('}tbl'):
            cells=[]
            for cell in element.iter():
                if cell.tag.endswith('}tc'):
                    addr=next((n.attrib for n in cell if n.tag.endswith('}cellAddr')), {})
                    span=next((n.attrib for n in cell if n.tag.endswith('}cellSpan')), {})
                    cells.append({'address':addr,'span':span,'text':cell_text(cell)})
            tables.append(cells)
            return '\n[표 셀: address는 행·열, span은 병합 범위]\n'+json.dumps(cells,ensure_ascii=False)+'\n'
        if element.tag.endswith('}t'):
            return element.text or ''
        if element.tag.endswith('}script'):
            return '〔수식 원문: '+(element.text or '')+'〕'
        return ''.join(render(child) for child in element)

    paragraphs, tables = [], []
    with zipfile.ZipFile(data_path) as archive:
        names = sorted(n for n in archive.namelist() if re.fullmatch(r'Contents/section\d+\.xml',n))
        if not names:
            raise ValueError('HWPX 본문 없음')
        for name in names:
            root = ET.fromstring(archive.read(name))
            for paragraph in root:
                if not paragraph.tag.endswith('}p'):
                    continue
                # hp:t만 읽는다. 이미지 대체 설명·바이너리·서식 정보는 본문으로 합치지 않는다.
                text = clean(render(paragraph))
                if text:
                    paragraphs.append(text)
    text='\n'.join(paragraphs)
    if '제1조' not in text or '환급' not in text:
        raise ValueError('표준약관 본문 불완전')
    boundaries=list(re.finditer(r'(?m)^제\d+조[\s(（]',text))
    units=[]
    if boundaries and boundaries[0].start()>0:
        units.append(unit('preamble','terms_preamble',text[:boundaries[0].start()],atomic=True))
    for i, match in enumerate(boundaries):
        end=boundaries[i+1].start() if i+1<len(boundaries) else len(text)
        units.append(unit(f'terms_article_{i}','standard_terms_article',text[match.start():end],
                          atomic=True,start=match.start(),end=end,offset_basis='hwpx_extracted_text'))
    return text,attach_refund_tables(units),tables


def extract_supporting_admin(source, data, *, context=None):
    """추가 고시의 전체 조문·별표를 보존하며 적용 범위를 확대 해석하지 않는다."""
    context = context if context is not None else ProcessingContext()
    root=ET.fromstring(data)
    info=root.find('행정규칙기본정보')
    if info is None or info.findtext('행정규칙명')!=source['title']:
        raise ValueError('추가 고시의 요청명·본문명 불일치')
    meta={n.tag:n.text or '' for n in info}
    if meta.get('시행일자')!=source.get('expected_effective_date',meta.get('시행일자')):
        raise ValueError('추가 고시의 적용 연도 불일치')
    units=[unit(f'article_{i}','general_rule',n.text or '',atomic=True,
                source_locator=f'조문내용[{i+1}]')
           for i,n in enumerate(root.findall('조문내용')) if (n.text or '').strip()]
    for i,n in enumerate(root.findall('.//별표단위')):
        text=n.findtext('별표내용') or ''
        if not text.strip():
            raise ValueError('추가 고시 별표 본문 누락')
        units.append(unit(f'appendix_{i}','appendix_table',text,atomic=True,
            appendix_number=n.findtext('별표번호'),appendix_title=n.findtext('별표제목'),
            start=0,end=len(text),offset_basis=f'별표단위[{i+1}].별표내용',
            source_locator=f'별표[{i+1}] '+(n.findtext('별표제목') or ''),
            warnings=['XML 문자 표 전체 보존. ? 등 원문 문자를 보정하지 않으며 셀 관계·원문 대조 필요.']))
    if source['source_id']=='minimum_wage_2026':
        import pdfplumber
        path=context.project_root/source['body_pdf_file']
        if digest(path.read_bytes())!=source['body_pdf_sha256']:
            raise ValueError('고시 본문 PDF 지문 불일치')
        with pdfplumber.open(path) as pdf:
            pages=[{'page_number':i+1,'text':p.extract_text() or '', 'tables':p.extract_tables()}
                   for i,p in enumerate(pdf.pages)]
        text='\n'.join(p['text'] for p in pages)
        if not all(t in text for t in [source['title'],'10,320원','2026. 1. 1. ~ 2026. 12. 31.','209시간','주 소정근로 40시간']):
            raise ValueError('2026년 고시 금액·적용 기간·환산 조건 확인 실패')
        if units:
            raise ValueError('API 본문이 추가됨: PDF와 재대조 필요')
        units=[unit('notice_body','notice_body',text,atomic=True,source_locator='공식 고시 첨부 PDF 전문',
                    page_numbers=[p['page_number'] for p in pages])]
        extra={'derived_pdf_file':source['body_pdf_file'],'derived_pdf_sha256':source['body_pdf_sha256'],
               'pdf_pages':pages,'valid_from':'2026-01-01','valid_until':'2026-12-31',
               'body_source_url':source['body_pdf_url'],'body_review':'공식 PDF 1페이지 표·본문 시각 대조 완료'}
    else:
        extra={}
    if not units:
        raise ValueError('추가 고시의 본문·별표가 모두 비어 있음')
    extra['additional_materials']={n.tag:ET.tostring(n,encoding='unicode') for n in root
                                 if n.tag not in ['행정규칙기본정보','조문내용','별표']}
    extra['additional_materials_policy']='개정별 부칙·이유·첨부 목록은 정제에 보존. 과거 적용을 확정하지 않아 기본 검색 제외.'
    return units,meta,extra


CIVIL_RECOVERY_GUIDES = {
    'easylaw_consumer_small_claim': [
        ('scope','소액사건심판의 대상',1), ('filing','소액사건심판의 제기',1),
        ('recommendation','피고에 대한 이행 권고',1), ('hearing','변론',1), ('judgment','판결의 선고',1)],
    'easylaw_consumer_payment_order': [
        ('definition','지급명령(독촉절차)',1), ('filing','지급명령의 신청',1),
        ('service','지급명령의 발령',1), ('objection_finality','지급명령의 확정',1)],
    'easylaw_payment_order_application': [
        ('application','지급명령 신청서 제출',1), ('service','송달',1),
        ('objection','이의신청',1), ('litigation','소송의 제기',1), ('stamp_correction','인지 등의 보정',1)],
    'easylaw_civil_enforcement': [
        ('execution_copy','집행문의 부여',2), ('property_disclosure','재산명시신청',1),
        ('property_inquiry','재산조회',1), ('movables_claims','유체동산·채권에 대한 강제집행',1)]}


HOUSING_GUIDES = {
    'easylaw_housing_termination': [
        ('return','임차주택의 반환 및 임차보증금의 반환',1),
        ('registration_move','임차권등기명령신청권의 취득',1)],
    'easylaw_housing_restoration': [
        ('care','임차주택의 사용·수익에 따른 의무',1),
        ('restoration','임차주택의 반환의무 및 원상회복의무',1)],
    'easylaw_housing_distribution': [
        ('definition','배당요구',1), ('creditors','배당요구를 할 수 있는 채권자',1),
        ('deadline_exceptions','배당요구의 시기 및 종기',1), ('application','배당요구의 신청',1),
        ('tenant_share_exceptions','임차인의 배당액',1),
        ('maintain_priority','대항력과 우선변제권의 유지',1),
        ('paid','임차인이 보증금 전액을 배당받은 경우',1),
        ('unpaid','임차인이 보증금 전액을 배당받지 못한 경우',1),
        ('delivery','임차주택의 인도',1)]}


def extract_civil_recovery_guide(source, data, *, context=None):
    """검토한 소제목 단위로 조건·예외를 묶고 안내 기준일과 개정 공지를 보존한다."""
    context = context if context is not None else ProcessingContext()
    identifier=source['source_id']
    configuration = HOUSING_GUIDES if identifier in HOUSING_GUIDES else CIVIL_RECOVERY_GUIDES
    if identifier not in configuration:
        raise ValueError('검토하지 않은 민사 회수 안내 ID')
    tree=HTMLTree(decode_html(data)).root
    clean=lambda n:' '.join(n.text().split())
    bodies=tree.find(lambda n:'ovDivbox' in n.attrs.get('class','').split())
    wrappers=tree.find(lambda n:'ovDivbox1' in n.attrs.get('class','').split())
    if len(bodies)!=1 or len(wrappers)!=1:
        raise ValueError('민사 회수 안내 본문 누락·중복 또는 차단 화면')
    match=re.search(r'이 정보는 (\d{4})년 (\d{1,2})월 (\d{1,2})일 기준으로 작성',clean(wrappers[0]))
    if not match:
        raise ValueError('공식 안내 작성 기준일 누락')
    as_of='-'.join([match[1],match[2].zfill(2),match[3].zfill(2)])
    alerts=[clean(n) for n in tree.find(lambda n:n.tag=='a' and '업데이트 예정' in clean(n))]
    warnings=['공식 생활법령 해설이며 법령 원문·판례와 구분. 안내 기준일 이후 개정 및 계약 당사자·환급청구액·증빙·송달주소 확인 필요.',
              '헬스장·필라테스 현금·일시불·이미 지급한 이용료 환급 청구의 민사절차 참고용. 카드사 잔여 할부금 지급 거절과 구분. 승소·확정이 실제 회수를 보장하지 않으며 집행 가능한 채무자 재산·권리 순위·도산 여부를 별도 확인.']
    if identifier in HOUSING_GUIDES:
        warnings = [
            '공식 주택임대차 해설이며 법령 원문·판례와 구분. 안내 작성 기준일 이후 개정과 계약·인도·등기·송달 시점을 별도 확인.',
            '보증금 반환 지연손해금은 계약 종료만으로 발생·이율을 일률 확정하지 않음. 주택 인도·이행 제공, 약정, 금전채무 지체 및 소장 등 송달과 소송촉진법 제3조의 예외를 함께 확인.',
            '원상회복비 공제는 계약·특약, 입주 당시 상태, 통상손모와 귀책 훼손, 공사·견적·영수증 등 증빙에 따라 개별 판단. 배당요구 예외와 대항요건 유지·권리 순위는 구분하며 배당액·전액 회수를 보장하지 않음.']
    if alerts:
        warnings.append('원문 개정 검토 공지: '+' / '.join(alerts)+' 현행 법령 대조 전 자동 기한·비용 계산에 사용하지 않음.')
    sections={}
    current=None
    for node in bodies[0].children:
        if not hasattr(node,'tag'):
            continue
        classes=node.attrs.get('class','').split()
        if any(c.startswith('plv1') for c in classes):
            current=None
        if 'plv2a' in classes:
            current=[]
            sections.setdefault(clean(node),[]).append(current)
        if current is not None and any(c.startswith('plv') for c in classes):
            current.append(node)
    units=[]
    for key,heading,count in configuration[identifier]:
        found=sections.get(heading,[])
        if len(found)!=count:
            raise ValueError('민사 회수 필수 소제목 누락·중복: '+heading)
        nodes=[n for group in found for n in group]
        text='\n'.join(clean(n) for n in nodes if clean(n))
        if not text or all(clean(n)==heading for n in nodes):
            raise ValueError('민사 회수 소제목 본문 누락: '+heading)
        units.append({'unit_id':key,'role':('housing_' if identifier in HOUSING_GUIDES else 'civil_recovery_')+key,'text':text,'atomic':True,
                      'source_locator':'.ovDivbox > '+heading,
                      'source_node_ids':[n.attrs['id'] for n in nodes if n.attrs.get('id')],
                      'warnings':[]})
    checks={
        'easylaw_consumer_small_claim':['3,000만원','분할','인정되지','2주','이의신청','확정판결'],
        'easylaw_consumer_payment_order':['주소보정','외국','2주','이의신청한 범위','강제집행'],
        'easylaw_payment_order_application':['청구 취지와 원인','법인','보정명령','공시송달','2주','각하'],
        'easylaw_civil_enforcement':['집행문을 부여받지 않아도','다만','승계','재산목록','부족한 경우','추심명령·전부명령'],
        'easylaw_housing_termination':['동시이행항변권','다만','등기를 마치면','지연손해금'],
        'easylaw_housing_restoration':['다만','구체적·개별적으로','임대 당시','제615조','제654조'],
        'easylaw_housing_distribution':['종기','배당요구를 하지 않아도','서면','경매개시결정 전에','경매개시결정 후에','대항력과 우선변제권','동시이행관계']}
    joined=' '.join(u['text'] for u in units)
    if not all(term in joined for term in checks[identifier]):
        raise ValueError('민사 회수 조건·예외 누락: '+identifier)
    included={h for _,h,_ in configuration[identifier]}
    if identifier in HOUSING_GUIDES:
        context = housing_law_context(identifier, context=context)
        for unit in units:
            unit['required_context_chunk_ids'] = context
        if identifier == 'easylaw_housing_distribution':
            warning = ('원문 인용 조문 불일치: 안내의 민사집행규칙 제91조는 수집 현행본에서 강제관리의 수익 처리 조문이고, '
                       '민사집행법 제148조 제3호는 선등기 가압류채권자이다. 원문 문자는 보존하되 경매 배당 신청의 직접 근거로 해당 인용을 사용하지 않음. '
                       '종기·서면은 법 제84·88조와 규칙 제48조, 배당 참가 범위는 법 제148조 각 호를 대조. 임차권등기 예외의 개별 적용은 추가 확인.')
            warnings.append(warning)
            for unit in units:
                if unit['unit_id'] in {'deadline_exceptions','tenant_share_exceptions'}:
                    unit['warnings'].append(warning)
    return {'units':units,'information_as_of':as_of,'source_update_notices':alerts,
            'source_date_note':'안내 작성 기준일만 보존. 게시일·법령 시행일로 대체하지 않음. 개정 검토 공지는 경고에 보존.',
            'scope_included':[u['source_locator'] for u in units],
            'scope_excluded':[{'scope':'메뉴·푸터·첨부·외부 접수 화면·이미지 흐름도·tplv 외부참조/전자소송 안내',
                               'reason':'검토한 HTML 문자 본문만 검색. 이미지 alt를 절차 본문으로 대체하지 않음'},
                              {'scope':sorted(set(sections)-included),
                               'reason':('주택 반환·원상회복·배당 신청 범위 밖 소제목. 배당 순위 표의 조세 항목·순위는 현행 조세법 대조 전 검색에서 제외; 원본 보존. 개별 배당액 계산에 사용하지 않음.' if identifier in HOUSING_GUIDES else '이번 민사 회수 범위 밖 소제목. 원본 HTML 보존; 대여금 사례·가압류를 환급 집행권원으로 임의 확대하지 않음')}],
            'citation_discrepancies': ([
                {'unit_ids':['deadline_exceptions','tenant_share_exceptions'],
                 'source_citation':'민사집행규칙 제91조 제1항', 'current_article_title':'수익의 처리',
                 'review_note':'강제관리 수익 배당 절차와 강제경매 절차를 구분. 종기 법 제84조 및 서면 규칙 제48조 연결; 인용을 임의 수정하지 않음.'},
                {'unit_ids':['tenant_share_exceptions'], 'source_citation':'민사집행법 제148조 제3호',
                 'current_article_text':'첫 경매개시결정등기전에 등기된 가압류채권자',
                 'review_note':'같은 조 제4호의 선등기·매각으로 소멸하는 우선변제권과 구분. 안내 인용을 그대로 적용하지 않음.'}]
                if identifier=='easylaw_housing_distribution' else []),
            'warnings':warnings}


def housing_law_context(identifier, *, context=None):
    """검토한 현행 조문을 연결한다. 계약 당시 버전으로 간주하지 않는다."""
    context = context if context is not None else ProcessingContext()
    scope = {
        'easylaw_housing_termination': {
            '민법':{'제379조','제387조','제397조','제460조','제536조'},
            '주택임대차보호법':{'제3조의2','제3조의3','제4조'},
            '소송촉진 등에 관한 특례법':{'제3조'},
            '소송촉진 등에 관한 특례법 제3조제1항 본문의 법정이율에 관한 규정':{'전문'}},
        'easylaw_housing_restoration': {'민법':{'제374조','제390조','제393조','제615조','제634조','제654조'}},
        'easylaw_housing_distribution': {
            '민사집행법':{'제84조','제88조','제145조','제148조'},
            '민사집행규칙':{'제48조'},
            '주택임대차보호법':{'제3조의2','제3조의3','제3조의5','제8조'},'민법':{'제536조'}}}[identifier]
    records = _law_records(context)
    result=[]
    for name,labels in scope.items():
        found=[c for c in records if c.get('law_name')==name and c.get('article_label') in labels]
        if {c['article_label'] for c in found} != labels or len({c['document_id'] for c in found}) != 1:
            raise ValueError('주택임대차 연결 조문 누락 또는 버전 혼합: '+name)
        result.extend(c['chunk_id'] for c in found)
    return result


FITNESS_CLOSURE_GUIDES={'kca_fitness_closure_gym','kca_fitness_closure_pilates','kca_damage_redress_procedure'}

CONSUMER_REDRESS_GUIDES={'kca_redress_requirements','kca_redress_submission','kca_mediation_procedure'}


def consumer_redress_context(identifier, *, context=None):
    """피해구제와 조정의 서로 다른 기간·효력을 현행 조문에 연결한다."""
    context = context if context is not None else ProcessingContext()
    labels=({'제55조','제57조','제58조','제61조','제63조','제65조','제66조','제67조'}
            if identifier=='kca_mediation_procedure' else {'제55조','제57조','제58조'})
    records = _law_records(context)
    result=[]
    for name,expected in [('소비자기본법',labels),('소비자기본법 시행령',{'제44조'})]:
        found=[c for c in records if c.get('law_name')==name and c.get('article_label') in expected]
        if {c['article_label'] for c in found}!=expected or len({c['document_id'] for c in found})!=1:
            raise ValueError('소비자 피해구제 연결 법령 누락 또는 버전 혼합: '+name)
        result.extend(c['chunk_id'] for c in found)
    return result


def extract_consumer_redress_guide(source,data, *, context=None):
    context = context if context is not None else ProcessingContext()
    tree=HTMLTree(decode_html(data)).root
    clean=lambda node:' '.join(node.text().split())
    identifier=source['source_id']
    if identifier not in CONSUMER_REDRESS_GUIDES:
        raise ValueError('검토하지 않은 소비자 절차 안내')
    units=[]
    def one(nodes,label):
        if len(nodes)!=1:
            raise ValueError('소비자 절차 필수 영역 누락 또는 중복: '+label)
        return nodes[0]
    def section(title):
        h=one(tree.find(lambda n:n.tag=='h4' and clean(n)==title),title)
        parent=one(tree.find(lambda n:h in n.children),title+' 부모')
        return one([n for n in parent.children if getattr(n,'tag',None) in {'ul','div'} and
                    (n.tag=='ul' or 'cnsut_box' in n.attrs.get('class','').split())],title+' 본문')
    def add(key,node,role,locator,required,warning=None):
        text=clean(node)
        if not text or not all(phrase in text for phrase in required):
            raise ValueError('소비자 절차 조건·예외 누락: '+key)
        units.append({'unit_id':key,'role':role,'text':text,'source_locator':locator,'atomic':True,
                      'warnings':[warning] if warning else []})
    if identifier=='kca_redress_requirements':
        add('eligibility',section('피해구제 접수 해당 요건'),'redress_eligibility',
            'h4: 피해구제 접수 해당 요건 > .cnsut_box',
            ['사업자가 제공하는','입증서류 미제출','개인간 거래','소재파악','법원에 소송'])
        heading=one(tree.find(lambda n:n.tag=='h4' and clean(n)=='피해구제 접수 신청서 양식'),'신청서 양식 제목')
        parent=one(tree.find(lambda n:heading in n.children),'양식 제목 부모')
        children=[n for n in parent.children if hasattr(n,'tag')]
        index=children.index(heading)
        note=children[index+1] if index+1<len(children) else None
        if note is None or note.tag!='div' or 'mt10' not in note.attrs.get('class','').split():
            raise ValueError('전문분야 양식 제출 조건 구조 변경')
        add('specialized_forms',note,'specialized_application_forms','h4: 피해구제 접수 신청서 양식 + div.mt10',
            ['일반분야를 제외한','전문분야','필수제출 서류','접수불가능'],
            '이 필수제출 조건은 원문이 명시한 전문분야에 관한 안내. 일반분야의 모든 신청에 동일하게 확대하지 않음. 첨부 양식 자체는 미변환·미검토.')
    elif identifier=='kca_redress_submission':
        add('evidence',section('피해구제 접수 안내'),'application_evidence','h4: 피해구제 접수 안내 > .cnsut_box',
            ['계약서','영수증','이의를 제기한','전자문서','화면 캡처'])
        for label,key in [('온라인신청','online'),('방문신청','visit'),('우편신청','postal')]:
            box=one(tree.find(lambda n:'step_txtbx' in n.attrs.get('class','').split() and
                             any(getattr(c,'tag',None)=='p' and clean(c)==label for c in n.children)),label)
            required={'online':['소비자상담','안내받은 분','6개월 이내'],
                      'visit':['증빙서류를 지참','직접 방문'], 'postal':['증빙서류를 첨부','주소']}[key]
            add(key,box,'submission_method','.step_txtbx > p.stepnum: '+label,required,
                '6개월은 이 페이지의 온라인 신청 이용 조건. 일반적인 청구권 소멸시효 또는 모든 접수 방법의 기한으로 해석하지 않음.' if key=='online' else
                '방문·우편 주소와 접수 기능은 수집 후 변경될 수 있으므로 신청 전 원문 페이지에서 확인.')
        address=one(tree.find(lambda n:n.tag=='ul' and 'help_txt' in n.attrs.get('class','').split() and
                             '일반우편 이용시 분실의 위험' in clean(n)),'우편 주소·품목별 예외')
        add('postal_address',address,'submission_address','.help_txt: 우편 접수 주소·섬유/신발 예외',
            ['등기우편','섬유','신발'], '원문 수집 시점의 주소. 최신 접수 주소는 원문에서 재확인.')
    else:
        add('definition',section('분쟁조정이란?'),'mediation_definition','h4: 분쟁조정이란? > ul',
            ['양 당사자의 수락','재판상 화해','강제집행'],
            '조정 신청만으로 재판상 화해 효력이 생기지 않음. 소비자기본법 제67조의 수락·수락 간주 요건과 통지일을 함께 확인. 적법한 기간 내 불수락이면 양 당사자 수락을 전제한 성립으로 단정하지 않음. 실제 집행은 법원 절차와 채무자 재산에 따라 별도 검토.')
        box=section('분쟁조정 절차 안내')
        steps=box.find(lambda n:n.tag=='li' and 'step_li' in n.attrs.get('class','').split())
        if len(steps)!=6:
            raise ValueError('분쟁조정 6단계 누락 또는 변경')
        scope=[('request',['합의가 이루어지지 않은','제65조']),('commission',['150명','5명']),
               ('investigation',['사실조사','시험검사']),('meeting',['3~11명']),
               ('acceptance',['통지를 받은 날부터 15일','서면','의사표시가 없는','재판상 화해']),
               ('completion',['30일','정당한 사유','연장'])]
        for node,(key,required) in zip(steps,scope):
            add(key,node,'mediation_step','h4: 분쟁조정 절차 안내 > li.step_li: '+key,required)
    links=consumer_redress_context(identifier, context=context)
    for u in units:
        u['required_context_chunk_ids']=links+(['kca_damage_redress_procedure_redress_exclusions_0000']
            if identifier!='kca_mediation_procedure' else [])
    return {'units':units,'information_as_of':None,
            'source_date_note':'페이지에 확인 가능한 작성·수정 기준일이 없어 null로 보존. 수집일을 안내 작성일이나 법령 시행일로 대체하지 않음.',
            'scope_included':[u['source_locator'] for u in units],
            'scope_excluded':[{'scope':'첨부 HWP/DOC 양식·필독사항·작성 예시 및 신청 화면/영상/절차도 이미지',
                               'reason':'링크 안내와 HTML 본문만 검토. 첨부 파일 내용·온라인 신청 기능은 미검증. 원본 HTML 보존'},
                              {'scope':'개인정보 안내·메뉴·푸터 및 알렛츠/티몬/위메프/해피머니 특정 사건 팝업',
                               'reason':'일반 절차 범위 밖. 특정 사건의 제출 조건·마감일을 일반 신청에 적용하지 않음'}],
            'warnings':['공식 소비자 절차 안내이며 법령·판례가 아님. 연결한 소비자기본법·시행령은 수집 시점 현행본이며 안내의 작성 시점은 확인되지 않음. 법령 시행일은 연결 청크에서 확인.',
                        '피해구제 합의권고와 분쟁조정 성립의 효력은 다름. 접수·상담·조정 신청이 환급이나 강제집행을 보장하지 않음.']}


def installment_context(*, context=None):
    """각 요건의 조문을 버전 단위로 연결하고 누락·혼합을 거부한다."""
    context = context if context is not None else ProcessingContext()
    records = _law_records(context)
    scope={'할부거래에 관한 법률':{'제2조','제3조','제4조','제16조'},
           '할부거래에 관한 법률 시행령':{'제2조','제4조','제11조'}}
    selected=[]
    versions={}
    for name,articles in scope.items():
        found=[c for c in records if c.get('law_name')==name and c.get('article_label') in articles]
        if {c.get('article_label') for c in found}!=articles or len({c['document_id'] for c in found})!=1:
            raise ValueError('할부거래법 근거 조문 누락 또는 버전 혼합: '+name)
        selected.extend(found)
        versions[name]={'document_id':found[0]['document_id'],'effective_date':found[0]['dates']['effective_date']}
    return {'required_context_chunk_ids':[c['chunk_id'] for c in selected], 'versions':versions,
            'review_note':'안내 원문의 3개월 할부 표현을 보존하고 법 제2조의 2개월 이상에 걸쳐 3회 이상 지급 요건을 함께 확인. 법 제16조의 사유·신용제공자 통지·당시 미지급 잔여금·서면 효력 및 시행령 제11조의 금액 조건, 적용제외·다른 법률 관계를 연결. 연결본은 수집 시점 현행 법령이며 2024/2025 안내 당시 구법·개별 계약일의 적용을 확정하지 않음.'}


def extract_fitness_closure_guide(source,data, *, context=None):
    context = context if context is not None else ProcessingContext()
    tree=HTMLTree(decode_html(data)).root
    clean=lambda n:' '.join(n.text().split())
    units=[]
    identifier=source['source_id']
    def one(nodes,label):
        if len(nodes)!=1:
            raise ValueError('폐업 공식 안내 필수 영역 누락 또는 중복: '+label)
        return nodes[0]
    def add(key,node,role,locator,warnings=None):
        text=clean(node)
        if not text:
            raise ValueError('폐업 안내 빈 본문: '+key)
        item={'unit_id':key,'role':role,'text':text,'source_locator':locator,'atomic':True,'warnings':warnings or []}
        units.append(item)
        return item
    card_warning='할부항변권은 요건에 맞는 거래의 잔여 할부금 지급 거절에 관한 안내. 이미 낸 이용료의 자동 환급이나 일시불·현금 결제의 동일한 지급 거절을 보장하지 않음. 결제 방식·할부 조건·채무불이행 및 카드사 확인 필요. 연결한 할부거래법·시행령 근거의 기간·횟수·금액·통지·예외 조건을 함께 확인. 개별 계약일·거래에 대한 법률 적용은 별도 검토.'
    publication_date=None
    publication_period=None
    if identifier=='kca_fitness_closure_gym':
        main=one(tree.find(lambda n:n.attrs.get('id')=='view_wrap'),'헬스장 게시 본문')
        sections={}
        current=None
        for node in main.children:
            if not hasattr(node,'tag'):
                continue
            if node.tag=='h5':
                current=clean(node)
                if current in sections:
                    raise ValueError('폐업 안내 제목 중복')
                sections[current]=[]
            elif current and node.tag=='p' and clean(node):
                sections[current].append(node)
        for label,key,role in [('헬스장 ‘폐업’으로 인한 피해도 지속적으로 발생','closure_risk','closure_risk'),
                ('장기 또는 다회 계약은 신중히 결정하고 신용카드 할부 결제 이용','card_evidence','payment_and_evidence')]:
            nodes=sections.get(label,[])
            if not nodes:
                raise ValueError('헬스장 폐업·결제 안내 제목 누락')
            add(key,nodes[0],role,'#view_wrap > h5: '+label,[card_warning] if key=='card_evidence' else [])
        if not all(w in units[1]['text'] for w in ['20만 원 이상','3개월 이상','내용증명','문자']):
            raise ValueError('헬스장 할부 조건·증빙 누락')
        body=one(tree.find(lambda n:n.attrs.get('id')=='contentBody'),'헬스장 게시일')
        match=re.search(r'등록일\s+(\d{4}-\d{2}-\d{2})',clean(body))
        if not match:
            raise ValueError('공식 게시 등록일 누락')
        publication_date=match[1]
    elif identifier=='kca_fitness_closure_pilates':
        main=one(tree.find(lambda n:n.attrs.get('id')=='contents'),'필라테스 기사 본문')
        definition=one(main.find(lambda n:n.tag=='p' and clean(n).startswith('할부항변권 :')),'할부항변권 정의')
        add('card_objection',definition,'card_objection','#contents > p: 할부항변권 정의',[card_warning])
        if not all(w in units[-1]['text'] for w in ['잔여할부금','3개월','20만 원 이상']):
            raise ValueError('할부항변권 잔여금·조건 누락')
        action=one(main.find(lambda n:n.tag=='p' and clean(n).startswith('필라테스 폐업으로 인한 피해를 예방하기 위해서는')),'폐업 증빙·대응')
        add('closure_action',action,'closure_action','#contents > p: 폐업 예방·잔여 횟수·해지 통지',[card_warning])
        if not all(w in units[-1]['text'] for w in ['잔여 횟수','계약해지일','사업자에게 계약해지 의사','카드사에 할부항변권']):
            raise ValueError('폐업 대응·증빙 누락')
        table=one(main.find(lambda n:n.tag=='table'),'계약 단계별 주의사항 표')
        u=add('evidence_table',table,'closure_evidence','#contents > table: 단계별 주의사항',[card_warning])
        cells=[[{'tag':c.tag,'text':clean(c),'rowspan':int(c.attrs.get('rowspan',1)),
                 'colspan':int(c.attrs.get('colspan',1))} for c in row.children if getattr(c,'tag',None) in {'td','th'}]
               for row in table.find(lambda n:n.tag=='tr')]
        if not cells or any(not row or any(not c['text'] or c['rowspan']<1 or c['colspan']<1 for c in row) for row in cells):
            raise ValueError('폐업 안내 표 셀 누락')
        if not all(w in u['text'] for w in ['폐업·연락 두절','내용증명','계약서','이용 일수']):
            raise ValueError('폐업 표 제한·증빙 누락')
        u['official_table']={'rows':cells,'layout_policy':'원문 물리 행·셀 순서와 rowspan/colspan 보존. 병합 셀의 의미를 임의 확장하지 않음.'}
        paragraphs=main.find(lambda n:n.tag=='p' and '1372 소비자상담센터' in clean(n))
        add('consultation',one(paragraphs,'1372 상담'),'consultation','#contents > p: 1372 상담')
        publication_period='2025-05'
    elif identifier=='kca_damage_redress_procedure':
        expected=[('피해구제란?','redress_definition','redress_definition'),
                  ('이런 경우에는 피해구제 대상에서 제외됩니다.','redress_exclusions','redress_exclusions'),
                  ('피해구제 절차 안내','redress_procedure','redress_procedure')]
        for title,key,role in expected:
            heading=one(tree.find(lambda n:n.tag=='h4' and clean(n)==title),title)
            parent=one(tree.find(lambda n:heading in n.children),'제목 부모')
            content=[n for n in parent.children if getattr(n,'tag',None) in {'ul','div'} and
                     ('cnsut_box' in n.attrs.get('class','').split() or n.tag=='ul')]
            add(key,one(content,'피해구제 본문'),role,'h4: '+title)
        if not all(w in units[1]['text'] for w in ['폐업','연락이 불가능','소재파악']):
            raise ValueError('폐업 연락두절 피해구제 제외 조건 누락')
        if not all(w in units[2]['text'] for w in ['먼저 상담','30일','90일','토·일요일','합의권고','소비자분쟁조정위원회']):
            raise ValueError('피해구제 사전 상담·기한·조정 단계 누락')
        limitation='공식 안내는 사업자의 부도·폐업 등으로 연락이 불가능하거나 소재파악이 안 되는 경우를 피해구제 제외 대상으로 명시. 상담·피해구제 신청이 환급이나 강제집행을 보장하지 않음.'
        for u in units:
            if u['unit_id']!='redress_exclusions':
                u['warnings']=[limitation]
                u['required_context_chunk_ids']=[identifier+'_redress_exclusions_0000']
    else:
        raise ValueError('검토하지 않은 폐업 공식 안내 ID')
    legal_basis={}
    if identifier in {'kca_fitness_closure_gym','kca_fitness_closure_pilates'}:
        legal_basis=installment_context(context=context)
        for u in units:
            if u['role'] in {'payment_and_evidence','card_objection','closure_action','closure_evidence'}:
                u['required_context_chunk_ids']=legal_basis['required_context_chunk_ids']
                u['warnings'].append(legal_basis['review_note'])
            elif u['role'] in {'closure_risk','consultation'}:
                u['required_context_chunk_ids']=[name+'_'+key+'_0000'
                    for name,sections in CIVIL_RECOVERY_GUIDES.items() for key,_,_ in sections]
                u['warnings'].append('현금·일시불·기지급 이용료의 민사 회수 참고 안내를 연결. 안내의 개정 검토 공지와 현행 법령·송달 가능성·채무자 재산을 함께 확인. 실제 환급 보장 아님.')
    return {'units':units,'publication_date':publication_date,'publication_period':publication_period,
            'legal_basis':legal_basis,
            'source_date_note':'게시 등록일 또는 웹진 발행 연월만 보존. 수정일·법령 시행일·현행 적용일로 대체하지 않음.',
            'scope_included':[u['source_locator'] for u in units],
            'scope_excluded':[{'scope':'본문 밖 메뉴·대기 화면·팝업·푸터·담당자 연락처·통계·장식 이미지',
                              'reason':'이번 폐업·증빙·대응 절차 범위 밖. 원본 HTML 보존'},
                             {'scope':'첨부 PDF·외부 접수 페이지·온라인 기능 안내·청렴 안내',
                              'reason':'이번 HTML 본문 수집 범위 밖. 개별 접수 화면·첨부·이미지 전수 검증은 하지 않음'}],
            'warnings':['공식 소비자 안내이며 법령·판례가 아님. 안내 시점 이후 변경 및 구체적 계약·결제 조건 확인 필요.',
                        '사업자에게 환급 요구, 카드사에 잔여 할부금 지급 거절, 소비자원 상담·피해구제를 구분. 환급금의 실제 회수·민사집행은 별도 검토.']}


OVERSEAS_GUIDES = {'kca_overseas_types', 'kca_overseas_checklist', 'kca_overseas_returns'}


def extract_overseas_guide(source, data, *, context=None):
    """본문 영역과 확인한 단위를 지정하여 유형·조건·원문 표를 보존한다."""
    context = context if context is not None else ProcessingContext()
    tree = HTMLTree(decode_html(data)).root
    mains = tree.find(lambda n: n.attrs.get('id') == 'contents')
    if len(mains) != 1:
        raise ValueError('KCA 해외직구 안내 본문 누락 또는 중복')
    main = mains[0]
    clean = lambda n: ' '.join(n.text().split())
    classes = lambda n, value: value in n.attrs.get('class', '').split()
    units = []
    excluded = [{'scope': '본문 밖 메뉴·대기 화면·로그인·푸터 및 장식 이미지',
                 'reason': '안내 본문이 아닌 화면 요소. 이미지 별도 의미 추론 없음. 원본 보존'}]

    def unique(nodes, label):
        if len(nodes) != 1:
            raise ValueError('공식 안내 필수 영역 누락 또는 중복: '+label)
        return nodes[0]

    def add(identifier, node, role, locator, trade_types=None, warnings=None):
        text = clean(node)
        if not text:
            raise ValueError('공식 안내 본문이 비어 있음: '+locator)
        unit = {'unit_id': identifier, 'role': role, 'text': text, 'atomic': True,
                'source_locator': '#contents > '+locator, 'trade_types': trade_types or [],
                'warnings': warnings or []}
        units.append(unit)
        return unit

    identifier = source['source_id']
    if identifier == 'kca_overseas_types':
        add('definition', unique(main.find(lambda n: classes(n,'border')), '해외직구 정의'),
            'trade_definition', 'div.border')
        novice = main.find(lambda n: n.tag == 'ul' and classes(n,'novice'))
        if len(novice) != 2:
            raise ValueError('거래 유형 정의·과정 목록 누락')
        types = [('direct_delivery','직접배송'),('forwarding','배송대행'),('purchase_agent','구매대행')]
        for group, prefix, role in [(novice[0], 'type', 'trade_definition'), (novice[1], 'process', 'trade_process')]:
            rows = [n for n in group.children if getattr(n,'tag',None) == 'li']
            if len(rows) != 3:
                raise ValueError('해외직구 유형 3종 목록 불일치')
            for row, (kind, title) in zip(rows, types):
                if not clean(row).startswith(title):
                    raise ValueError('거래 유형 순서·제목 변경')
                add(prefix+'_'+kind,row,role,'ul.novice '+prefix+' > li: '+title,[kind])
        table = unique(main.find(lambda n:n.tag=='table'), '유형별 장단점 표')
        rows = table.find(lambda n:n.tag=='tr')
        cells = [[{'tag': n.tag, 'text': clean(n), 'rowspan':int(n.attrs.get('rowspan',1)),
                   'colspan':int(n.attrs.get('colspan',1)), 'scope':n.attrs.get('scope')}
                  for n in row.children if getattr(n,'tag',None) in {'th','td'}] for row in rows]
        if (len(cells)!=4 or any(len(row)!=3 for row in cells)
                or any(c['rowspan']!=1 or c['colspan']!=1 or not c['text'] for row in cells for c in row)
                or [c['text'] for c in cells[0]]!=['유형','이런 점이 좋아요!','이런 점은 유의하세요!']
                or [row[0]['text'] for row in cells[1:]]!=[title for _,title in types]):
            raise ValueError('해외직구 비교표 구조 변경: 원문 재검토 필요')
        u=add('comparison',table,'trade_comparison','table: 해외직구 유형별 장·단점', [k for k,_ in types])
        u['official_table']={'headers':[c['text'] for c in cells[0]], 'rows':cells}
        # 각 셀의 열 제목을 반복하여 관계를 검색 본문에도 보존한다.
        u['text']='해외직구 유형별 장·단점 점검\n'+'\n'.join(
            ' | '.join(header+': '+cell['text'] for header,cell in zip(u['official_table']['headers'],row))
            for row in cells[1:])
        excluded += [{'scope':'배송 가능 여부 tip·나에게 맞는 유형선택',
                      'reason':'이번 유형 정의·과정·조건 비교 범위 밖. 원본 보존'}]
    elif identifier == 'kca_overseas_checklist':
        payment = unique(main.find(lambda n:classes(n,'itemList') and '결제내역' in n.text()),'결제 증빙 안내')
        add('payment_evidence',payment,'purchase_evidence','div.itemList: 반품·취소 대비')
        tips=unique(main.find(lambda n:classes(n,'tip_layout')),'유형별 피해예방')
        blocks=[n for n in tips.children if getattr(n,'tag',None) in {'ul','div'}]
        if len(blocks)!=5:
            raise ValueError('유형별 예방·대응 블록 변경')
        required=['해외직접배송','주문번호','배송대행업체','구매대행업체','1372']
        if any(word not in clean(block) for word,block in zip(required,blocks)):
            raise ValueError('유형별 예방·대응 내용 누락')
        add('direct_delivery',blocks[0],'trade_conditions','div.tip_layout > 직접배송 안내',['direct_delivery'])
        add('direct_contact',blocks[1],'dispute_procedure','div.tip_layout > 해외 쇼핑몰 대처방안',['direct_delivery'])
        delivery=unique(blocks[2].find(lambda n:n.tag=='li' and clean(n).startswith('배송대행업체를 통한 거래 시에는')), '배송대행 사고·증거 안내')
        add('forwarding_evidence',delivery,'trade_conditions','div.tip_layout > 배송대행 > 운송 사고·증거',['forwarding'])
        unit=add('purchase_agent',blocks[3],'trade_conditions','div.tip_layout > 구매대행 안내',['purchase_agent'],
            ['국내소재 구매대행에 관한 안내. 7일 청약철회 안내만으로 모든 거래의 반품 가능 여부를 단정하지 않음. 전자상거래법 제17·18조의 예외·비용 조건을 함께 확인.'])
        if not all(word in unit['text'] for word in ['국내소재의 구매대행은 국내법', '7일', '반품', '배송비용']):
            raise ValueError('국내 구매대행 적용 범위 또는 비용 조건 누락')
        lawname='전자상거래 등에서의 소비자보호에 관한 법률'
        records = _law_records(context)
        context=[c['chunk_id'] for c in records if c.get('law_name')==lawname and c.get('article_label') in {'제17조','제18조'}]
        if not context or {c.get('article_label') for c in records if c['chunk_id'] in context}!={'제17조','제18조'}:
            raise ValueError('전자상거래법 청약철회·비용 청크 연결 누락')
        unit['required_context_chunk_ids']=context
        add('purchase_agent_contact',blocks[4],'dispute_procedure','div.tip_layout > 국내업체 피해구제',['purchase_agent'])
        excluded += [{'scope':'구매품목·관세 한도·지역별 세금·유형 선택 및 외부 조회 링크',
                      'reason':'이번 반품·거래조건 범위 밖. 관세·지역 세금의 현행성은 검증하지 않음. 원본 보존'}]
    elif identifier == 'kca_overseas_returns':
        box=unique(main.find(lambda n:classes(n,'box-style')),'수령·반품 조건')
        u=add('return_conditions',box,'return_conditions','div.box-style: 수령/반품 단계 유의사항',
            ['direct_delivery','forwarding','purchase_agent'],
            ['해외직구 일반 안내. 판매자·대행업체 약정, 파손 발생 구간, 적용 국가·법령에 따라 구체적 책임과 비용을 별도 확인.'])
        if not all(word in u['text'] for word in ['단순변심','제품하자','국내에서 배송대행지까지','국제배송비','단, 배송대행지에서 국내로 오는 과정','보상을 요구']):
            raise ValueError('하자 반품 비용 또는 배송대행 파손 예외 누락')
        excluded += [{'scope':'외부 반품가이드 연결 페이지', 'reason':'링크 대상은 이번 수집에 포함하지 않음'}]
    else:
        raise ValueError('검토하지 않은 해외직구 안내 ID')
    return {'units':units, 'scope_included':[u['source_locator'] for u in units], 'scope_excluded':excluded,
            'source_date_note':'안내 페이지에 게시·수정일 미표시. 수집일을 법령 시행일 또는 안내 기준일로 대체하지 않음.',
            'warnings':['한국소비자원의 공식 소비자 안내이며 법령·판례가 아님. 직접배송·배송대행·구매대행을 구분.',
                        '해외 판매자에 국내 구매대행 기준을 일괄 적용하지 않음. 계약 상대방·거래조건·적용 법령은 개별 확인.']}


def extract_labor_guide(source, data):
    """승인한 공식 HTML의 범위를 명시해 추출. 구조 변경·본문 누락은 실패 처리."""
    tree = HTMLTree(decode_html(data)).root
    identifier = source['source_id']
    if identifier == 'moel_wage_procedure':
        panels = tree.find(lambda n: n.attrs.get('id') == 'tabpanel01')
        if len(panels) != 1:
            raise ValueError('노동포털 진정/고소 본문 영역 누락 또는 중복')
        panel = panels[0]
        expected = ['진정/고소', '신청방법', '진정 제기', '진정 취하', '처리절차']
        sections = []
        current = None
        for node in panel.children:
            if not hasattr(node, 'tag'):
                continue
            headings = node.find(lambda n: n.tag == 'h4')
            if headings:
                if len(headings) != 1:
                    raise ValueError('공식 안내 제목 구조 변경')
                current = [' '.join(headings[0].text().split()), []]
                sections.append(current)
            elif current is not None and (node.tag == 'ul' or 'contBox' in node.attrs.get('class','').split()):
                current[1].append(' '.join(node.text().split()))
        if [title for title, _ in sections] != expected or any(not parts or not all(parts) for _, parts in sections):
            raise ValueError('공식 안내 필수 절차 누락 또는 제목 변경')
        units = [{'unit_id': f'procedure_{i}', 'role': 'procedure', 'text': title+'\n'+'\n'.join(parts),
                  'source_locator': '#tabpanel01 > h4: '+title, 'atomic': True}
                 for i, (title, parts) in enumerate(sections)]
        return {'units': units, 'scope_included': expected,
                'scope_excluded': [{'scope': '본문 외 메뉴·로그인·서식 버튼', 'reason': '안내 본문이 아닌 화면 요소'},
                    {'scope': '다른 탭의 대지급금·융자·민사소송·도산 및 연령별 지원 안내',
                     'reason': '이번 신고 절차·준비자료 보강 범위 밖. 원본 HTML에는 보존'}],
                'source_date_note': '페이지에 게시·수정일 미표시. 수집일을 시행일로 대체하지 않음.'}
    if identifier == 'moel_wage_evidence_1000323154':
        answers = tree.find(lambda n: n.tag == 'div' and n.attrs.get('style') == 'padding-top:0px'
                            and '급여자료' in n.text() and '답변일' in n.text())
        if len(answers) != 1:
            raise ValueError('1350 준비자료 답변 영역 누락 또는 중복')
        answer = ' '.join(answers[0].text().split())
        start = answer.find('- 진정 제기 시')
        end = answer.find('◆ 진정제기 방법', start)
        date_match = re.search(r'답변일\s*:\s*(\d{4})\.(\d{2})\.(\d{2})', answer)
        if start < 0 or end <= start or not date_match:
            raise ValueError('1350 준비자료 구간 또는 답변일 누락')
        text = answer[start:end].strip()
        end = start + len(text)
        if not all(word in text for word in ['근로계약서', '급여자료', '근로시간', '별도 증빙이 어려운 경우', '대질 조사']):
            raise ValueError('준비자료 예시 또는 증빙 부족 안내 누락')
        return {'units': [{'unit_id': 'evidence_0', 'role': 'supporting_documents', 'text': text,
                    'source_locator': '1350 상담 ID 1000323154 > 준비자료 답변 구간', 'atomic': True,
                    'start': start, 'end': end, 'offset_basis': 'normalized_answer_text'}],
                'normalized_answer_text': answer, 'information_as_of': '-'.join(date_match.groups()),
                'source_date_note': 'information_as_of는 해당 상담 답변일. 법령 시행일·페이지 전체 최종 수정일이 아님.',
                'scope_included': ['진정 준비자료 예시', '증빙 부족 시 조사 안내'],
                'scope_excluded': [{'scope': '질문·다른 상담 답변 및 신고시점·소멸시효·지연이자·접수 경로',
                    'reason': '이번 준비자료 구간 밖. 신고 경로는 노동포털 안내 사용. 원본 HTML에는 보존'}]}
    raise ValueError('정제 범위가 검토되지 않은 공식 안내 ID')


def preprocess(source, *, context=None):
    context = context if context is not None else ProcessingContext()
    path=context.project_root/source['raw_file']
    data=path.read_bytes()
    if digest(data)!=source['raw_sha256']:
        raise ValueError('수집 원본 지문 불일치')
    doc={**source,'schema_version':1,'units':[],'embedding_candidate':False,'needs_manual_review':False,
         'publication_date':None,'updated_date':None,'decision_date':None,'effective_date':None,
         'warnings':[],'processed_at':datetime.now(timezone.utc).isoformat()}
    kind=source['document_type']
    if kind=='official_guide':
        doc.update(extract_consumer_redress_guide(source,data, context=context) if source['source_id'] in CONSUMER_REDRESS_GUIDES
                   else extract_civil_recovery_guide(source,data, context=context) if source['source_id'] in CIVIL_RECOVERY_GUIDES or source['source_id'] in HOUSING_GUIDES
                   else extract_fitness_closure_guide(source,data, context=context) if source['source_id'] in FITNESS_CLOSURE_GUIDES
                   else extract_overseas_guide(source,data, context=context) if source['source_id'] in OVERSEAS_GUIDES
                   else extract_labor_guide(source, data))
        doc['guide_id']=source['source_id']
        if source['source_id'] not in OVERSEAS_GUIDES | FITNESS_CLOSURE_GUIDES | CONSUMER_REDRESS_GUIDES | CIVIL_RECOVERY_GUIDES.keys() | HOUSING_GUIDES.keys():
            doc['warnings']=['공식 절차 안내이며 법령·판례와 구분. 수집 이후 접수 화면·절차 변경 가능.',
                         '준비자료는 상담을 돕는 예시이며 모든 자료가 있어야만 신고할 수 있다는 의미가 아님.']
    elif kind=='mediation_case':
        title,units,dates=extract_case(data)
        if title!=source['title']:
            raise ValueError('목록 제목과 상세 사례 제목 불일치')
        doc.update(title=title,units=units,publication_date=dates.get('등록일'),updated_date=dates.get('수정일'),
                   decision_date=dates.get('결정일') or dates.get('조정결정일'))
        doc['source_missing_fields']=['party_claims'] if not any(u['role']=='party_claims' for u in units) else []
        doc['warnings']=['분쟁조정 사례이며 법원 판례가 아님. 게시 수정일을 결정일로 대체하지 않음.',
                         '사례 당시 사실관계·법령 적용에 따른 판단. 현행 일반 기준으로 단정하지 않음.']
        if source['source_id'] in {'kca_case_1003739114','kca_case_1003640779'}:
            if __package__:
                from .review_pilates_law_versions import verified_case
            else:
                from review_pilates_law_versions import verified_case
            verifier = context.case_verifier if context.case_verifier is not None else verified_case
            doc['law_version_review']=verifier(doc)
            doc['warnings']+=doc['law_version_review']['warnings']
    elif kind=='administrative_rule':
        if source['source_id'] in {'continual_contract_refund','minimum_wage_2026','minimum_wage_simple_labor'}:
            units,meta,extra=extract_supporting_admin(source,data, context=context)
            doc.update(extra)
        else:
            units,meta=extract_admin(data)
            units=apply_reviewed_consumer_appendices(source,units, context=context)
        doc.update(units=units,administrative_metadata=meta,effective_date=meta.get('시행일자'))
        doc['warnings']=(['공식 고시. 적용 범위·예외·유효 기간과 상위 법령을 함께 확인. 계산 자동화는 별도 검증.']
                         if source['source_id'] in {'continual_contract_refund','minimum_wage_2026','minimum_wage_simple_labor'}
                         else ['합의 또는 권고 기준. 법률·표준약관·개별 사례와 구분. 수집 당시 현행 버전.'])
    elif source['format']=='hwpx' and kind=='standard_terms':
        text,units,tables=extract_hwpx(path)
        doc.update(units=units,extracted_text=text,tables=tables)
        match=re.search(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(제정|개정)',text)
        doc['terms_revision_date']='-'.join([match[1],match[2].zfill(2),match[3].zfill(2)]) if match else None
        doc['warnings']=['공식 표준약관이며 모든 계약에 자동 적용되는 법률은 아님. 계약 약관 확인 필요.']
        doc['warnings'].append('HWPX 표 행·열 및 병합 범위 보존. 수식은 HWP 표기 원문이며 계산하지 않음.')
    elif source['source_id'] in PARCEL_VERSIONS and source['format']=='hwp':
        doc.update(extract_parcel_pdf(source, data, context=context))
        doc['status']='processed_historical' if doc['historical_reference'] else 'processed'
        doc['warnings']=['공식 표준약관이며 모든 계약에 자동 적용되는 법률은 아님. 계약 약관 확인 필요.',
                         '2020년본은 수집한 두 버전 중 우선 사용본. 최신 현행 여부는 별도 확인 필요.']
        if doc['historical_reference']:
            doc['warnings']=['2007년 개정 약관: 과거 버전 비교용이며 기본 검색·임베딩에서 제외.']
    elif source['format']=='hwp':
        doc['status']='raw_only_requires_conversion'
        doc['warnings']=['HWP 원본 보존. 본문 변환 및 대조 전 검색 제외.']
    else:
        doc['status']='reference_only'
    doc['embedding_candidate']=bool(doc['units']) and not doc['needs_manual_review'] and not doc.get('historical_reference',False)
    return doc


def run(collection_report=None, output_dir=OUTPUT_DIR, *, context=None,
        report_path=None, update_manifest=True):
    """정제 결과를 저장하고 보고서 딕셔너리를 반환한다. API를 호출하지 않는다.

    context는 읽을 참조 자료, output_dir는 저장 위치다. 사용자 지정 출력에서는
    갱신된 manifest도 출력 폴더에 저장하여 기존 참조 manifest를 보호한다.
    자료별 오류는 failures에 기록하며 입력 보고서·저장 오류는 호출부에 전달한다.
    """
    context = context if context is not None else ProcessingContext()
    output_dir = Path(output_dir).resolve()
    collection_report = (Path(collection_report).resolve() if collection_report is not None else
                         context.processed_dir / "consumer_collection_report.json")
    report_path = Path(report_path).resolve() if report_path is not None else output_dir / "consumer_preprocessing_report.json"
    manifest_output = context.manifest_path if output_dir == OUTPUT_DIR.resolve() else output_dir / "source_manifest.json"
    if (output_dir.is_relative_to(context.raw_dir) or context.raw_dir.is_relative_to(output_dir)
            or report_path.is_relative_to(context.raw_dir)
            or (update_manifest and manifest_output.is_relative_to(context.raw_dir))):
        raise ValueError("정제 결과가 원본 자료 폴더를 덮어쓸 수 없습니다.")
    if report_path in (collection_report, context.manifest_path, context.appendix_review_path):
        raise ValueError("결과 보고서가 입력·검토 기록을 덮어쓸 수 없습니다.")
    report=json.loads(collection_report.read_text(encoding='utf-8'))
    result={'saved':[],'failures':[],'excluded':[],'duplicate_aliases':[]}
    seen={}
    parcel_documents=[]
    for source in report['saved']:
        try:
            doc=preprocess(source, context=context)
            if doc['source_id'] in PARCEL_VERSIONS:
                parcel_documents.append(doc)
            fingerprint=digest(json.dumps([(u['role'],u['text']) for u in doc['units']],ensure_ascii=False).encode())
            if doc['units'] and fingerprint in seen:
                result['duplicate_aliases'].append({'source_id':doc['source_id'],'representative_id':seen[fingerprint]})
                continue
            if doc['units']:
                seen[fingerprint]=doc['source_id']
            write_json(output_dir/(doc['source_id']+'.json'),doc)
            result['saved'].append({'file':doc['source_id']+'.json','source_id':doc['source_id'],
                'document_type':doc['document_type'],'embedding_candidate':doc['embedding_candidate'],
                'domain':doc['domain'],'unit_count':len(doc['units']),'raw_sha256':doc['raw_sha256']})
            if not doc['embedding_candidate']:
                result['excluded'].append({'source_id':doc['source_id'],'reason':doc.get('status','manual_review')})
        except (ValueError,OSError,KeyError,ET.ParseError,UnicodeError,zipfile.BadZipFile,struct.error,zlib.error) as exc:
            result['failures'].append({'source_id':source['source_id'],'error':str(exc)})
    if len(parcel_documents)==len(PARCEL_VERSIONS):
        comparison=compare_parcel_versions(parcel_documents)
        write_json(output_dir/'parcel_terms_revision_comparison.json',comparison)
        result['parcel_revision_comparison']={'file':'parcel_terms_revision_comparison.json','counts':comparison['counts']}
    write_json(report_path,result)
    if not update_manifest:
        return result
    manifest=json.loads(context.manifest_path.read_text(encoding='utf-8'))
    entries={e['source_id']:e for e in result['saved']}
    aliases={e['source_id']:e['representative_id'] for e in result['duplicate_aliases']}
    originals={e['source_id']:e for e in report['saved']}
    for source in manifest['sources']:
        original=originals.get(source['source_id'])
        if original:
            source['raw_file']=original['raw_file']
            source['raw_sha256']=original['raw_sha256']
            source['collection_status']='saved'
        entry=entries.get(source['source_id'])
        source['review_status']=('historical_reference' if source['source_id'] == 'ftc_terms_11125_file_e27d5f98' and entry and entry['unit_count']
                                 else 'structural_validation_passed' if entry and entry['embedding_candidate']
                                 else 'duplicate_alias' if source['source_id'] in aliases
                                 else 'reference_or_conversion_required')
        source['representative_id']=aliases.get(source['source_id'])
    if update_manifest:
        write_json(manifest_output,manifest)
    return result



def main(argv=None):
    """CLI 인자 처리·요약 출력·종료 코드."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest-file", type=Path, default=MANIFEST)
    parser.add_argument("--appendix-review-file", type=Path, default=APPENDIX_REVIEW_FILE)
    parser.add_argument("--law-chunks", type=Path, default=CHUNK_LAWS_DATA_PATH / "chunks.jsonl")
    parser.add_argument("--no-update-manifest", action="store_true")
    args = parser.parse_args(argv)
    context = ProcessingContext(processed_dir=args.reference_dir, manifest_path=args.manifest_file,
                                appendix_review_path=args.appendix_review_file, law_chunk_path=args.law_chunks)
    result = run(args.collection_report, args.output_dir, context=context,
                 update_manifest=not args.no_update_manifest)
    print(f"정제 {len(result['saved'])} / 실패 {len(result['failures'])} / 검색 제외 {len(result['excluded'])}")
    return 1 if result['failures'] else 0


if __name__=='__main__':
    raise SystemExit(main())
