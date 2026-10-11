"""공식 소비자 자료를 유형·역할·시점·출처를 보존하며 정제한다."""
import json
import re
import struct
import zlib
from collections import Counter
from difflib import SequenceMatcher, unified_diff
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .collect_consumer_sources import ROOT, RAW_DIR, MANIFEST, HTMLTree, decode_html, digest, write_json
else:
    from collect_consumer_sources import ROOT, RAW_DIR, MANIFEST, HTMLTree, decode_html, digest, write_json

OUTPUT_DIR = MANIFEST.parent

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


def extract_parcel_pdf(source, data):
    import pdfplumber
    basename, revision, article_count, version_role = PARCEL_VERSIONS[source['source_id']]
    pdf_path = OUTPUT_DIR/'converted'/(basename+'.pdf')
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
            'derived_pdf_file':pdf_path.relative_to(ROOT).as_posix(),
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


def preprocess(source):
    path=ROOT/source['raw_file']
    data=path.read_bytes()
    if digest(data)!=source['raw_sha256']:
        raise ValueError('수집 원본 지문 불일치')
    doc={**source,'schema_version':1,'units':[],'embedding_candidate':False,'needs_manual_review':False,
         'publication_date':None,'updated_date':None,'decision_date':None,'effective_date':None,
         'warnings':[],'processed_at':datetime.now(timezone.utc).isoformat()}
    kind=source['document_type']
    if kind=='mediation_case':
        title,units,dates=extract_case(data)
        if title!=source['title']:
            raise ValueError('목록 제목과 상세 사례 제목 불일치')
        doc.update(title=title,units=units,publication_date=dates.get('등록일'),updated_date=dates.get('수정일'),
                   decision_date=dates.get('결정일') or dates.get('조정결정일'))
        doc['source_missing_fields']=['party_claims'] if not any(u['role']=='party_claims' for u in units) else []
        doc['warnings']=['분쟁조정 사례이며 법원 판례가 아님. 게시 수정일을 결정일로 대체하지 않음.',
                         '사례 당시 사실관계·법령 적용에 따른 판단. 현행 일반 기준으로 단정하지 않음.']
    elif kind=='administrative_rule':
        units,meta=extract_admin(data)
        doc.update(units=units,administrative_metadata=meta,effective_date=meta.get('시행일자'))
        doc['warnings']=['합의 또는 권고 기준. 법률·표준약관·개별 사례와 구분. 수집 당시 현행 버전.']
    elif source['format']=='hwpx' and kind=='standard_terms':
        text,units,tables=extract_hwpx(path)
        doc.update(units=units,extracted_text=text,tables=tables)
        match=re.search(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(제정|개정)',text)
        doc['terms_revision_date']='-'.join([match[1],match[2].zfill(2),match[3].zfill(2)]) if match else None
        doc['warnings']=['공식 표준약관이며 모든 계약에 자동 적용되는 법률은 아님. 계약 약관 확인 필요.']
        doc['warnings'].append('HWPX 표 행·열 및 병합 범위 보존. 수식은 HWP 표기 원문이며 계산하지 않음.')
    elif source['source_id'] in PARCEL_VERSIONS and source['format']=='hwp':
        doc.update(extract_parcel_pdf(source, data))
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


def run():
    report=json.loads((OUTPUT_DIR/'consumer_collection_report.json').read_text(encoding='utf-8'))
    result={'saved':[],'failures':[],'excluded':[],'duplicate_aliases':[]}
    seen={}
    parcel_documents=[]
    for source in report['saved']:
        try:
            doc=preprocess(source)
            if doc['source_id'] in PARCEL_VERSIONS:
                parcel_documents.append(doc)
            fingerprint=digest(json.dumps([(u['role'],u['text']) for u in doc['units']],ensure_ascii=False).encode())
            if doc['units'] and fingerprint in seen:
                result['duplicate_aliases'].append({'source_id':doc['source_id'],'representative_id':seen[fingerprint]})
                continue
            if doc['units']:
                seen[fingerprint]=doc['source_id']
            write_json(OUTPUT_DIR/(doc['source_id']+'.json'),doc)
            result['saved'].append({'file':doc['source_id']+'.json','source_id':doc['source_id'],
                'document_type':doc['document_type'],'embedding_candidate':doc['embedding_candidate'],
                'domain':doc['domain'],'unit_count':len(doc['units']),'raw_sha256':doc['raw_sha256']})
            if not doc['embedding_candidate']:
                result['excluded'].append({'source_id':doc['source_id'],'reason':doc.get('status','manual_review')})
        except (ValueError,OSError,KeyError,ET.ParseError,UnicodeError,zipfile.BadZipFile,struct.error,zlib.error) as exc:
            result['failures'].append({'source_id':source['source_id'],'error':str(exc)})
    if len(parcel_documents)==len(PARCEL_VERSIONS):
        comparison=compare_parcel_versions(parcel_documents)
        write_json(OUTPUT_DIR/'parcel_terms_revision_comparison.json',comparison)
        result['parcel_revision_comparison']={'file':'parcel_terms_revision_comparison.json','counts':comparison['counts']}
    write_json(OUTPUT_DIR/'consumer_preprocessing_report.json',result)
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
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
    write_json(MANIFEST,manifest)
    print(f"정제 {len(result['saved'])} / 실패 {len(result['failures'])} / 검색 제외 {len(result['excluded'])}")
    return result


if __name__=='__main__':
    raise SystemExit(bool(run()['failures']))
