"""9대 분야 45문항의 자료 근거 검토와 BM25 검색 진단.

uv run python -m src.scripts.tests.test_domain_coverage_pmh
직접 실행은 보고서만 작성한다. unittest 실행은 데이터에 쓰지 않는다.
검색/키워드 적중으로 충분성을 판정하지 않으며 검토한 근거와 메모를 사용한다.
"""
import hashlib
import json
import unittest
from collections import Counter
from unittest.mock import patch

from src.const.config import (
    CHUNK_DATA_PATH,
    DATA_PATH,
    PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH,
    PROJECT_PATH,
)

from src.scripts.tests.test_consumer_coverage_pmh import load_chunks, SEARCH_OPTIONS, SEARCH_POLICY
from src.scripts.tests.test_precedent_retrieval_pmh import BM25

ROOT = PROJECT_PATH
OUTPUT = PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH
REPORT_JSON = OUTPUT / "domain_coverage_report_pmh.json"
REPORT_MD = OUTPUT / "domain_coverage_report_pmh.md"
STATUS_LABELS = {"supported": "기본 근거 확보", "partial": "근거 일부·자료 보완",
                 "restricted": "원문 확인 전 적용 보류", "needs_review": "추가 내용 검토"}


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_fingerprint():
    files = sorted(p for p in CHUNK_DATA_PATH.rglob("*") if p.is_file())
    return sha([(p.relative_to(ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])


def question_bank(chunks):
    """선정 근거는 자료 내용 검토 결과. 선택 조건은 근거 위치를 찾기 위한 도구다."""
    bank = []
    def law(title, label):
        ids = [c["chunk_id"] for c in chunks if c.get("title") == title
               and c.get("article_label") == label and c.get("content_type") == "main_article"]
        if not ids:
            raise ValueError(f"검토한 법령 조문 누락: {title} {label}")
        return ids
    def add(domain, key, query, required, status, note, ids, action=""):
        bank.append({"domain": domain, "question_id": key, "query": query,
                     "required_evidence": required.split(";"), "review_status": status,
                     "review_note": note, "reviewed_chunk_ids": ids, "follow_up": action})
    housing = "전세·월세 분쟁"
    add(housing,"housing_repair","월세 집의 난방시설이 고장났는데 집주인이 수리해야 하나요?",
        "임대인의 유지 의무;사소한 수선과 사용 불가능한 고장의 구분","supported",
        "민법 제623조와 안내서 난방시설 수선 사례의 구분을 확인. 계약 특약과 고장 원인은 별도 사실확인 필요.",
        law("민법","제623조") + ["guide_23ffe4c8fd5715cb_p0047_table_0000"])
    add(housing,"housing_renewal","주택 임대차 계약갱신을 요구하면 집주인이 거절할 수 있나요?",
        "요구 기간;행사 횟수;거절 사유","supported","제6조의 기간과 제6조의3의 횟수·거절 사유를 함께 확인. 실제 계약 체결 시점은 별도 확인.",
        law("주택임대차보호법","제6조") + law("주택임대차보호법","제6조의3"))
    add(housing,"housing_increase","집주인이 월세를 올리려는데 증액 한도와 간격이 있나요?",
        "증액 청구 사유;증액 간격;한도와 조례 가능성","supported","제7조에 기본 한도와 기간 및 조례 가능성 있음. 지역 조례 수치는 별도 확보 대상.",law("주택임대차보호법","제7조"))
    add(housing,"housing_termination","묵시적으로 갱신된 월세계약은 언제 해지할 수 있나요?",
        "묵시적 갱신 조건;통지 후 효력 시점","supported","제6조와 제6조의2를 함께 확인. 일반 고정기간 계약의 임의 중도해지와 구분.",law("주택임대차보호법","제6조")+law("주택임대차보호법","제6조의2"))
    add(housing,"housing_priority","전입신고와 확정일자를 받으면 보증금 우선변제권은 어떻게 생기나요?",
        "주택 인도와 주민등록;확정일자;효력 발생 시점","supported","법령의 요건과 안내서의 발생 시점 설명이 함께 있음.",law("주택임대차보호법","제3조")+law("주택임대차보호법","제3조의2")+["guide_23ffe4c8fd5715cb_p0031_body_0000"])
    used = "중고거래 피해"
    add(used,"used_nondelivery","중고거래 대금을 보냈는데 물건을 보내지 않으면 무조건 사기인가요?",
        "기망과 재산 취득;단순 계약 불이행과 구분","supported","안내서가 거래 당시 기망과 단순 미배송의 차이를 설명하고 형법 조문이 있음.",law("형법","제347조")+["guide_246eabd0676d5957_p0007_body_0000"])
    add(used,"used_defect","개인에게 산 중고물건에 숨겨진 하자가 있으면 환불을 요구할 수 있나요?",
        "하자와 매수인의 인식;해제·손해배상 구분;행사 기간","supported","민법 제580조의 준용 조문 제575조, 제582조와 개인거래 안내를 함께 확인. 무조건 환불 가능하다고 일반화하지 않음.",law("민법","제580조")+law("민법","제575조")+law("민법","제582조")+["guide_246eabd0676d5957_p0010_body_0000"])
    add(used,"used_phishing","중고거래 판매대금이 보이스피싱 피해금이면 무조건 반환해야 하나요?",
        "정상 거래대금 여부;악의·중과실;입증과 사건별 사실관계","supported","선정 판례와 안내서에 판단 조건이 있음. 악의·중과실 및 실제 거래 여부를 확인해야 함.",["240967_0002","guide_246eabd0676d5957_p0022_body_0000"])
    add(used,"used_report","중고거래 사기 신고는 어디서 하고 어떤 증거를 준비해야 하나요?",
        "신고 경로;방문 절차;거래 증거","supported","ECRM 접수·방문 안내와 선정 판례의 이체내역·문자·판매글 증거를 확인. 판례 증거 목록은 필수 서류 전부를 뜻하지 않음.",["guide_246eabd0676d5957_p0018_body_0000","guide_246eabd0676d5957_p0019_image_summary_0000","605583_0001"])
    add(used,"used_minor","미성년 자녀가 개인에게 산 중고물건의 계약을 취소할 수 있나요?",
        "법정대리인 동의;처분허락 재산;속임수 예외;취소권자;취소 후 반환","supported","기존 제5·6·17조에 제140조(취소권자)·제141조(취소 효과)를 연결해 기본 요건·예외·반환 범위 근거를 확인. 실제 동의·속임수·현존 이익과 플랫폼 약정은 개별 사실 검토 대상이며 개인거래에 사업자 환불 기준을 일괄 적용하지 않음.",law("민법","제5조")+law("민법","제6조")+law("민법","제17조")+law("민법","제140조")+law("민법","제141조"))
    shopping = "온라인 쇼핑 환불"
    add(shopping,"shopping_cost","인터넷 쇼핑몰에서 단순변심으로 반품하면 배송비는 누가 부담하나요?",
        "단순변심과 하자 구분;반환 비용 부담;위약금 금지","supported","제18조의 비용 부담과 제17조의 사유 구분을 함께 확인.",law("전자상거래 등에서의 소비자보호에 관한 법률","제17조")+law("전자상거래 등에서의 소비자보호에 관한 법률","제18조"))
    add(shopping,"shopping_period","인터넷으로 산 물건을 단순변심으로 반품할 수 있는 기간은 언제부터인가요?",
        "서면·공급 시점;기산점;철회 제한 사유","supported","제17조의 기산점과 제외 사유를 확인. 개별 제품·계약 사실은 추가 필요.",law("전자상거래 등에서의 소비자보호에 관한 법률","제17조"))
    add(shopping,"shopping_custom","주문제작이라는 이유만으로 한복 반품을 거절할 수 있나요?",
        "개별 제작 여부;고지와 동의;시행령 예외","supported","기존 사례와 법 제17조에 시행령 제21조 원문을 연결. 개별 생산·중대한 피해·사전 고지·서면 동의 조건을 함께 확인. 사례 시점의 구법 적용은 별도 확인.",["kca_case_1003164054_case_reasoning_0000"]+law("전자상거래 등에서의 소비자보호에 관한 법률","제17조")+law("전자상거래 등에서의 소비자보호에 관한 법률 시행령","제21조"))
    add(shopping,"shopping_defect","온라인 주문 내용과 다른 제품을 받았을 때 반품 기간과 배송비는 어떻게 되나요?",
        "계약과 다른 이행의 기간;배송비 부담","supported","제17조 제3항과 제18조 제10항을 함께 확인.",law("전자상거래 등에서의 소비자보호에 관한 법률","제17조")+law("전자상거래 등에서의 소비자보호에 관한 법률","제18조"))
    add(shopping,"shopping_overseas","해외직구 반품비가 너무 비싸면 국내 쇼핑몰과 같은 기준으로 환불받나요?",
        "직구·구매대행 구분;비용 고지와 입증;계약 상대방;반품 사유·운송 구간별 조건","supported","KCA 공식 안내로 직접배송·배송대행·쇼핑몰형/위임형 구매대행을 구분. 해외 직접배송의 교환·환불 조건이 국내와 다를 수 있다는 안내, 국내소재 구매대행 안내 및 법 제17·18조 예외·비용 조건을 연결. 하자 반품의 배송대행지까지 국제배송비와 대행지→국내 파손 보상 조건, 단순변심 비용 부담을 구분. 기존 비용 입증 안내·사례를 함께 사용. 개별 해외 판매자·국가의 준거법·관할·약정 및 실제 비용은 별도 확인하며 모든 직구에 국내법을 일괄 적용하지 않음.",["guide_383b62bce56bf1ce_p0032_table_0000","kca_case_1003641715_case_reasoning_0000"]+[c['chunk_id'] for c in chunks if (c.get('source_id')=='kca_overseas_types' and c.get('unit_id') in {'type_direct_delivery','type_forwarding','type_purchase_agent','comparison'}) or (c.get('source_id')=='kca_overseas_checklist' and c.get('unit_id') in {'direct_delivery','payment_evidence','purchase_agent'}) or c.get('source_id')=='kca_overseas_returns']+law("전자상거래 등에서의 소비자보호에 관한 법률","제17조")+law("전자상거래 등에서의 소비자보호에 관한 법률","제18조"))
    fitness = "헬스장·필라테스 환불"
    add(fitness,"fitness_gym","헬스장 이용권을 개인 사정으로 중도해지하면 환불은 어떻게 계산하나요?",
        "개시 전후;기간·횟수 공제;실제 이용료와 위약금","supported","헬스장 약관의 표 셀과 수식·이용료·위약금 정의가 한 청크에 보존됨. 실제 계약 적용과 금액 입력은 별도 검증.",["ftc_terms_46118_file_5b139f26_terms_article_7_0000"])
    add(fitness,"fitness_before","필라테스 수업 시작 전에 개인 사정으로 해지하면 어떤 금액이 공제되나요?",
        "등록기간 개시 전;회원 귀책;이용료·위약금 정의","supported","필라테스 약관의 개시 전 회원 귀책 환급 표와 정의를 확인.",["ftc_terms_47762_file_05d51a47_terms_article_8_0000"])
    add(fitness,"fitness_cancelled","필라테스 수업이 폐강됐는데 정상가로 공제하고 환불한다면 어떻게 하나요?",
        "사업자 귀책;할인가·정상가 공제;계속거래 적용","supported","폐강 사례에 방문판매법 제2·31·32조 및 산정기준의 정의·적용 범위·공제 조건·위약금 별표 원문을 연결. 사업자 귀책과 소비자 귀책 기준을 구분. 시행령 제40·41조와 시행규칙 제22조를 추가 연결해 해지 예외·환급 지연 기준을 확보. 사례의 2023년 계약·해지 전후 법령 본문과 수집본을 대조. 법 제31·32·52조, 시행령 제13·40·41조, 시행규칙 제22조는 동일하고 제2조 전체 변경과 제10호 계속거래 정의의 동일성을 구분. 2019-9호 고시는 같은 버전이며 약관법 제9조 당시 원문을 사례 임베딩 문맥에 보존. 기본 근거 확보; 조정 취지에 따른 위약금 면제·실제 환급액은 일반화하지 않음.",["kca_case_1003739114_case_reasoning_0000","kca_case_1003739114_case_reasoning_0001"]+law("방문판매 등에 관한 법률","제2조")+law("방문판매 등에 관한 법률","제31조")+law("방문판매 등에 관한 법률","제32조")+law("방문판매 등에 관한 법률","제52조")+law("방문판매 등에 관한 법률 시행령","제13조")+law("방문판매 등에 관한 법률 시행령","제40조")+law("방문판매 등에 관한 법률 시행령","제41조")+law("방문판매 등에 관한 법률 시행규칙","제22조")+[c["chunk_id"] for c in chunks if c.get("source_id")=="continual_contract_refund" and c.get("unit_id") in {"article_1","article_2","article_3","article_4","appendix_0"}],"계약·해지 시점 대조 완료. 조정일 미상·개별 계약 약정·귀책사유·이용 증빙 검토")
    add(fitness,"fitness_transfer","양도받은 필라테스 이용권을 해지하면 환급받을 수 있나요?",
        "양도 계약;회원 사정;환급 산정의 사건별 조건","partial","양도 사례에 방문판매법 제31·32조와 산정기준을 연결. 사례 금액을 일반화하지 않으며 시행령 제40·41조와 시행규칙 제22조를 추가 연결. 2023년 계약·해지 시점의 관련 조문과 수집본의 동일성 대조 완료. 약관법 제9조 당시 본문도 사례 문맥에 연결. 양도 약정·실제 지급액·채권 동일성과 사건별 이용 증빙은 추가 검토 필요.",["kca_case_1003640779_case_reasoning_0000","kca_case_1003640779_case_decision_0000"]+law("방문판매 등에 관한 법률","제31조")+law("방문판매 등에 관한 법률","제32조")+law("방문판매 등에 관한 법률","제52조")+law("방문판매 등에 관한 법률 시행령","제13조")+law("방문판매 등에 관한 법률 시행령","제40조")+law("방문판매 등에 관한 법률 시행령","제41조")+law("방문판매 등에 관한 법률 시행규칙","제22조")+["continual_contract_refund_article_2_0000","continual_contract_refund_article_4_0000","continual_contract_refund_appendix_0_0000"],"시점 대조 완료; 양도 약정·실제 지급액·채권 동일성 및 조정일 미상 유지")
    add(fitness,"fitness_closed","헬스장이 폐업하면 남은 이용료를 돌려받는 기준과 회수 방법은 무엇인가요?",
        "사업자 귀책 환급;해지·잔여 이용 증빙;할부 조건과 잔여금;상담·피해구제 절차와 제외 조건","partial","기존 환급 약관에 KCA 헬스장·필라테스 폐업 대응 안내와 소비자원 피해구제 절차를 연결. 잔여 횟수·해지일·내용증명 증빙, 사업자 해지 통지·카드사 잔여할부금 지급 거절 안내 및 조건을 확보. 사업자 부도·폐업 등으로 연락 불가능·소재 미확인 시 피해구제 제외 조건을 함께 보존. 할부거래법 제2·3·4·16조와 시행령 제2·4·11조를 연결해 기간·횟수·금액·통지·잔여금·적용제외 조건을 확보. 안내의 3개월 표현과 법령의 2개월 이상·3회 이상 요건을 구분. 수집 시점 현행본이며 안내 당시 구법·개별 계약 적용은 별도 검토. 현금·일시불·기지급 환급금의 기본 민사 회수 안내 4건을 추가: 소액사건 청구범위·분할 금지, 지급명령 신청·송달·주소보정·2주 이의·소송 전환, 집행문 예외·재산명시·조회·채권집행. 원문 안내 기준일 2026-09-15와 민사소송법 2026-10-02 개정 검토 공지를 보존. 현행 조문 대조와 개별 채무자·재산·도산 검토가 남아 부분 확보 유지.",["ftc_terms_46118_file_5b139f26_terms_article_7_0000"]+[c['chunk_id'] for c in chunks if c.get('source_id') in {'kca_fitness_closure_gym','kca_fitness_closure_pilates','kca_damage_redress_procedure','easylaw_consumer_small_claim','easylaw_consumer_payment_order','easylaw_payment_order_application','easylaw_civil_enforcement'}]+law('할부거래에 관한 법률','제2조')+law('할부거래에 관한 법률','제3조')+law('할부거래에 관한 법률','제4조')+law('할부거래에 관한 법률','제16조')+law('할부거래에 관한 법률 시행령','제2조')+law('할부거래에 관한 법률 시행령','제4조')+law('할부거래에 관한 법률 시행령','제11조'),"기본 민사 회수 안내 보강 완료; 2026-10-02 민사소송법 개정·관련 현행 조문 대조; 안내·계약 당시 적용 법령 및 개별 채무자·재산·도산 검토")
    wage = "아르바이트 임금체불"
    add(wage,"wage_after_exit","아르바이트를 그만뒀는데 마지막 급여는 언제까지 지급해야 하나요?",
        "퇴직 후 지급 기한;합의 연장 예외","supported","근로기준법 제36조 본문과 예외 확보.",law("근로기준법","제36조"))
    add(wage,"wage_minimum","올해 아르바이트 시급이 최저임금보다 낮은지 어떻게 확인하나요?",
        "해당 연도 적용 금액;산입범위;적용 예외","partial","2026년 고시의 시간급·적용 기간·월 환산 조건, 최저임금법 적용 범위·수습 예외·산입범위와 시행령 환산 조문을 확보. 제6조는 2018년 부칙의 연도별 특례와 연결. 시행규칙 제2·3조·별표3과 단순노무직종 지정 고시를 추가 확보. 시행규칙 제2조의 원문 이미지·대체문자, 실제 업무의 직업분류 및 급여 항목 대조가 남아 부분 확보 유지.",law("최저임금법","제3조")+law("최저임금법","제5조")+law("최저임금법","제6조")+law("최저임금법","제7조")+law("최저임금법 시행령","제3조")+law("최저임금법 시행령","제5조")+law("최저임금법 시행령","제6조")+law("최저임금법 시행규칙","제2조")+law("최저임금법 시행규칙","제3조")+[c["chunk_id"] for c in chunks if c.get("unit_id")=="008480_282351_20260102_appendix_000300E"]+["minimum_wage_simple_labor_article_0_0000"]+["minimum_wage_2026_notice_body_0000"]+[c["chunk_id"] for c in chunks if c.get("unit_id")=="000129_218303_20200526_supplement_2018061215666"],"시행규칙 원문 이미지·대체문자 및 실제 급여 항목·업무의 직업분류 대조")
    add(wage,"wage_weekly","주 15시간 안팎으로 일하는 아르바이트의 주휴수당 조건은 무엇인가요?",
        "유급휴일;개근;소정근로시간 기준","supported","법 제18조·제55조와 시행령 제30조를 함께 확인. 실제 스케줄·근로자성 판단은 별도.",law("근로기준법","제18조")+law("근로기준법","제55조")+law("근로기준법 시행령","제30조"))
    add(wage,"wage_report","밀린 아르바이트 급여는 어디에 진정하거나 고소하나요?",
        "관할 노동관서;진정과 고소 차이;진행 경로;준비자료 예시;증빙 부족 시 조사","supported","노동포털 공식 안내로 진정·고소 구분, 온라인 진정·관할 노동관서 방문, 25일(토요일·공휴일 제외)과 연장·조사·시정·송치 절차를 확인. 1350 상담의 근로계약서·급여자료·근로시간 증빙 예시와 증빙 부족 시 조사 안내를 연결. 상담 답변일은 2026-05-14이며 자료를 모두 갖춰야만 신고할 수 있다는 의미가 아님. 로그인 후 실제 신청 화면 및 개별 사건 처리는 검증 범위 밖.",["guide_cad7b1de1e4dc64c_p0078_body_0000"]+[c['chunk_id'] for c in chunks if c.get('source_id') in {'moel_wage_procedure','moel_wage_evidence_1000323154'}])
    add(wage,"wage_retirement","아르바이트도 퇴직금을 받을 수 있나요? 근무 기간과 시간 조건이 궁금해요.",
        "계속근로기간;평균 소정근로시간;근로자 해당 여부","supported","퇴직급여법 제4조의 기간·시간 기본 조건 확보. 근로자성 논란과 금액 산정은 질문 범위 밖.",law("근로자퇴직급여 보장법","제4조"))
    dismissal = "갑작스러운 해고"
    add(dismissal,"dismissal_notice","사장이 오늘부터 나오지 말라고 하면 해고예고수당을 받을 수 있나요?",
        "예고와 수당;계속근로기간;예외","supported","제26조의 예고·수당·예외와 적용 범위를 함께 확인. 사실관계별 계산은 별도.",law("근로기준법","제26조")+law("근로기준법","제11조"))
    add(dismissal,"dismissal_written","구두로만 해고 통보를 받았는데 서면이 필요한가요?",
        "서면 사유·시기;효력;사업장 적용 범위","supported","제27조와 해고 적용 범위 안내를 함께 확인.",law("근로기준법","제27조")+["guide_cad7b1de1e4dc64c_p0013_body_0000"])
    add(dismissal,"dismissal_remedy","부당해고 구제는 어디에 언제까지 신청하나요?",
        "노동위원회;신청 기한;사업장 적용 범위","supported","제28조의 기관·기한과 안내서의 적용 범위 사례 확보.",law("근로기준법","제28조")+["guide_cad7b1de1e4dc64c_p0014_table_0000"])
    add(dismissal,"dismissal_small","직원 4명인 가게에서도 해고예고와 부당해고 구제를 똑같이 적용하나요?",
        "법 제11조;시행령 제7·7조의2;별표1;제26·27·28조 구분","supported","상시근로자 수 산정 제7조의2를 추가 연결하고 제11조·시행령 제7조·별표1과 제26·27·28조를 대조. 수집본 별표1에는 제26조가 기재되고 제27·28조는 기재되지 않아 예고와 구제 적용 범위를 구분하는 기본 근거 확보. 직원 명부상 4명만으로 상시인원을 확정하지 않으며 실제 근로자 수·계속근로기간·예외 적용은 별도 검증.",law("근로기준법","제11조")+law("근로기준법 시행령","제7조")+law("근로기준법 시행령","제7조의2")+law("근로기준법","제26조")+law("근로기준법","제27조")+law("근로기준법","제28조")+[c["chunk_id"] for c in chunks if c.get("law_name")=="근로기준법 시행령" and c.get("content_type")=="appendix" and "appendix_000100" in c["chunk_id"]])
    add(dismissal,"dismissal_probation","수습 중이라는 이유만으로 해고예고 없이 해고할 수 있나요?",
        "계속근로기간과 수습 구분;정당한 이유;예고 예외","supported","제26조의 계속근로기간 예외와 제23조 기본 원칙 확보. 수습이라는 명칭만으로 예외를 단정하지 않음.",law("근로기준법","제26조")+law("근로기준법","제23조"))
    deposit = "보증금 반환"
    add(deposit,"deposit_move","계약이 끝났는데 보증금을 못 받은 상태에서 이사하려면 무엇을 확인해야 하나요?",
        "임차권등기 신청;등기를 마친 이후 권리 유지","supported","제3조의3에 신청 조건과 등기 완료 후 효력 있음. 신청만 한 상태와 구분해야 함.",law("주택임대차보호법","제3조의3"))
    add(deposit,"deposit_small","소액보증금 우선변제는 지역과 계약 시점에 따라 얼마까지 가능한가요?",
        "지역별 범위·금액;담보권과 개정 시점;경과 규정","partial","제8조와 안내서에 시행령 제10·11조 및 수집본의 개정별 부칙을 연결. 현재 지역별 임차인 범위와 우선변제액을 구분. 과거 버전 본문 및 담보권 시점별 경과조치 대조는 미완료.",law("주택임대차보호법","제8조")+["guide_23ffe4c8fd5715cb_p0067_body_0000"]+law("주택임대차보호법 시행령","제10조")+law("주택임대차보호법 시행령","제11조")+[c["chunk_id"] for c in chunks if c.get("law_name")=="주택임대차보호법 시행령" and c.get("content_type")=="supplement" and "담보물권" in c["text"]],"주택임대차 시행령 과거 버전과 담보권 설정 시점별 경과조치 대조")
    add(deposit,"deposit_delay","보증금 반환이 늦어지면 언제부터 어떤 이율로 지연손해금을 청구하나요?",
        "동시이행;이행 제공;약정과 법정이율;송달 및 항쟁 예외","supported","민법 제379·387·397·460·536조, 소송촉진법 제3조의 송달 다음날 원칙·항쟁 예외 및 조 번호 없는 법정이율 규정 전문(연 100분의 12)을 연결. 공식 안내의 등기 완료 후 인도·이사 조건과 147076의 명도 다음날·항쟁 기간 구분을 검토. 판례는 2007년 학원 건물 하급심이며 원문 연20%는 당시 이율로 경고. 계약 종료만으로 자동 지체·12%를 적용하지 않으며 사건별 인도·이행 제공·송달·약정·시점 검증은 별도.",
        ["easylaw_housing_termination_return_0000","easylaw_housing_termination_registration_move_0000"]+law("민법","제379조")+law("민법","제387조")+law("민법","제397조")+law("민법","제460조")+law("민법","제536조")+law("소송촉진 등에 관한 특례법","제3조")+law("소송촉진 등에 관한 특례법 제3조제1항 본문의 법정이율에 관한 규정","전문")+[c['chunk_id'] for c in chunks if c.get('precedent_id')=='147076' and c.get('role')=='civil_historical_interest'])
    add(deposit,"deposit_restore","집주인이 원상복구비를 보증금에서 공제하려는데 전액 부담해야 하나요?",
        "원상회복 범위;특약과 당시 상태;통상손모;실제 비용 입증","partial","공식 주택 안내의 계약 경위·당시 상태·수리 변경 개별 판단과 민법 근거, 147076의 명확한 특약·통상손모·견적·공사기간·잔액 공제 판단을 확보. 추가 판례는 학원 건물 하급심 참고 사례이며 주택 직접 판례로 간주하지 않음. 50%·5일은 개별 인정치로 경고. 주택의 통상손모 공제 직접 사례·개별 비용 증빙 적용은 추가 검토 유지.",
        ["easylaw_housing_restoration_care_0000","easylaw_housing_restoration_restoration_0000"]+law("민법","제374조")+law("민법","제390조")+law("민법","제393조")+law("민법","제615조")+law("민법","제634조")+law("민법","제654조")+[c['chunk_id'] for c in chunks if c.get('precedent_id')=='147076' and c.get('role') in {'civil_case_facts','civil_court_discussion'}],"주택 통상손모·비용 공제 직접 사례 및 계약·당시 상태·비용 증빙 적용 검토")
    add(deposit,"deposit_auction","집이 경매에 넘어갔을 때 보증금을 받으려면 배당요구를 해야 하나요?",
        "배당요구 종기;신청 방법;요구 없이 배당받는 예외;대항요건 유지","partial","민사집행법 제84·88·145·148조와 규칙 제48조, 공식 안내의 종기·서면·경매신청자 및 임차권등기 예외·대항요건 유지 설명을 확보. 안내의 규칙 제91조는 현행 강제관리 수익 처리이고 법 제148조 제3호는 선등기 가압류채권자로 원문 인용 불일치를 기록·임베딩 경고에 보존. 실제 권리 순위·배당액 계산 및 불일치 인용의 적용을 확정하지 않아 부분 확보 유지.",
        [c['chunk_id'] for c in chunks if c.get('source_id')=='easylaw_housing_distribution']+law("민사집행법","제84조")+law("민사집행법","제88조")+law("민사집행법","제145조")+law("민사집행법","제148조")+law("민사집행규칙","제48조")+law("주택임대차보호법","제3조의2")+law("주택임대차보호법","제3조의3")+law("주택임대차보호법","제3조의5")+law("주택임대차보호법","제8조"),"공식 배당 안내의 규칙 제91조·법 제148조 제3호 인용 대조 및 개별 등기·권리순위 확인")
    parcel = "택배 분실"
    parcel22 = ["ftc_terms_11185_file_a321590d_terms_article_22_0000"]
    add(parcel,"parcel_loss","택배가 분실되면 운송장 가격과 영수증 중 무엇을 기준으로 배상하나요?",
        "가액 기재 여부;손해액 입증;사업자 과실과 예외","supported","2020년 약관 제22조의 멸실 배상 부분 확인. 같은 조문의 특정일시 지연 단위 경고는 보존하며 멸실 계산과 구분.",parcel22)
    add(parcel,"parcel_no_value","운송장에 물품 가격을 안 적었다면 분실 배상 한도는 어떻게 되나요?",
        "가액 미기재 한도;할증 요금;고의·중과실 예외","supported","제22조 제3항과 제4항을 함께 확인. 모든 상황에 한도를 일률 적용하지 않음.",parcel22)
    add(parcel,"parcel_deadline","택배 분실이나 파손을 알렸는데 배상 청구 기간은 언제까지인가요?",
        "일부 손상 통지;전부 멸실 기산점;은폐 예외","supported","약관 제25조의 손상·전부 멸실 차이와 예외를 확인.",["ftc_terms_11185_file_a321590d_terms_article_25_0000"])
    add(parcel,"parcel_payment","택배 분실 영수증 등 입증서류를 냈으면 언제까지 배상해야 하나요?",
        "입증서류 제출 시점;우선 배상 기한;허위 서류 예외","supported","제22조 제5항 확보. 약관 적용·서류 사실확인과 실제 지급은 별도.",parcel22)
    add(parcel,"parcel_special_delay","특정 날짜에 필요한 택배가 늦었는데 운임액의 200으로 배상하면 되나요?",
        "특정일시 지연 단위;원문 이상;안내서와 대조","restricted","약관은 단위 없이 200으로 끝나고 안내서는 200%로 설명. 원문 경고가 있어 임의 보정·계산을 금지하고 확인 보류.",parcel22+["guide_94e9d4bd30eadf0b_p0028_body_0000"],"공식 약관 원문 단위 확인·발행기관 대조;자동 환산 금지")
    consumer = "소비자 피해"
    add(consumer,"consumer_defect","온라인으로 산 불량 제품의 교환이나 환불을 거절하면 어떤 근거를 확인하나요?",
        "거래 유형;계약 불일치;환급과 반환 비용","supported","온라인 구매로 범위를 한정하면 전자상거래법 철회·효과 조문 확보. 오프라인 모든 제품에 일반화하지 않음.",law("전자상거래 등에서의 소비자보호에 관한 법률","제17조")+law("전자상거래 등에서의 소비자보호에 관한 법률","제18조"))
    add(consumer,"consumer_mediation","소비자 피해구제와 분쟁조정은 어디에 신청하고 어떻게 진행되나요?",
        "접수 창구;합의 권고와 조정;필요 자료·절차","supported","기존 KCA 피해구제 정의·제외·절차를 재활용하고 공식 신청 요건·증빙·온라인/방문/우편 접수 및 조정 안내를 추가 연결. 소비자기본법 제55·57·58·65·66·67조와 시행령 제44조의 신청·합의권고·기간·조정·효력 근거 확보. 일반분야와 전문분야 조건, 온라인 6개월과 소멸시효, 수락/수락 간주 및 불수락을 구분. 안내 작성일은 미확인이고 신청 화면·첨부 양식·실제 집행은 검증 범위 밖.",
        ["guide_383b62bce56bf1ce_p0035_body_0000"]+[c['chunk_id'] for c in chunks if c.get('source_id')=='kca_damage_redress_procedure' or
            c.get('source_id')=='kca_redress_requirements' or
            c.get('source_id')=='kca_redress_submission' or
            (c.get('source_id')=='kca_mediation_procedure' and c.get('unit_id') in {'definition','request','acceptance','completion'})]+
        law('소비자기본법','제55조')+law('소비자기본법','제57조')+law('소비자기본법','제58조')+
        law('소비자기본법','제65조')+law('소비자기본법','제66조')+law('소비자기본법','제67조')+law('소비자기본법 시행령','제44조'),
        "신청 시 최신 접수 주소·화면·분야별 첨부 양식 재확인; 구체적 기간 계산·사건 당시 법령 및 실제 집행은 별도 검토")
    add(consumer,"consumer_repeat","수리를 반복해도 같은 하자가 생기는 제품은 모두 환불받을 수 있나요?",
        "품목별 기준;수리 횟수·보증기간;교환·환급 구분","partial","기존 반지 사례에 별표Ⅱ 전자제품·사무용기기 및 귀금속·보석 표를 추가 연결. 전자제품의 반복 하자·보증기간 조건과 귀금속의 분쟁 유형을 구분해 보존. 모든 품목의 기준을 확보한 것은 아니며 XML 표의 문자·병합 관계 대조 필요.",["kca_case_1003604873_case_reasoning_0000","consumer_dispute_standard_product_electronics_office_0000","consumer_dispute_standard_product_jewelry_0000"],"서비스 대상 품목 확정 후 나머지 품목의 원본 정제 확대 및 공식 별표 대조")
    add(consumer,"consumer_warranty","영수증이 없으면 제품 품질보증기간과 구매일을 어떻게 판단하나요?",
        "품목별 보증기간;구매일 증명;영수증 없는 경우","supported","기존 고시 제3조·별표Ⅲ·Ⅳ에 소비자기본법 제16조와 시행령 제8조·별표1 일반적 기준을 연결. 구매/인도/교환일부터의 기산 원칙, 판매일 확인 곤란 시 제조일/수입통관일 이후 3월 기준 및 제조/통관일 미표시 시 사업자의 판매일 입증 예외 확보. 시행령 별표1의 해당 PDF 페이지는 시각 대조했고 기준 전체를 단일 청크로 보존. 고시 별표Ⅲ·Ⅳ의 공식 PDF 4페이지를 시각 대조하고 품목·기간·공통 조건·빈 셀과 병합 관계를 보존. XML 원문은 별도로 보존하며 구매일 증명 질문의 기본 근거를 확보. 계약별 적용·최신 개정 전수 검증은 별도.",
        ["consumer_dispute_standard_article_2_0000","consumer_dispute_standard_appendix_0003_0000","consumer_dispute_standard_appendix_0004_0000"]+
        law('소비자기본법','제16조')+law('소비자기본법 시행령','제8조')+
        [c['chunk_id'] for c in chunks if c.get('law_name')=='소비자기본법 시행령' and c.get('unit_title','').startswith('일반적 소비자분쟁해결기준')],
        "별표Ⅲ·Ⅳ 수집본 대조 완료. 개별 제품의 보증 조건·계약 및 적용 시점·최신 개정 확인")
    add(consumer,"consumer_criteria","소비자분쟁해결기준은 바로 강제되는 법적 배상 기준인가요?",
        "합의·권고 기준 성격;별도 약정과 적용","supported","고시 목적 조문과 택배 안내서의 별도 약정·합의 권고 설명 확보. 실제 집행력·소송 효력 전반은 별도 검토.",["consumer_dispute_standard_article_0_0000","guide_94e9d4bd30eadf0b_p0005_body_0000"])
    return bank


# 최초 내용 검토 시 선정된 근거 본문 해시. 이후 변경 시 자동으로 재검토 표시.
REVIEWED_HASHES = {
    '003572_286997_20260701_art_40_0_chunk_0000': '20f612e9798d752daa1b16e537fa95835abd21ca03d52f105f8ae7f03999ca8a',
    '003572_286997_20260701_art_41_0_chunk_0000': 'ef3e4f0b7439c7cb92c4fcc7e8904734bb61cee28509e0dc26eaa5a97d886669',
    '007196_284567_20260317_art_22_0_chunk_0000': '2b6c8b1fadc1d4ab882c07778710ab79cdc4c2bc56d566d02213a83aa5a59aaf',
    '008480_282351_20260102_art_2_0_chunk_0000': '0716c1d960b43740c68ca30d1f77bca8fa7e8f8442ffe7a1503bd79bda212262',
    '008480_282351_20260102_art_3_0_chunk_0000': '6d7fbbda2799e9359280ee44b160b200511e4d1214affd544506b4cf95fd9ec7',
    '008480_282351_20260102_appendix_000300E_chunk_0000': 'cc636ef4dedd28e0fee194491b8b472d106e22d7034dfff89d125d731ec5dc10',
    '008480_282351_20260102_appendix_000300E_chunk_0001': '0065c38bc7c501e8468bfad1390810f2e5727040db812f20e4ab756dab2df2b3',
    'minimum_wage_simple_labor_article_0_0000': '6ac19228471f7c065236c489e668d510cac6531809b2f871955cf1106c0f7d3f',
    "000129_218303_20200526_art_3_0_chunk_0000": "efcb703bc4c023f2ca8d9bb1c10b52457765cb3cc5b7cfb3a755c2826bae6dbb",
    "000129_218303_20200526_art_5_0_chunk_0000": "0d1f3ed006b676e038be6df1f351adeaad618229724d205d69ffb3abccd8b83b",
    "000129_218303_20200526_art_6_0_chunk_0000": "e702874587bfdb93f134153750afb4e874b93d8af4e3f38fb3f058c97f0fd9a4",
    "000129_218303_20200526_art_6_0_chunk_0001": "e8e55543be136e0c9dba879deeb37ca8cebc4bd5210388498700d85fbabcd44a",
    "000129_218303_20200526_art_7_0_chunk_0000": "c147a3b2abab3f9b8a7f6dca80cb253657478ec9e2417f9391a41eed43d75129",
    "000129_218303_20200526_supplement_2018061215666_chunk_0000": "49bc47fcfeada4a2b1203156de630aa97023bb7039fb87a25eb7d9f39b14aba9",
    "000354_290239_20261002_art_2_0_chunk_0000": "98f693592b4fd9c5f20ff3f20f1879fbb72d0ac346f1f1bf65041ea70b6ca82a",
    "000354_290239_20261002_art_2_0_chunk_0001": "d77cb1cc0765606e81366d57e92012e93a73e45fbd528d06afe0e09be0b0ae53",
    "000354_290239_20261002_art_2_0_chunk_0002": "a3cfcf8ef310c3cc722f3dc4ccf76eef488d7d4facfd8f2cdce170f89e281af1",
    "000354_290239_20261002_art_31_0_chunk_0000": "aa5baa3892c32ff9e58bc84c0eb6aebebbfcf6399c8030387e929c229be979de",
    "000354_290239_20261002_art_32_0_chunk_0000": "dfd03a04aa7380b773cf45d303bd63a865e6acffb334f41ae9dbca91790157d9",
    "004950_287183_20260701_art_10_0_chunk_0000": "d9c1679dd5c116dd3f997e3a0ed41a802a40e0ab3a575823b6c65d952c774779",
    "004950_287183_20260701_art_11_0_chunk_0000": "0ebada2e5db79df94c16157880bd5de57ed1ac1761c96bc1c7daa78b076b6d5f",
    "004950_287183_20260701_supplement_1984061411441_chunk_0000": "ce63a3bc76dd4ba1005fae66320e4ca57c9933f3eb52eba8fa151808c796ed64",
    "004950_287183_20260701_supplement_1987120112283_chunk_0000": "0fe785604a61d213b17165d6d4dae93e024a377f11aec3909fb0aa989dc442b8",
    "004950_287183_20260701_supplement_1990021912930_chunk_0000": "e00a457efcb7a478230b0075b8b5cf4e3d4f10f10c9fc1d0156f26c3378d6519",
    "004950_287183_20260701_supplement_1995101914785_chunk_0000": "193d1791402edfde26b55db1b2f06165d497982947953b695ecd52510267d411",
    "004950_287183_20260701_supplement_2001091517360_chunk_0000": "61cf757d6f0ac9bdc48cac2cef90ac8e71dd87edbc09615d2f6b0634af4f3010",
    "004950_287183_20260701_supplement_2002061917627_chunk_0000": "9c26365a54f292eaa87a42b6c233572ab047d6f1f90b51377d6d3dd4eb9a4ead",
    "004950_287183_20260701_supplement_2007102320334_chunk_0000": "bef4e0c285bcec2788adc84e73a8376d5ad8d380a84de99359859acb215e70bc",
    "004950_287183_20260701_supplement_2008082120971_chunk_0000": "1925eb30de33f88e9b6739c11bbd3dd951368ca61d86422a485b31940d188148",
    "004950_287183_20260701_supplement_2009073021650_chunk_0000": "1a3acf551b1aed7feec6c84342b4b8edc63385652cf37de634a8f7cb428c6099",
    "004950_287183_20260701_supplement_2009092121744_chunk_0000": "94bed1f88227317bc7d2cae2c902cb10173fa270c4b1aff3b76b27e8fa79e37a",
    "004950_287183_20260701_supplement_2010072122284_chunk_0000": "dac088b019d4eb60390f94958ff20253f26bf11f05f014e05a7061caff054efb",
    "004950_287183_20260701_supplement_2012010623488_chunk_0000": "fb86d729bde00294fc5affbe240e61f890cb7b1aa176e338db96c9858f7ab2bf",
    "004950_287183_20260701_supplement_2013032324415_chunk_0000": "e70f579ee6487bc52ff9cae48cd4e6ffda57965f300432030be91d87a7f6dc67",
    "004950_287183_20260701_supplement_2013123025035_chunk_0000": "a1d5faa92f48f93b1cac56718afe69961e945b23571036413d0ed89113fb99af",
    "004950_287183_20260701_supplement_2016012226922_chunk_0000": "4cdbbaf7b0bc838c40bc643d761027a7479b228b47d1ddc9a738efebc4f321c3",
    "004950_287183_20260701_supplement_2016033127078_chunk_0000": "bddc17a6a0dcc2f38540355c1ec43d099d958f4623757add80902a0f6e8139f8",
    "004950_287183_20260701_supplement_2016112927614_chunk_0000": "1459b81fae29ef0c97f50ddf1a3786b4f192464356bc793f73ad3e08f6027f98",
    "004950_287183_20260701_supplement_2017052928053_chunk_0000": "ea538eea38d6f689c2b3ee4bca02ad916e2eaba691f394e2913573e083a2def8",
    "004950_287183_20260701_supplement_2018091829162_chunk_0000": "aed5928458d686ba1a8a4d5de8053f09337dc712dfdbd152a373f7fed04ecbb4",
    "004950_287183_20260701_supplement_2020092931080_chunk_0000": "10d73df826cf35b69398574ffde354026a4f1529e3b0c8b750e026bd3b515fc9",
    "004950_287183_20260701_supplement_2020120831243_chunk_0000": "b795ce7957f410864718b0f948bf6b61d3271ab6c5e7bcd14d834b0173ba19c1",
    "004950_287183_20260701_supplement_2021040631614_chunk_0000": "01bc8e2f6b67109ce183728a3090fb848f82b32c089cc0be0334f9c1ee54b320",
    "004950_287183_20260701_supplement_2021051131673_chunk_0000": "807f11767256a7ef551473393d02931a422773ee95630126a0dbc741ead88b0b",
    "004950_287183_20260701_supplement_2022122033105_chunk_0000": "6d6cf36414afbba82cc2a1662266d68d0c80bc6da16c58f2c068149b1494d930",
    "004950_287183_20260701_supplement_2023022133254_chunk_0000": "479df97ea8a35684e50792f03ac4fa94a08522c897d1ac34637d53c105171fdc",
    "004950_287183_20260701_supplement_2023092633771_chunk_0000": "0f34706f2275be5924614cda0cee9e408b886306b3a9b5834d70787c88da79e3",
    "004950_287183_20260701_supplement_2024060434550_chunk_0000": "da8eacd7ad801b8bf9c4696208f9b209aa2152c0c13c742eba729737f27f749e",
    "004950_287183_20260701_supplement_2024122435089_chunk_0000": "10ccb31f2c04173b68f3acb4c3a2e458364a68d06237cf23445e845128843b46",
    "004950_287183_20260701_supplement_2024123135161_chunk_0000": "a371ab5cc497ba4453f438ec5b17d090e0cd3f34c06068d0ad7b7c52cbe6bd72",
    "004950_287183_20260701_supplement_2025123035947_chunk_0000": "9d2e3ad465a46657bc84b721da313138a1f4be1993ef9c0414c90b6001464be1",
    "004950_287183_20260701_supplement_2026062336423_chunk_0000": "d6d11b0a6fd3d94ea71e1627d43cd0a1378c164b538877c78806ffd04da36b3f",
    "005247_206564_20190101_art_3_0_chunk_0000": "f0cb43a3753f12d3920c32a93efcc4e29192e47db4fc6aca72df091de16d3c8a",
    "005247_206564_20190101_art_5_0_chunk_0000": "52a0b8da571748bfdc1068ea6426e92356fd688b6c24facd63db9d12e4b2dad7",
    "005247_206564_20190101_art_6_0_chunk_0000": "83ab561e3d612dea5f47da184852cd830a858c2d5b94eaaadd41d7def9c67608",
    "009338_288143_20260721_art_21_0_chunk_0000": "0057539cbcb6620ad2fd79b8d594ac8cbfaeb8961943738d0ce210c7bc6d3c8e",
    "continual_contract_refund_article_0_0000": "2a02d319bdab0f6131c01ae4fc13dea89a5cde82f3eccdbda7c7d17ef0f06ab2",
    "continual_contract_refund_article_1_0000": "bca0909b9cdfc026db4b4ee2635a3f0631e32a58bca57fc8d32762460f066329",
    "continual_contract_refund_article_2_0000": "13072e1ef3422c815467108e2267614f92a03958a76ecb36d76ec1e0f929cf22",
    "continual_contract_refund_article_3_0000": "b480cddbccf5b7e8513b6ea7747556f4c393cc25e456133e952754b6bbe7b93e",
    "continual_contract_refund_article_4_0000": "29807ffa4ab3cb7c6b9bafccbf9ea795cd9c1011b26a7ae4376d2becd3c5c75e",
    "continual_contract_refund_article_5_0000": "66df4aab4e5a988797952c34486996c335e8811278d69707fdb8ecd8d494b7c4",
    "continual_contract_refund_article_6_0000": "ef2d7610629a3276db61917c7360087a14893412695acfa2b0999032b2fad554",
    "continual_contract_refund_article_7_0000": "25d5c8e3857373f99540ee7872ec2779d80f5fbd9a616979a2c2f7222650b721",
    "continual_contract_refund_appendix_0_0000": "63916d03c6ba2c8284a474ae6c7e83c2b103996fb3cbcd498680aaeee0ae85e7",
    "minimum_wage_2026_notice_body_0000": "8d5cc011a20682ef495eef79adf27917634d733135e749cff8dcc3d7bb700ce7",

    "001706_284415_20260317_art_140_0_chunk_0000": "e625b80f84ad1e2713a7205637a40c4969480cd81d4afa2b658f4875ce25f351",
    "001706_284415_20260317_art_141_0_chunk_0000": "fde13942b75b9a3d69476764322eac59b45f1687306fc19ee91e9625a5abacc6",
    "003058_270551_20251023_art_7_2_chunk_0000": "5ba43af3594ff65d53f1b262b7432e7950da66f3dca00b5713e6f7bafbedd0ab",
    "003058_270551_20251023_art_7_2_chunk_0001": "4c69a0fa29f1bc2b22ddc359d335adcd832cc04904668894f2b0657ad1926ea9",
    "consumer_dispute_standard_product_jewelry_0000": "46fde5d4a269104195acb57a13d345d5be5a34889577ed8c031603c162ac667c",
    "consumer_dispute_standard_product_electronics_office_0000": "21f18c3101e528e7be10912b0224df8d8cf6fb1ed15bb7d30488cffa8f1ea906",
    "consumer_dispute_standard_appendix_0003_0000": "c818b0e0c7a7ac2e2617bd90cb2f281f27283b57195d3c48a1f44664f92c8a27",
    "consumer_dispute_standard_appendix_0004_0000": "c4c21728f3f61770a307416c222013b30a85f81fedcf963ea5774b8a2821991b",
    "001248_276291_20260102_art_3_0_chunk_0000": "555073c96bf2234af3a99dfe47d671b6d40b6a063a8e55e393772c860acf096d",
    "001248_276291_20260102_art_3_2_chunk_0000": "47104f5ec1ef5f77aab693f9f3528b58e88221b07eaa039a6a0fdb3f063cef2a",
    "001248_276291_20260102_art_3_2_chunk_0001": "01477b22897898489a2cde4a63010bcebdd04dcb8480bbb5b83f391e76de8cdb",
    "001248_276291_20260102_art_3_3_chunk_0000": "e223211ee8f2a6a1f8b2a5875f3bee81dadc1950fc86b05120dbbc2f4ab465a9",
    "001248_276291_20260102_art_3_3_chunk_0001": "10917253631e9bdd2716577f733455d40af27c4eaa6ef70f5ad2a5c43deb90f5",
    "001248_276291_20260102_art_6_0_chunk_0000": "8c15d993c444ad543a7b3608100d6987369f462f4c8c1467cfd15595612cf44e",
    "001248_276291_20260102_art_6_2_chunk_0000": "3a7d2b85991f7c16df640d9c763516c43d9fe4e984aff7ed13e07e6646b93bab",
    "001248_276291_20260102_art_6_3_chunk_0000": "28a61ceaab7198913375c45a089a98f8d668d79fe4c8a14d624980a8cc4c1d59",
    "001248_276291_20260102_art_6_3_chunk_0001": "d9c051ebd27379f0b964555f7de68330ef64adeb88461b0751b5b4ca784879d5",
    "001248_276291_20260102_art_7_0_chunk_0000": "1cdd1538eecbcc2bb55172417b804b225bc8d775fe580214677929655f89ada2",
    "001248_276291_20260102_art_8_0_chunk_0000": "2ae06b605e461fe68eb3b7363bfa33b3d357d8309edf3e9ef0e81d908d687560",
    "001692_290185_20261002_art_347_0_chunk_0000": "ef14920c2e3d04c54b4d957feb77054640825072c03c6ed2b725ded49e4cfdd5",
    "001706_284415_20260317_art_17_0_chunk_0000": "cd6b66f1558a4487d77f8ee16f13f801dce3194f0f2880c4f39a9624fb89525a",
    "001706_284415_20260317_art_379_0_chunk_0000": "f15275169244e7966b47160454cd8e2938d1f8b8f1736edf104cfb83182aaa4a",
    "001706_284415_20260317_art_575_0_chunk_0000": "0913ce5e8b257a804243e7f24d2e36a438207ad59dbc2d84834b877e5c7919d5",
    "001706_284415_20260317_art_580_0_chunk_0000": "240a268f6f5313fec43996283f8fea9d46235fc01c735b59d5089ab165e0faf7",
    "001706_284415_20260317_art_582_0_chunk_0000": "69b9b3f065ab47bc30c0b5320340e0f6b854ecbfeccb2959c419531e854717a4",
    "001706_284415_20260317_art_5_0_chunk_0000": "e303f393b2fd6d22ca06c58b9e372f007a8d8935601618a7577c9b3e2d3bce09",
    "001706_284415_20260317_art_623_0_chunk_0000": "d53633aafa631d5b1fa9cf28e8cac6e9424a7b68120626c5f1654fe5e58979d4",
    "001706_284415_20260317_art_6_0_chunk_0000": "60e8767ecf70f5c6ae590423b1817e77dc8a60d83698468e5c6e1a64e4ef960b",
    "001872_285279_20261008_art_11_0_chunk_0000": "27b2fba424fe592ec5ff90f95941bb693712ab49f1f5f7d3c8060c3b169980cf",
    "001872_285279_20261008_art_18_0_chunk_0000": "eaa03a0f6b7762dbcc6ca9900b7ce0b46ac90b8448aff73c2be73929d06aa794",
    "001872_285279_20261008_art_23_0_chunk_0000": "efcaa1cbd34bd547f2104ee8517d024375a0bbee64e756bd82eae49e04aeda42",
    "001872_285279_20261008_art_26_0_chunk_0000": "c5db4569f75ff6a727acec7c03ea81f33232a1b3d46c71b2cc15a6bfe9abc2d2",
    "001872_285279_20261008_art_27_0_chunk_0000": "ebabeabd14196f30ddbd57e29de826c4a2ac7be4ce2bd2f573fbed001910c84b",
    "001872_285279_20261008_art_28_0_chunk_0000": "138660b8f86901d0e5b264f06f01db00db301de6902254f4b92212f9747cae81",
    "001872_285279_20261008_art_36_0_chunk_0000": "1a407952c492676358a540b100de3c0448a95a3e693d009917110ffb78ef2d7e",
    "001872_285279_20261008_art_55_0_chunk_0000": "65b9b4c6a067b607bfc43d8454373cf906981a83b74e947820cea19c9e24facc",
    "003058_270551_20251023_appendix_000100E_chunk_0000": "4d9ec3f9db2cb40667839305d59f0ba25fb792e71d3b4037b333ffd6e958ef8e",
    "003058_270551_20251023_appendix_000100E_chunk_0001": "f685cba56fe8479f993fa0be878e0145d6d982952ff78f91efeb80a6750de3f2",
    "003058_270551_20251023_art_30_0_chunk_0000": "ed484d7ca6ec40a84d91f5f61a2c115e0ff4990bb6733bf2cc7aa1ce053d7ec6",
    "003058_270551_20251023_art_7_0_chunk_0000": "0919365ca3a775a6cbb25af2fcb9c7c37a7cdfb52fa9e8810b4acb21006403f6",
    "009318_282793_20260721_art_17_0_chunk_0000": "5005bbc753a5276415385862a99c63d0e95d5f7fdbb2550c7b555f5c3206e086",
    "009318_282793_20260721_art_17_0_chunk_0001": "91d76ac1e4b731d796780a4afcc72894082453cc65181633f5fcdb7c034faab6",
    "009318_282793_20260721_art_18_0_chunk_0000": "ddd98c1e858fa59454aff22b29bf1968fe1c73df3826a90cf57e838bf77aa695",
    "009318_282793_20260721_art_18_0_chunk_0001": "736c4623e773a9d5e4a47c1bed2db3184c2ab326fb00759b9bf63ba488fc5886",
    "009883_284455_20260918_art_4_0_chunk_0000": "da4a39f7018846239fd3ea375e5babe2caf28295e18dd1057398f2fd7cfb1c21",
    "240967_0002": "987b2f5b398bb2d2110e775cfd75cd68dffa049c787c2576451f1f36029bfdf5",
    "605583_0001": "5b4e3aef20d8ce35ac3b80dc929dfec19c7ca4f3a9cd273c368fcf98171a3fad",
    "consumer_dispute_standard_article_0_0000": "3b837c475a89ee680a1257b2c6df08ea31d1d3fd39e91e5565a774a2d5e3a8d4",
    "consumer_dispute_standard_article_2_0000": "9d9c3819815aa20642f8100549226c9d7e07c2bd1328e4fe3f5fe0c57451f19b",
    "ftc_terms_11185_file_a321590d_terms_article_22_0000": "05d5aaa0882e3b84faf5623f94706dada7850a81e71341bb0a96a5dee61a254c",
    "ftc_terms_11185_file_a321590d_terms_article_25_0000": "548c9da87e48046315647abe88e307bb13fc9cb4e55154c6e31449cba8ec95c4",
    "ftc_terms_46118_file_5b139f26_terms_article_7_0000": "15e1beb54acba93d86388b64ec6ddae3a115c3ff2069cb38a27eb7f0fa8877b0",
    "ftc_terms_47762_file_05d51a47_terms_article_8_0000": "9a84960a030c71cd63d8f3096230642d92c8772f63bd65eeb1f6efb3ddebeb5b",
    "guide_23ffe4c8fd5715cb_p0031_body_0000": "d1d80ab7fc96828a2ce4ee443cc02a5cff868ce8f4fc530e0036d7e8572c47a3",
    "guide_23ffe4c8fd5715cb_p0045_body_0001": "0a198e33464393311eb639a18ade7058c6540847168e9dba0157e5722c7b331a",
    "guide_23ffe4c8fd5715cb_p0046_body_0000": "11b69d14176a535f1db7610a719c56de88e82a7b53b0dba46f47674e87f793d7",
    "guide_23ffe4c8fd5715cb_p0047_body_0000": "08d349fe0c1405c7f8ae0efbda330d9709c621c19983e722d6e8f03fa61c0b8d",
    "guide_23ffe4c8fd5715cb_p0047_table_0000": "80283958ac8532b5952e0b33a8988b93be9dae94fa29286c58514710ff165498",
    "guide_23ffe4c8fd5715cb_p0067_body_0000": "e8d5979ffb00670c967b73eded3d0b534abe29f4c5f997345b05b9bb1b0fab83",
    "guide_23ffe4c8fd5715cb_p0068_body_0000": "c0af5585a8eadf3b305d6fad2fdec98ddaae65531f907371ee9c8688c3547e76",
    "guide_23ffe4c8fd5715cb_p0076_body_0001": "0bbce1ec0b5c7b2591293577e736107b6102606f91c709ca1ac3fb8c234be6bb",
    "guide_246eabd0676d5957_p0007_body_0000": "7ae479a489ea011ad7b52953dd9f9dd13472e8ca5b188e4b580ed9515962c639",
    "guide_246eabd0676d5957_p0010_body_0000": "db963b1c3d64ed36cd67a18ada5fd18b9751f6ad21bbd7af74dd80459a2055a4",
    "guide_246eabd0676d5957_p0018_body_0000": "9ad775ced59c45947a40013600256ac2fcd111b2be77759f0c4bc60a5ed9044b",
    "guide_246eabd0676d5957_p0019_image_summary_0000": "1f45e848dd7a70ab1241345fa87f1bd07ceca09f975f2f0f506e3f82bb17af08",
    "guide_246eabd0676d5957_p0022_body_0000": "97758a180005571bc36a4121a30459c1e166f7591be3a41b6a8e10a075a49e78",
    "guide_383b62bce56bf1ce_p0032_table_0000": "2b926126106700a668bd6bec8d33f966fa6d4fa5699f13461daaa964ca5dc061",
    "guide_383b62bce56bf1ce_p0035_body_0000": "0a50c806bb952e2f71dce3677d8867046dd5c76303214e2abbcab7bd1212cb17",
    "guide_94e9d4bd30eadf0b_p0005_body_0000": "aa982d50d8e5883cddfd1a35a93a473661ab6606f0a65b1d1234f036a702c4f8",
    "guide_94e9d4bd30eadf0b_p0028_body_0000": "184211d01f54e4f41913efd6645e0ec088a8f555970c09000ecc4ea595945270",
    "guide_cad7b1de1e4dc64c_p0013_body_0000": "cb28879d47678b7cf0ea9fbf36c8031b3b5323900d79a0e0d94a592702413102",
    "guide_cad7b1de1e4dc64c_p0014_table_0000": "7e423e8bf7db89bffdddc49a3b1af8dff7867334fc5504a28e014d06c4868a12",
    "guide_cad7b1de1e4dc64c_p0078_body_0000": "e10c989dad85d9024000bccda6fea1db6b09eb82c719a903762773ed71d31731",
    "kca_case_1003164054_case_reasoning_0000": "cdcb3a8633ac12baddd00fbb37a2e8f44881a93a19bbc3553a6fe69c2d42ed91",
    "kca_case_1003604873_case_reasoning_0000": "20aeeae0ea4a2b484cd350211a86e987d88daf569057158709159cb737c9529a",
    "kca_case_1003640779_case_decision_0000": "e3f61c0f686e108808a356eac88194c2bd3998e5c6e8e9cbf8192a62ad6caf3a",
    "kca_case_1003640779_case_reasoning_0000": "97aedb04aa339c2f967124841febd3874e29b5a04eaece10be62910a4bbb79e0",
    "kca_case_1003641715_case_reasoning_0000": "88656de5d1e22eac1a4f7700c5b9915acf0cc2eff7315bf972da5ebd7bb0f68a",
    "kca_case_1003739114_case_reasoning_0000": "226c58ba0bee5a25587b7f3e10497bedc40ecec8230f0fe7a60532d83fef486b",
    "kca_case_1003739114_case_reasoning_0001": "411e844d4a7ce6f1edcecc2a0efb1cb7202e2a66fcabd32b7829320bbc52ad2d"
}
REVIEWED_HASHES.update({'moel_wage_procedure_procedure_0_0000': 'f70d51749138fb74ac087e60cd9ac3d63c43d0c112131084ed297e0786f48ae7', 'moel_wage_procedure_procedure_1_0000': 'ef517a66fd31fb53d14051c05a876f622eabde002c3e3ca87848121b7c2b8724', 'moel_wage_procedure_procedure_2_0000': 'ca9083b79b389f2d4bba9dc11c9285f8e3c4d3e00cc3ef6d967330bf49549104', 'moel_wage_procedure_procedure_3_0000': '6aaf0f982ae13888cea0d95ea45ecdd7dbf5975bf55acf9deded2b2b8f59e0f3', 'moel_wage_procedure_procedure_4_0000': '29f40c11e6a4d52d98474829b3da3ce5a5aa40e7a3a7e34f64c0be06b0584229', 'moel_wage_evidence_1000323154_evidence_0_0000': 'de9041799ff8a353be7889c38f92f4b21cb78283d7c9faa7487296d399cc0462'})
REVIEWED_HASHES.update({'kca_overseas_types_definition_0000': 'aab1ebd3bee70accafebd4355d2948d628f1d452878f9000a5aa50d24517e1d0', 'kca_overseas_types_type_direct_delivery_0000': 'af3baf37935cfd36d5eeeb3c8da2463d759f07c0e7a7b0c705db015da79f8a29', 'kca_overseas_types_type_forwarding_0000': '4961878b066cb7267770a0d448694f32e1eab3ed2e31bd791f16158fd3a3faee', 'kca_overseas_types_type_purchase_agent_0000': '8205aa5951a82b80898eb92a5ac709080c2da736f7f7ddfcfd01bf76c019ba21', 'kca_overseas_types_process_direct_delivery_0000': 'c2858026279bcd363031d685ff2235608f0181fed987eed1c62d00650c16d6f5', 'kca_overseas_types_process_forwarding_0000': '671551b68ee55ba736a399cf26f29dd9cee46a83f78b5e79a7a81738e4940174', 'kca_overseas_types_process_purchase_agent_0000': 'd0e279ff69761fc911df51afb9653d453d81e9e90adb5f483e70492d83fddaca', 'kca_overseas_types_comparison_0000': '0f658ef21a223c9a6e4b4843266d4f0a0fe281b962cc968c8d31a474f804258e', 'kca_overseas_checklist_payment_evidence_0000': 'c6763a9d28e7a1c40669f73f675ec605be723cfab6f848ec115c205af9deb4e5', 'kca_overseas_checklist_direct_delivery_0000': 'cf22d567bd8a4e14dc4c0c71d90c54483744da4a84341ff6b118bb44c90b4464', 'kca_overseas_checklist_direct_contact_0000': '23d37289a3d180e1daa88531385c3a1814c10f7f9c5170e194a70873eba4a8ae', 'kca_overseas_checklist_forwarding_evidence_0000': '4c5cb567f3fa49a98b8832b7d4b7727970d2fa294d4c78820601d8b97b75f1b1', 'kca_overseas_checklist_purchase_agent_0000': 'bdb7de80dd5fdc4b19fdcac98c31eca011d40928b7d6d6aa80d7c167e430649c', 'kca_overseas_checklist_purchase_agent_contact_0000': '7034e2f6df74a2eb2bf1887de332c7f0919a5f431ae93cf05c2c59cbe7cb7cd4', 'kca_overseas_returns_return_conditions_0000': '543673301b5205ab6c040c9ecba5a1d167c6432f2f1ac65257a30f0aa1929f7c'})
REVIEWED_HASHES.update({'kca_fitness_closure_gym_closure_risk_0000': 'd972df3fd8b30f17eb6f98dbbe5f1148a8d2633df639142e27f3dad0d2545d1c', 'kca_fitness_closure_gym_card_evidence_0000': '73b66bce1cf042451b23b1c08fb47514f69f54d8cb6af229fe1667f38854558c', 'kca_fitness_closure_pilates_card_objection_0000': '24350444c0aa6c0c6511191da062a6e1e50d6ae6f06c4ce7814dfc0423af757a', 'kca_fitness_closure_pilates_closure_action_0000': 'c5ae7f3797376ca5e03fcabb9c6bc6d5743bce89ab37c7db331a76f3b5d54418', 'kca_fitness_closure_pilates_evidence_table_0000': '5cf0aa017576e00aa4d72244bb364e261d234337c4c36c11901feabaf5c91bba', 'kca_fitness_closure_pilates_consultation_0000': '53ea899392214da74e6cbbcbc5c84a184f1473dd377ea82346bcd459c77d0ee6', 'kca_damage_redress_procedure_redress_definition_0000': 'd10a2cf2e1aadb1b4ae7e959dad6d60dd0de4f156fb0e83956aa42d8cd2d4968', 'kca_damage_redress_procedure_redress_exclusions_0000': '60a88abc88f2188eb79ef0d46aed4abbf39d41bc11df02d851cb19c41c9d2070', 'kca_damage_redress_procedure_redress_procedure_0000': '53f728f89fe6bdb194e72bfc5fdd0520b2ff4a580c9d6d61cc5b133de749cd5c'})
REVIEWED_HASHES.update({'000355_289407_20260908_art_2_0_chunk_0000': 'e1e6f9326628012b3a87778d2186b4fb910ba4f9c367a869de6b321420d2d29c', '000355_289407_20260908_art_2_0_chunk_0001': 'e2312bdc85dc2dfa76efaef8e03bf3e168a22b6463471279be07b1eccb30d37a', '000355_289407_20260908_art_3_0_chunk_0000': '8cce8f37e04b4cfc51e01289ff20e5ef2360a1fa01594b029c40d1c11dd5aaa2', '000355_289407_20260908_art_4_0_chunk_0000': 'd35ef8bd665bac6f7ad28ed0439b63ba2ac4840831b8d6270c296f42169f8464', '000355_289407_20260908_art_16_0_chunk_0000': 'c0f3c011f4c06a947b0a6db08039b0b5a3060f99e836d07553a9cbea8d74121f', '005498_287001_20260701_art_2_0_chunk_0000': '4e963495e3a3e38be593af4eedd2e35ba193023ea894f5ec2cf74203e7845258', '005498_287001_20260701_art_4_0_chunk_0000': '03643eeca0caf895b13be3c94f0ea7dcaa724d1838035b523c2119dc7078c55a', '005498_287001_20260701_art_11_0_chunk_0000': '4d0faae2cffe3fd17fa615b797e63e7456e10f2c7ec48b0f63bae6a79be4d609'})


def audit():
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    bank = question_bank(chunks)
    index = BM25(chunks,evidence_mode=True)
    legacy = BM25(chunks)
    previous=json.loads(REPORT_JSON.read_text(encoding='utf-8')) if REPORT_JSON.exists() else {}
    baseline=previous.get('preprocessing_baseline')
    if baseline is None and previous.get('questions'):
        baseline={'input_fingerprint':previous['input_fingerprint'],'chunk_count':previous['chunk_count'],
                  'questions':[{'question_id':q['question_id'],'reviewed_chunk_ids':q['reviewed_chunk_ids'],
                                'review_status':q['review_status'],'chunk_hit':q['reviewed_chunk_hit_at_5']}
                               for q in previous['questions']]}
    supplement_baseline=previous.get('source_supplement_baseline')
    if supplement_baseline is None and previous.get('questions'):
        supplement_baseline={'chunk_count':previous['chunk_count'],
          'questions':[{'question_id':q['question_id'],'reviewed_chunk_ids':q['reviewed_chunk_ids'],
                        'chunk_hit':q['reviewed_chunk_hit_at_5']} for q in previous['questions']]}
    delegated_baseline=previous.get('delegated_source_baseline')
    if delegated_baseline is None and previous.get('questions'):
        delegated_baseline={'chunk_count':previous['chunk_count'],'search_policy':previous['search_policy'],
          'questions':[{'question_id':q['question_id'],'reviewed_chunk_ids':q['reviewed_chunk_ids'],
                        'chunk_hit':q['reviewed_chunk_hit_at_5']} for q in previous['questions']]}
    refinement_baseline=previous.get('search_refinement_baseline')
    if refinement_baseline is None and previous.get('questions'):
        refinement_baseline={'chunk_count':previous['chunk_count'],'search_policy':previous['search_policy'],
          'input_fingerprint':previous['input_fingerprint'],
          'questions':[{'question_id':q['question_id'],'reviewed_chunk_ids':q['reviewed_chunk_ids'],
                        'chunk_hit':q['reviewed_chunk_hit_at_5']} for q in previous['questions']]}
    records = []
    for item in bank:
        row = dict(item)
        evidence = []
        stale = False
        for identifier in item["reviewed_chunk_ids"]:
            c = by_id.get(identifier)
            if c is None:
                stale = True
                continue
            if REVIEWED_HASHES.get(identifier) != hashlib.sha256(c["text"].encode()).hexdigest():
                stale = True
            evidence.append({"chunk_id": identifier, "document_id": c["document_id"],
                             "evaluation_source_id": c["precedent_id"], "title": c["title"],
                             "source_url": c.get("source_url"), "source_path": c.get("source_path"),
                             "source_locator": c.get("source_locator") or c.get("article_label") or c.get("section_heading"),
                             "page_number": c.get("page_number"), "dates": c.get("dates"),
                             "excerpt": c["text"][:650], "text_sha256": hashlib.sha256(c["text"].encode()).hexdigest(),
                             "warnings": c.get("warnings", []), "source_anomaly": c.get("source_anomaly"),
                             "calculation_restricted_clause": c.get("calculation_restricted_clause")})
        if stale:
            row["review_status"] = "needs_review"
            row["follow_up"] = "선정 근거 변경/누락: 기존 검토 메모 재확인 필요. " + row["follow_up"]
        expected_sources = {e["evaluation_source_id"] for e in evidence}
        expected_ids = {e["chunk_id"] for e in evidence}
        hits = index.search(item["query"], top_k=5,**SEARCH_OPTIONS)
        old_hits = legacy.search(item["query"], top_k=5)
        row.update(evidence=evidence, evidence_snapshot_valid=not stale,
                   search_results=[{"rank": rank, "chunk_id": c["chunk_id"], "document_id": c["document_id"],
                                    "title": c["title"], "score": c["score"], "role":c.get('role'),
                                    "matched_span":c.get('matched_span'),
                                    "warnings":c.get('warnings',[]),"source_anomaly":c.get('source_anomaly'),
                                    "calculation_restricted_clause":c.get('calculation_restricted_clause'),"excerpt": c["text"][:450]}
                                   for rank, c in enumerate(hits, 1)],
                   legacy_search_results=[{'chunk_id':c['chunk_id'],'document_id':c['document_id'],'score':c['score']} for c in old_hits],
                   legacy_source_hit_at_5=any(c['precedent_id'] in expected_sources for c in old_hits) if evidence else None,
                   legacy_chunk_hit_at_5=any(c['chunk_id'] in expected_ids for c in old_hits) if evidence else None,
                   reviewed_source_hit_at_5=any(c["precedent_id"] in expected_sources for c in hits) if evidence else None,
                   reviewed_chunk_hit_at_5=any(c["chunk_id"] in expected_ids for c in hits) if evidence else None)
        if row["review_status"] == "supported":
            row["next_step"] = "현재 질문 범위의 기본 근거 유지" if row["reviewed_chunk_hit_at_5"] else "근거는 있음: 대체 검색 결과의 관련성 검토 후 청크 검색·문맥 전달 보완"
        else:
            row["next_step"] = row["follow_up"]
        records.append(row)
    summaries = []
    for domain in dict.fromkeys(r["domain"] for r in records):
        group = [r for r in records if r["domain"] == domain]
        summaries.append({"domain": domain, "question_count": len(group),
                          "statuses": dict(Counter(r["review_status"] for r in group)),
                          "reviewed_source_hits": sum(r["reviewed_source_hit_at_5"] is True for r in group),
                          "reviewed_chunk_hits": sum(r["reviewed_chunk_hit_at_5"] is True for r in group),
                          "questions_with_evidence": sum(bool(r["evidence"]) for r in group)})
    current={r['question_id']:r for r in records}
    prior_rows=baseline['questions'] if baseline else []
    comparison_rows=[{'question_id':q['question_id'],'before':q['chunk_hit'],
                      'after':any(h['chunk_id'] in q['reviewed_chunk_ids'] for h in current[q['question_id']]['search_results']) if q['reviewed_chunk_ids'] else None}
                     for q in prior_rows]
    fixed_supplement=[{'question_id':q['question_id'],'before':q['chunk_hit'],
                      'after':any(h['chunk_id'] in q['reviewed_chunk_ids'] for h in current[q['question_id']]['search_results']) if q['reviewed_chunk_ids'] else None}
                     for q in (supplement_baseline or {}).get('questions',[])]
    refinement_rows=[{'question_id':q['question_id'],'before':q['chunk_hit'],
        'after':any(h['chunk_id'] in q['reviewed_chunk_ids'] for h in current[q['question_id']]['search_results']) if q['reviewed_chunk_ids'] else None}
        for q in (refinement_baseline or {}).get('questions',[])]
    delegated_rows=[{'question_id':q['question_id'],'before':q['chunk_hit'],
        'after':any(h['chunk_id'] in q['reviewed_chunk_ids'] for h in current[q['question_id']]['search_results']) if q['reviewed_chunk_ids'] else None}
        for q in (delegated_baseline or {}).get('questions',[])]
    return {"delegated_source_baseline":delegated_baseline,
            "delegated_source_comparison":{
                'before':sum(r['before'] is True for r in delegated_rows),
                'after':sum(r['after'] is True for r in delegated_rows),
                'improved':[r['question_id'] for r in delegated_rows if r['before'] is False and r['after'] is True],
                'regressed':[r['question_id'] for r in delegated_rows if r['before'] is True and r['after'] is False],
                'questions':delegated_rows},
            "search_refinement_baseline":refinement_baseline,
            "search_refinement_comparison":{
                'before':sum(r['before'] is True for r in refinement_rows),
                'after':sum(r['after'] is True for r in refinement_rows),
                'improved':[r['question_id'] for r in refinement_rows if r['before'] is False and r['after'] is True],
                'regressed':[r['question_id'] for r in refinement_rows if r['before'] is True and r['after'] is False],
                'questions':refinement_rows},
            "source_supplement_baseline":supplement_baseline,
            "source_supplement_comparison":{'before':sum(r['before'] is True for r in fixed_supplement),
                'after':sum(r['after'] is True for r in fixed_supplement),
                'regressed':[r['question_id'] for r in fixed_supplement if r['before'] is True and r['after'] is False],
                'improved':[r['question_id'] for r in fixed_supplement if r['before'] is False and r['after'] is True],
                'questions':fixed_supplement},
            "evaluation_type": "reviewed_evidence_and_offline_bm25_domain_audit",
            "reviewer": "Codex의 수집 자료 내용 검토. 전문가 법률 검증 아님",
            "chunk_count": len(chunks), "question_count": len(records),
            "input_fingerprint": source_fingerprint(), "search_policy":SEARCH_POLICY,"summary": summaries, "questions": records,
            "preprocessing_baseline":baseline,
            "preprocessing_comparison":{'basis':'수정 전 선정 청크 목록을 고정한 누적 비교. 이후 원문 보강·검색 정책 변경도 포함하며 전처리 단독 효과가 아님.',
                                        'before_chunk_hits':sum(r['before'] is True for r in comparison_rows),
                                        'after_chunk_hits':sum(r['after'] is True for r in comparison_rows),
                                        'improved':[r['question_id'] for r in comparison_rows if r['before'] is False and r['after'] is True],
                                        'regressed':[r['question_id'] for r in comparison_rows if r['before'] is True and r['after'] is False],
                                        'questions':comparison_rows},
            "search_comparison":{'legacy_source_hits':sum(r['legacy_source_hit_at_5'] is True for r in records),
                                 'legacy_chunk_hits':sum(r['legacy_chunk_hit_at_5'] is True for r in records),
                                 'improved_source_hits':sum(r['reviewed_source_hit_at_5'] is True for r in records),
                                 'improved_chunk_hits':sum(r['reviewed_chunk_hit_at_5'] is True for r in records),
                                 'chunk_improved':[r['question_id'] for r in records if r['legacy_chunk_hit_at_5'] is False and r['reviewed_chunk_hit_at_5'] is True],
                                 'chunk_regressed':[r['question_id'] for r in records if r['legacy_chunk_hit_at_5'] is True and r['reviewed_chunk_hit_at_5'] is False],
                                 'source_regressed':[r['question_id'] for r in records if r['legacy_source_hit_at_5'] is True and r['reviewed_source_hit_at_5'] is False]},
            "limitations": ["수동 선정 45문항의 기본 근거 점검이며 전체 분야 커버리지 완료율이 아님.",
                            "기본 근거 확보는 명시한 질문 범위·수집본 내용에 한정. 실제 사건 적용·현행성을 보장하지 않음.",
                            "BM25는 점수용 본문 구간과 셀 문자·용어 확장·주장 감점을 사용하고 헤더 토큰은 본문의 0.5배로 가중하며 문서 버전당 최대 2개 청크로 상위 5개 반환. 기존 문서당 1개 방식과 동일 자료·선정 근거로 비교.",
                            "기대 근거 목록은 완전한 정답 목록이 아님. 미적중은 답변 불가의 증명이 아님.",
                            "선정 근거의 본문이 바뀌면 기존 내용 검토를 재사용하지 않고 추가 검토로 표시.",
                            "원문 전체의 추출 무결성·PDF 모든 시각 요소·최신 개정은 이 테스트가 자동 검증하지 않음.",
                            "임베딩·Qdrant·LLM 답변·환불 계산·실제 권리 판단은 평가하지 않음."]}


def render(report):
    lines = ["# 분야별 데이터 충분성 검토", "", f"검토 질문 {report['question_count']}개 / 검색 청크 {report['chunk_count']}개", "",
             "충분성 점수 대신 질문별 근거 검토 상태를 표시합니다. 기본 근거 확보도 전체 답변 품질·현행 적용을 보장하지 않습니다.", "",
             "| 분야 | 질문 | 기본 근거 | 자료 보완 | 적용 보류 | 추가 검토 | 출처 적중 / 근거 있음 | 청크 적중 / 근거 있음 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in report["summary"]:
        counts = s["statuses"]
        lines.append(f"| {s['domain']} | {s['question_count']} | {counts.get('supported',0)} | {counts.get('partial',0)} | {counts.get('restricted',0)} | {counts.get('needs_review',0)} | {s['reviewed_source_hits']}/{s['questions_with_evidence']} | {s['reviewed_chunk_hits']}/{s['questions_with_evidence']} |")
    comparison=report['search_comparison']
    lines += ['', '## 기존 검색과 개선 검색 비교', '',
              f"- 동일 자료·선정 근거로 비교: 출처 {comparison['legacy_source_hits']} → {comparison['improved_source_hits']}건, 선정 청크 {comparison['legacy_chunk_hits']} → {comparison['improved_chunk_hits']}건.",
              '- 청크 적중 개선: '+', '.join(comparison['chunk_improved']),
              '- 청크 적중 하락: '+(', '.join(comparison['chunk_regressed']) or '없음'),
              '- 출처 적중 하락: '+(', '.join(comparison['source_regressed']) or '없음'),
              '- 품질보증 질문의 선정 근거에는 추가 검토한 별표 Ⅲ·Ⅳ를 포함하므로 이전 보고서의 근거 목록과 다릅니다. 검색 개선 비교는 양쪽 모두 현재 근거 목록을 사용합니다.']
    change=report['preprocessing_comparison']
    lines += ['', '## 이번 전처리 수정 전후 비교', '',
              f"- 수정 전 근거 청크 목록을 고정한 비교: {change['before_chunk_hits']} → {change['after_chunk_hits']}건.",
              '- 개선 질문: '+(', '.join(change['improved']) or '없음'),
              '- 하락 질문: '+(', '.join(change['regressed']) or '없음'),
              '- 미성년자 취소·소규모 사업장 해고의 검토 상태 변경은 실제 근거 연결·내용 대조 결과이며 검색 적중만으로 판정하지 않습니다.']
    batch=report['source_supplement_comparison']
    lines += ["", "## 원문 7종 보강 전 고정 근거의 현재 검색 결과", "",
              f"- 보강 직전의 선정 근거 ID 고정(이후 검색 설정 변경 포함): {batch['before']} → {batch['after']}건.",
              "- 개선: "+(', '.join(batch['improved']) or '없음'),
              "- 하락: "+(', '.join(batch['regressed']) or '없음'),
              "- 새 근거 추가에 따른 현재 평가 결과와 구분하며 없는 근거가 적중한 것으로 처리하지 않습니다."]
    refinement=report['search_refinement_comparison']
    lines += ["", "## 검색 표현·헤더 가중치 수정 전후", "",
              f"- 수정 직전 선정 청크 ID 고정: {refinement['before']} → {refinement['after']}건.",
              "- 개선: "+(', '.join(refinement['improved']) or '없음'),
              "- 하락: "+(', '.join(refinement['regressed']) or '없음'),
              "- 본문 점수 비중 1, 헤더 0.5. 환급 행 검색 표현에 조건·수식·검색 별칭을 사용하고 공통 정의는 전체 원문에 보존합니다."]
    delegated=report['delegated_source_comparison']
    lines += ["", "## 위임 규정·직종 고시 4종 보강", "",
              f"- 보강 직전 선정 근거 ID 고정: {delegated['before']} → {delegated['after']}건.",
              "- 개선: "+(', '.join(delegated['improved']) or '없음'),
              "- 하락: "+(', '.join(delegated['regressed']) or '없음'),
              "- 새 근거 연결은 현재 내용 검토에 반영하되, 수집 개수만으로 부분 확보를 완료로 바꾸지 않습니다."]
    lines += ["", "## 자료 보강·대조 목록", ""]
    for q in report["questions"]:
        if q["review_status"] != "supported":
            lines.append(f"- **{q['domain']} / {q['question_id']}**: {q['follow_up']}")
    lines += ["", "## 근거는 있지만 검토한 청크가 상위 5개에 없는 질문", ""]
    for q in report["questions"]:
        if q["review_status"] == "supported" and not q["reviewed_chunk_hit_at_5"]:
            lines.append(f"- {q['question_id']}: {q['query']}")
    for q in report["questions"]:
        lines += ["", f"## {q['question_id']} · {q['domain']}", "", q["query"], "",
                  "- 필요한 근거: " + "; ".join(q["required_evidence"]),
                  "- 검토 상태: " + STATUS_LABELS[q["review_status"]],
                  "- 검토 메모: " + q["review_note"],
                  f"- 검색 진단: 출처 적중={q['reviewed_source_hit_at_5']}, 검토 청크 적중={q['reviewed_chunk_hit_at_5']}",
                  "- 후속 작업: " + q["next_step"], "", "### 검토 근거", ""]
        for e in q["evidence"]:
            lines += [f"- `{e['chunk_id']}` · {e['title']} · 원본 `{e['source_path']}` · 위치 {e['source_locator'] or e['page_number']}",
                      f"  - 원문 URL: {e['source_url'] or '미확인'}", "  - 본문 발췌: " + e["excerpt"].replace("\n", " ").rstrip(),
                      "  - 주의사항: " + str(e["source_anomaly"] or e["calculation_restricted_clause"] or " / ".join(e["warnings"]) or "별도 표시 없음")]
        if not q["evidence"]:
            lines.append("- 검토한 직접 근거 없음. 검색 결과 존재 여부와 별도로 자료 보완 필요.")
        lines += ["", "### 검색 상위 5개", ""]
        for h in q["search_results"]:
            lines.append(f"- {h['rank']}. `{h['chunk_id']}` · {h['title']} · BM25 {h['score']}")
        lines.append('- 기존 검색 청크 적중: '+str(q['legacy_chunk_hit_at_5']))
    lines += ["", "## 검증 범위의 한계", ""] + ["- " + x for x in report["limitations"]]
    lines += ["", "## 재실행 및 삭제", "", "`uv run python -m src.scripts.tests.test_domain_coverage_pmh`로 보고서만 갱신합니다.",
              "이 파일과 두 보고서는 기존 수집·정제·청크 생성 코드의 의존성이 아닙니다. 삭제해도 데이터 생성에는 영향이 없지만 질문·검토 기록과 재검증 기능은 사라집니다."]
    return "\n".join(lines) + "\n"


# 공식 민사 회수 안내 18개 청크 본문·조건·예외 수동 검토 스냅샷
REVIEWED_HASHES.update({'easylaw_consumer_small_claim_scope_0000': '199a76e4e710d2538cf9927691711a56a86e19ce4756e5e8ae3b6394247905f8', 'easylaw_consumer_small_claim_filing_0000': '95be2f0eee8f6ba3b6c6546d7ae3e4fabdc8fc832a4a949dbe2b213ef045613c', 'easylaw_consumer_small_claim_recommendation_0000': '5c87f2a29c36db9ccb8925e9b2d36c26eb1ae6034aa7629c0f202ef5218a9e44', 'easylaw_consumer_small_claim_hearing_0000': 'fa557f0094e61de63508631e4b603dd40f73828525497def0e50bdda86c61c96', 'easylaw_consumer_small_claim_judgment_0000': '85d46cca98f680c6b1eaa75e11429c93aff6e03b4dd7948df7516ec69e1e1050', 'easylaw_consumer_payment_order_definition_0000': 'daf99d47ee478dddd045e7718dddca5db9d91364d03afa6ba8e45e9951a29ca2', 'easylaw_consumer_payment_order_filing_0000': '353520e857e1ca6cdd53771ff70f567e20a4e607e6753de9ec6c9e4fa8d9dd73', 'easylaw_consumer_payment_order_service_0000': '19d57eea9a1cf2085921c0604e3d3b4b6c5dd68e64ba0a8c567fa991ffb8d596', 'easylaw_consumer_payment_order_objection_finality_0000': 'd5bbd41adaa3517a9e84881260284e0f74103dbbe03c623e88d815d02c072827', 'easylaw_payment_order_application_application_0000': 'bf2dc4426f5b68b149b2b9d5f300b27cd8843a4973b394db903417bdb7511f63', 'easylaw_payment_order_application_service_0000': '2485993c1bf6c7a2d8a9991a568775ab4f65f0f8ab7635c1195d9b9bbbcce190', 'easylaw_payment_order_application_objection_0000': '91a9da5a8d916e4e571dbde2a34578bc6b417e4b893972d04dcb4c3a7ea00dab', 'easylaw_payment_order_application_litigation_0000': 'a62cd615315c0db2566beec077e84bd8f61f82f1f4208692786566aabb926d62', 'easylaw_payment_order_application_stamp_correction_0000': '998e55c65f19d1421eb6aeec4d6e1dfe64f4176cb91c47ddd640f98b3187fff8', 'easylaw_civil_enforcement_execution_copy_0000': '35efd7594488023d1b1902c64a99c892596b810544bfe042fdbe768eaa72acc9', 'easylaw_civil_enforcement_property_disclosure_0000': '8d0ef9514349314a76d89bda4bdf20cb4d1e49c4fc9ccbc8668c33af9e53b703', 'easylaw_civil_enforcement_property_inquiry_0000': '05fa3ddbcfefdeae8d83425d37482e35e718a34364761b20a87e37efb0bb4771', 'easylaw_civil_enforcement_movables_claims_0000': '88471559c8477aec30d1ba3dad2f3d878cae37305c0c387da13fb850a5b070a2'})

def evaluate():
    report = audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps({"questions": report["question_count"], "chunks": report["chunk_count"],
                      "statuses": dict(Counter(q["review_status"] for q in report["questions"])),
                      "source_hits": sum(q["reviewed_source_hit_at_5"] is True for q in report["questions"]),
                      "chunk_hits": sum(q["reviewed_chunk_hit_at_5"] is True for q in report["questions"])}, ensure_ascii=False))
    return report


class DomainCoverageTests(unittest.TestCase):
    def test_deposit_supplement_distinguishes_current_rate_case_scope_and_bad_citations(self):
        rows={r['question_id']:r for r in self.report['questions']}
        self.assertEqual(rows['deposit_delay']['review_status'],'supported')
        self.assertIn('003984_208701_20190601_art_0_0_chunk_0000',rows['deposit_delay']['reviewed_chunk_ids'])
        self.assertEqual(rows['deposit_restore']['review_status'],'partial')
        self.assertIn('학원 건물',rows['deposit_restore']['review_note'])
        self.assertEqual(rows['deposit_auction']['review_status'],'partial')
        self.assertIn('인용 불일치',rows['deposit_auction']['review_note'])
        for identifier in ['deposit_delay','deposit_restore','deposit_auction']:
            self.assertTrue(rows[identifier]['evidence_snapshot_valid'])

    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_bank_has_five_distinct_questions_for_each_of_nine_domains(self):
        bank = self.report["questions"]
        self.assertEqual(len(bank), 45)
        self.assertEqual(len({q["question_id"] for q in bank}), 45)
        self.assertEqual(sorted(Counter(q["domain"] for q in bank).values()), [5]*9)

    def test_reviewed_evidence_has_unchanged_text_and_provenance(self):
        for q in self.report["questions"]:
            self.assertTrue(q["evidence_snapshot_valid"], q["question_id"])
            self.assertTrue(q["review_note"])
            self.assertTrue(q["required_evidence"])
            if q["review_status"] == "supported":
                self.assertTrue(q["evidence"])
            for evidence in q["evidence"]:
                self.assertTrue((ROOT / evidence["source_path"]).is_file())

    def test_search_never_changes_content_review_into_supported(self):
        bank = {q["question_id"]:q for q in question_bank(load_chunks())}
        for q in self.report["questions"]:
            self.assertEqual(q["review_status"], bank[q["question_id"]]["review_status"])
            if q["review_status"] != "supported":
                self.assertTrue(q["follow_up"])
        missing = next(q for q in self.report["questions"] if q["question_id"] == "wage_minimum")
        self.assertIsInstance(missing["reviewed_source_hit_at_5"],bool)
        self.assertIn("minimum_wage_2026_notice_body_0000",missing["reviewed_chunk_ids"])
        self.assertEqual(missing["review_status"], "partial")

    def test_parcel_unit_anomaly_is_not_treated_as_calculable(self):
        row = next(q for q in self.report["questions"] if q["question_id"] == "parcel_special_delay")
        self.assertEqual(row["review_status"], "restricted")
        self.assertTrue(any(e["source_anomaly"] and e["calculation_restricted_clause"] for e in row["evidence"]))

    def test_changed_evidence_invalidates_existing_content_review(self):
        identifier = self.report["questions"][0]["reviewed_chunk_ids"][0]
        with patch.dict(REVIEWED_HASHES, {identifier: "changed"}):
            changed = audit()
        row = changed["questions"][0]
        self.assertFalse(row["evidence_snapshot_valid"])
        self.assertEqual(row["review_status"], "needs_review")

    def test_audit_itself_does_not_write_to_existing_data(self):
        def fingerprints():
            return [(p.relative_to(ROOT).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in sorted(DATA_PATH.rglob("*")) if p.is_file()]
        before = fingerprints()
        audit()
        self.assertEqual(before, fingerprints())

    def test_saved_reports_match_current_audit(self):
        saved = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(saved, self.report)
        self.assertEqual(REPORT_MD.read_text(encoding="utf-8"), render(self.report))

    def test_pilates_comparison_supports_cancelled_class_but_keeps_assignment_conditions(self):
        rows={r['question_id']:r for r in self.report['questions']}
        self.assertEqual(rows['fitness_cancelled']['review_status'],'supported')
        self.assertEqual(rows['fitness_transfer']['review_status'],'partial')
        for key in ['fitness_cancelled','fitness_transfer']:
            self.assertTrue(rows[key]['evidence_snapshot_valid'])
            self.assertIn('000354_290239_20261002_art_52_0_chunk_0000',rows[key]['reviewed_chunk_ids'])
            self.assertIn('003572_286997_20260701_art_13_0_chunk_0000',rows[key]['reviewed_chunk_ids'])
            self.assertIn('조정일 미상',rows[key]['follow_up'])

    def test_comparison_counts_and_reviewed_appendices_support_warranty_basics(self):
        rows=self.report['questions']
        comparison=self.report['search_comparison']
        self.assertEqual(comparison['legacy_chunk_hits'],sum(r['legacy_chunk_hit_at_5'] is True for r in rows))
        self.assertEqual(comparison['improved_chunk_hits'],sum(r['reviewed_chunk_hit_at_5'] is True for r in rows))
        warranty=next(r for r in rows if r['question_id']=='consumer_warranty')
        self.assertEqual(warranty['review_status'],'supported')
        self.assertTrue({'consumer_dispute_standard_appendix_0003_0000','consumer_dispute_standard_appendix_0004_0000'}.issubset(warranty['reviewed_chunk_ids']))

    def test_redress_review_reuses_existing_procedure_and_new_guides_and_warranty_keeps_limits(self):
        rows={r['question_id']:r for r in self.report['questions']}
        mediation=rows['consumer_mediation']
        self.assertEqual(mediation['review_status'],'supported')
        self.assertTrue(mediation['evidence_snapshot_valid'])
        for identifier in ['kca_damage_redress_procedure_redress_exclusions_0000',
                           'kca_redress_requirements_eligibility_0000','kca_redress_submission_evidence_0000',
                           'kca_redress_submission_online_0000','kca_mediation_procedure_acceptance_0000']:
            self.assertIn(identifier,mediation['reviewed_chunk_ids'])
        warranty=rows['consumer_warranty']
        self.assertEqual(warranty['review_status'],'supported')
        self.assertTrue(warranty['evidence_snapshot_valid'])
        self.assertIn('003979_289451_20260911_appendix_000100E_chunk_0000',warranty['reviewed_chunk_ids'])
        self.assertIn('별표Ⅲ·Ⅳ',warranty['follow_up'])

    def test_fitness_closure_gains_civil_recovery_guides_but_current_law_comparison_remains_partial(self):
        row=next(q for q in self.report['questions'] if q['question_id']=='fitness_closed')
        self.assertEqual(row['review_status'],'partial')
        self.assertTrue(row['evidence_snapshot_valid'])
        self.assertIn('kca_damage_redress_procedure_redress_exclusions_0000',row['reviewed_chunk_ids'])
        self.assertIn('kca_fitness_closure_pilates_card_objection_0000',row['reviewed_chunk_ids'])
        self.assertIn('민사 회수',row['follow_up'])
        self.assertIn('2026-10-02',row['follow_up'])
        for identifier in ['easylaw_consumer_small_claim_scope_0000','easylaw_consumer_payment_order_objection_finality_0000','easylaw_payment_order_application_litigation_0000','easylaw_civil_enforcement_property_inquiry_0000']:
            self.assertIn(identifier,row['reviewed_chunk_ids'])
        self.assertTrue(any(e['title']=='할부거래에 관한 법률' for e in row['evidence']))
        self.assertTrue(any(e['title']=='할부거래에 관한 법률 시행령' for e in row['evidence']))

    def test_overseas_review_distinguishes_types_and_links_domestic_exceptions(self):
        row=next(q for q in self.report['questions'] if q['question_id']=='shopping_overseas')
        self.assertEqual(row['review_status'],'supported')
        self.assertTrue(row['evidence_snapshot_valid'])
        for identifier in ['kca_overseas_types_type_purchase_agent_0000','kca_overseas_returns_return_conditions_0000','kca_overseas_checklist_purchase_agent_0000']:
            self.assertIn(identifier,row['reviewed_chunk_ids'])
        self.assertIn('국내법을 일괄 적용하지 않음',row['review_note'])
        self.assertTrue(any(e['source_url']=='https://www.kca.go.kr/home/sub.do?menukey=7830' for e in row['evidence']))

    def test_wage_reporting_has_official_procedure_and_evidence_without_upgrading_other_gaps(self):
        row=next(q for q in self.report['questions'] if q['question_id']=='wage_report')
        self.assertEqual(row['review_status'],'supported')
        self.assertTrue(row['evidence_snapshot_valid'])
        self.assertIn('moel_wage_evidence_1000323154_evidence_0_0000',row['reviewed_chunk_ids'])
        self.assertIn('moel_wage_procedure_procedure_4_0000',row['reviewed_chunk_ids'])

    def test_collected_new_sources_do_not_hide_remaining_scope_limits(self):
        rows={r['question_id']:r for r in self.report['questions']}
        self.assertEqual(rows['shopping_custom']['review_status'],'supported')
        self.assertEqual(rows['fitness_cancelled']['review_status'],'supported')
        self.assertTrue(rows['fitness_cancelled']['evidence_snapshot_valid'])
        self.assertIn('조정일 미상',rows['fitness_cancelled']['follow_up'])
        for identifier in ['wage_minimum','deposit_small','fitness_transfer']:
            self.assertEqual(rows[identifier]['review_status'],'partial')
            self.assertTrue(rows[identifier]['evidence_snapshot_valid'])
            self.assertTrue(rows[identifier]['evidence'])
        self.assertTrue(any('산입' in e['excerpt'] for e in rows['wage_minimum']['evidence']))
        self.assertTrue(any('2024년부터는 100분의 0' in e['excerpt'] for e in rows['wage_minimum']['evidence']))

    def test_existing_law_connections_are_complete_and_product_scope_stays_partial(self):
        rows={r['question_id']:r for r in self.report['questions']}
        self.assertEqual(rows['used_minor']['review_status'],'supported')
        self.assertIn('001706_284415_20260317_art_141_0_chunk_0000',rows['used_minor']['reviewed_chunk_ids'])
        self.assertIn('001706_284415_20260317_art_140_0_chunk_0000',rows['used_minor']['reviewed_chunk_ids'])
        self.assertEqual(rows['dismissal_small']['review_status'],'supported')
        self.assertTrue({'003058_270551_20251023_art_7_2_chunk_0000','003058_270551_20251023_art_7_2_chunk_0001'}.issubset(rows['dismissal_small']['reviewed_chunk_ids']))
        self.assertEqual(rows['consumer_repeat']['review_status'],'partial')
        fixed=self.report['preprocessing_baseline']['questions']
        self.assertNotIn('001706_284415_20260317_art_141_0_chunk_0000',next(q for q in fixed if q['question_id']=='used_minor')['reviewed_chunk_ids'])

    def test_refund_and_criteria_refinement_preserves_frozen_evidence(self):
        rows={r['question_id']:r for r in self.report['questions']}
        for identifier in ['fitness_gym','consumer_criteria']:
            self.assertTrue(rows[identifier]['reviewed_chunk_hit_at_5'],identifier)
            self.assertTrue(rows[identifier]['evidence_snapshot_valid'])
        comparison=self.report['search_refinement_comparison']
        self.assertEqual(comparison['after'],sum(r['after'] is True for r in comparison['questions']))
        self.assertEqual(comparison['regressed'],[r['question_id'] for r in comparison['questions'] if r['before'] is True and r['after'] is False])
        self.assertLessEqual(self.report['search_refinement_baseline']['chunk_count'],self.report['chunk_count'])


# 2026-10-11 보증금 근거의 원문·본문 검토 스냅샷. 경고·심급·역사적 이율은 별도 보존.
REVIEWED_HASHES.update({'001215_286417_20260602_art_3_0_chunk_0000': '7c519028ab6d2022c554e113eedc22ee10218138e5d644419ea1294966302ff4', '001706_284415_20260317_art_374_0_chunk_0000': '9399b06cda0daf5aefb15fcadcdd335bed0c4bd5031269f603e619f3295fb859', '001706_284415_20260317_art_379_0_chunk_0000': 'f15275169244e7966b47160454cd8e2938d1f8b8f1736edf104cfb83182aaa4a', '001706_284415_20260317_art_387_0_chunk_0000': '89abf17a3c9554b5e93366fdca178585986fc1270abac20d94f5ff426f61a780', '001706_284415_20260317_art_390_0_chunk_0000': '753be47a8eba48a68d39404c745df08d1eab6a195efe2d865eddeca78efa8c84', '001706_284415_20260317_art_393_0_chunk_0000': 'abaf057ca2872ed1387982c17c118ea7f2eb5eef9e3e375ed4cdc387a5449943', '001706_284415_20260317_art_397_0_chunk_0000': '2c92dae5b28fc5a531312b001d380603eb3a212eeed1f8feb4d8516202afafb0', '001706_284415_20260317_art_460_0_chunk_0000': '8173e18ddc7cc0be0b6774e79878d9599f848103ae05359c5b7006fac7367dbc', '001706_284415_20260317_art_536_0_chunk_0000': 'de77a32c9d72536afc8c79dbe4e62c823217dd6b8949ecd67d8a9bf883539ec1', '001706_284415_20260317_art_615_0_chunk_0000': '60799ccbb33a43c65082010e87b2edfad4b07d3f40ada5bd1a55a1593eb0373e', '001706_284415_20260317_art_634_0_chunk_0000': '4a564c0b1d34fa87349f3197d8aad11dff4cef43917be7c72137d9aba894f7d3', '001706_284415_20260317_art_654_0_chunk_0000': '7e0a0728ce060d76fc875006002c564661e525bb18649e3502382da1758f33e0', '003984_208701_20190601_art_0_0_chunk_0000': '3cce9bedc50949209ecb473c8dc4f7fe26fbcda12530e4a7d7c7d0429655ccb3', '009290_268837_20260201_art_84_0_chunk_0000': '605df45ade0c1180783dd740e54db9fed7125004d5694c86d1963f19947d878a', '009290_268837_20260201_art_88_0_chunk_0000': '01010cb188b083bb2a5255f58e1e92ea087e7ae3e48d08016d26f85f19c89585', '009290_268837_20260201_art_145_0_chunk_0000': '8da4812354358e1102057180e583ac2febc1d40e8e2cfb1266fad5d025d19ab6', '009290_268837_20260201_art_148_0_chunk_0000': '0c005b7123d416fcc3fbf157d4aa2789389d69b3b9373e719cb1c58138a36049', '009325_272631_20250712_art_48_0_chunk_0000': '885c00712f49d1144e5e77bb556b7332de8b954d8393169ebb8260bf03d638e0', '147076_0000': 'c6aa054db21322d9fb4dd06ba4904f7ba10efb5f340f1d8e5f690084f113100a', '147076_0001': '23e4b00c5c2c8b0cfc0838017cca29e7c4eb3ad2f48c63ba574231c7978b7365', '147076_0002': '6685bdcf2bd7348ea51929111c80795674275786a2a78bfe406029ee117aeb7c', '147076_0003': '1196acfb08eadaaeb1b8551d8941685e5581403e37f5ceb3dbb6e0d4dced1fc3', '147076_0004': '4e1237c3faf9a549b1b256f4b716784bb7c6d99699be4f378b918a81d64f4ae3', '147076_0005': '4495a8472a05e88cb623e4ddbcb380ec24089ffbe9a4148de821d71f3235094e', '147076_0006': '4f00ae325badb8401c5eddb08081c7273e168491543588141dcfd36752b0d373', '147076_0007': 'c454163651838cdbf8748423e3f7695ffd9064e6f6ac6e77ee58ec2abcec2201', '147076_0008': 'c2c7f9f25058b05f492ff946a33f42ebc30765fc238f09cda1f4f0c7bb8d9a9a', '147076_0009': 'eae943856f5e82d74c0d659a6c0e9fcf61e3c13429bdfa1412fd4b10a7bfb219', 'easylaw_housing_termination_return_0000': '18d1b493fdb32e6ec905d1a79544769ab409b0796ab1d3a4d67dc4afb921af38', 'easylaw_housing_termination_registration_move_0000': '2624a48a225c69dbbd07707f8693d8b0379ab996e2a18f16d6d438bab46c680f', 'easylaw_housing_restoration_care_0000': '321b4ec24a2efc16397d8c67c6c24bbcb06be8cca480b3ca91bae3c76ed53f07', 'easylaw_housing_restoration_restoration_0000': '39b03e234f7fe4f8157fe20cc9fd5d9c893912fea90b293558c8420a46901750', 'easylaw_housing_distribution_definition_0000': '735129969852da5705986c76c20e5432785ee0a5b9640302ec7b796365bf9aaf', 'easylaw_housing_distribution_creditors_0000': 'cb7fd205aa8908b1cf1c8a2eb8fbf486b3df6f08ce7447d1fd58c7909b7d32bb', 'easylaw_housing_distribution_deadline_exceptions_0000': '2f3fc7b12908104f609875729ab56878641a56e5d306b199d00fc5899ab1a69f', 'easylaw_housing_distribution_application_0000': '97d4f45db67876621c52eb0ab57145da69cf798419951c2aff1f9d08adac8a0e', 'easylaw_housing_distribution_tenant_share_exceptions_0000': 'e07a3f3b7aa6341583df687b9f88117ee255632e7278407efa693d111811256a', 'easylaw_housing_distribution_maintain_priority_0000': '6933204e10eaa26a3a2c1ba3815109e5310c6987e2ad5e06589b1d57eebc687b', 'easylaw_housing_distribution_paid_0000': '70d271f50f4f3b3e0d449137920f16ca2517560ece78274214d49856b6b8cd46', 'easylaw_housing_distribution_unpaid_0000': '2a17aebebb77957f689b1f45146736d178a3720ca85915c4ccac862570d47f5a', 'easylaw_housing_distribution_delivery_0000': 'f64b1ce3075cc727b2f0390df72374e40aa631effb27a64e4b60ddbf33e093d2'})

REVIEWED_HASHES.update({'001248_276291_20260102_art_3_5_chunk_0000': 'c4d86ff7ea8d083a7d3f4de72c48c6b3efe8bc84ed2f686441c37f45cb81fc39'})

# 2026-10-11 소비자 절차 및 일반적 품질보증 기준의 원문 내용 검토 스냅샷.
REVIEWED_HASHES.update({'001589_283837_20260911_art_16_0_chunk_0000': '1b56f92edbb823a377f5ef785fdd67fed13886be29d80deac94b50e7f6b916a2', '001589_283837_20260911_art_55_0_chunk_0000': '435121988478120e7730ee7356016d75c1454afe453b2bd634d12a0a981eaa44', '001589_283837_20260911_art_57_0_chunk_0000': '6428feb24a65bfb94d0b82a6d7c24dd3c677f39dfdf2bedab4432f99eddc42c4', '001589_283837_20260911_art_58_0_chunk_0000': '67f02bd55d4a4721c55de49bafad6187a098a7765732a9d0760a2630907e75bc', '001589_283837_20260911_art_65_0_chunk_0000': '9b053079ef2c72b406bbd271e35e1307644d238800190df3f7ade7a9bfb3fca5', '001589_283837_20260911_art_66_0_chunk_0000': '2f2c0082749ea1bdbd7c7f6089e9f290e820f7d3cb90d158ea1b80a7e73a155f', '001589_283837_20260911_art_67_0_chunk_0000': '020c16cf4ebc260455b65a8444d4d3ef01326b98fe29bb442faf0841e3e34f9f', '003979_289451_20260911_appendix_000100E_chunk_0000': '15ba4474849198d527dd0cb5ab2b938655546340c8550d1f1147843e98c82a2a', '003979_289451_20260911_art_44_0_chunk_0000': 'dfa470518f49ab6eed139b417c1def696cec7ac2ab2be49fe4eb746e8b61e896', '003979_289451_20260911_art_8_0_chunk_0000': '936df6cd25cf11f9d203706226f41bdb53a696b5f81d22deb3df47b6abe121c6', 'kca_mediation_procedure_acceptance_0000': 'c7ab7b7662419f19f4996438b37961d17451e9ddd45463aa215b2ae8348b4451', 'kca_mediation_procedure_completion_0000': 'e6a3b188db5c3fab1e086ee6b6f9173f5198967b9305990ddf2bb18775823c0c', 'kca_mediation_procedure_definition_0000': 'd1c87b0aa1b8679b875ca02aaeb62d462b9a85dba3d00086a03d64701080657c', 'kca_mediation_procedure_request_0000': '5512f50db36e0250f6fc053cf574eb49f9821d16e5771dc300721daef1c878b3', 'kca_redress_requirements_eligibility_0000': 'b2f57f0fa2269f8a037b88c199024cea1904244b674759d0d501e3a6c771a62b', 'kca_redress_requirements_specialized_forms_0000': '2bbb13eaa2ff25ba333882e3ea2e138bfcd7736996af493723540e8863970a22', 'kca_redress_submission_evidence_0000': 'c44e6879a26ee0d3c5760abe6b0330f5df21bfeb733b6dce9e5cc33a4c525b41', 'kca_redress_submission_online_0000': '993c74b45d4940f2f5480c7f23e14dd949adb853fb7af2ecdcabe43493f60ea7', 'kca_redress_submission_postal_0000': '1409fdd23f27a9a399bfc9cfc47f7b7b920d77b455ead0535104e883f7af92e9', 'kca_redress_submission_postal_address_0000': 'cf2609d7a78ca8e9afd0d322b17a1c3ff041e6a5ba6f8f19383822dd932388c3', 'kca_redress_submission_visit_0000': '88e82ec6fe3998a397a13d246966aed185c3fae29d9a352cbae9f9566c881899'})


REVIEWED_HASHES.update({'000354_290239_20261002_art_52_0_chunk_0000':'52660a7113f580f7da90995d9e82b9b4366690c948808afcfabbf3be2fbaaf6d', '003572_286997_20260701_art_13_0_chunk_0000':'6bdbffb43e02c3f9a9d04ebfea8fea84ced76168e886889a9adb9b11bdbe316c'})

if __name__ == "__main__":
    evaluate()
