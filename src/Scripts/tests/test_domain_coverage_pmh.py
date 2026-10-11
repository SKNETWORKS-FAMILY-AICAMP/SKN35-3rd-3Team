"""9대 분야 45문항의 자료 근거 검토와 BM25 검색 진단.

uv run python -m src.Scripts.tests.test_domain_coverage_pmh
직접 실행은 보고서만 작성한다. unittest 실행은 데이터에 쓰지 않는다.
검색/키워드 적중으로 충분성을 판정하지 않으며 검토한 근거와 메모를 사용한다.
"""
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.Scripts.tests.test_consumer_coverage_pmh import load_chunks, SEARCH_OPTIONS, SEARCH_POLICY
from src.Scripts.tests.test_precedent_retrieval_pmh import BM25

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data/processed/retrieval_evaluation"
REPORT_JSON = OUTPUT / "domain_coverage_report_pmh.json"
REPORT_MD = OUTPUT / "domain_coverage_report_pmh.md"
STATUS_LABELS = {"supported": "기본 근거 확보", "partial": "근거 일부·자료 보완",
                 "restricted": "원문 확인 전 적용 보류", "needs_review": "추가 내용 검토"}


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_fingerprint():
    files = sorted(p for p in (ROOT / "data/chunk").rglob("*") if p.is_file())
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
        "개별 제작 여부;고지와 동의;시행령 예외","partial","KCA 한복 사례에 개별 제작·서면 동의 판단 있음. 전자상거래법 시행령 원문은 수집 목록에 없음.",["kca_case_1003164054_case_reasoning_0000"]+law("전자상거래 등에서의 소비자보호에 관한 법률","제17조"),"전자상거래법 시행령의 주문제작 예외 조항 원문 보강")
    add(shopping,"shopping_defect","온라인 주문 내용과 다른 제품을 받았을 때 반품 기간과 배송비는 어떻게 되나요?",
        "계약과 다른 이행의 기간;배송비 부담","supported","제17조 제3항과 제18조 제10항을 함께 확인.",law("전자상거래 등에서의 소비자보호에 관한 법률","제17조")+law("전자상거래 등에서의 소비자보호에 관한 법률","제18조"))
    add(shopping,"shopping_overseas","해외직구 반품비가 너무 비싸면 국내 쇼핑몰과 같은 기준으로 환불받나요?",
        "직구·구매대행 구분;비용 고지와 입증;계약 상대방","partial","해외 구매대행 안내와 사례는 있으나 모든 해외 판매자·직구 유형의 적용 근거는 확보 검토가 부족.",["guide_383b62bce56bf1ce_p0032_table_0000","kca_case_1003641715_case_reasoning_0000"],"거래 유형별 해외직구 공식 안내와 적용 제한 보강")
    fitness = "헬스장·필라테스 환불"
    add(fitness,"fitness_gym","헬스장 이용권을 개인 사정으로 중도해지하면 환불은 어떻게 계산하나요?",
        "개시 전후;기간·횟수 공제;실제 이용료와 위약금","supported","헬스장 약관의 표 셀과 수식·이용료·위약금 정의가 한 청크에 보존됨. 실제 계약 적용과 금액 입력은 별도 검증.",["ftc_terms_46118_file_5b139f26_terms_article_7_0000"])
    add(fitness,"fitness_before","필라테스 수업 시작 전에 개인 사정으로 해지하면 어떤 금액이 공제되나요?",
        "등록기간 개시 전;회원 귀책;이용료·위약금 정의","supported","필라테스 약관의 개시 전 회원 귀책 환급 표와 정의를 확인.",["ftc_terms_47762_file_05d51a47_terms_article_8_0000"])
    add(fitness,"fitness_cancelled","필라테스 수업이 폐강됐는데 정상가로 공제하고 환불한다면 어떻게 하나요?",
        "사업자 귀책;할인가·정상가 공제;계속거래 적용","partial","폐강 사례에 정상가 공제 조항과 계속거래 근거를 인용. 방문판매법·산정기준 원문 미수집으로 적용 범위 검증은 불완전.",["kca_case_1003739114_case_reasoning_0000","kca_case_1003739114_case_reasoning_0001"],"방문판매법과 계속거래 환급 산정기준 원문 보강")
    add(fitness,"fitness_transfer","양도받은 필라테스 이용권을 해지하면 환급받을 수 있나요?",
        "양도 계약;회원 사정;환급 산정의 사건별 조건","partial","양도 이용권 사례의 판단과 결론 있음. 해당 사례 금액을 다른 계약에 적용할 수 없고 원법 적용 범위 보강 필요.",["kca_case_1003640779_case_reasoning_0000","kca_case_1003640779_case_decision_0000"],"방문판매법·산정기준 및 양도 계약 조건을 함께 검토")
    add(fitness,"fitness_closed","헬스장이 폐업하면 남은 이용료를 돌려받는 기준과 회수 방법은 무엇인가요?",
        "사업자 귀책 환급;폐업 후 실제 회수 절차","partial","폐업을 해지 사유로 정한 약관과 환급 표는 있음. 폐업·연락두절 후 회수 절차 안내는 별도 검토 필요.",["ftc_terms_46118_file_5b139f26_terms_article_7_0000"],"폐업 피해구제·카드 결제 관련 공식 절차 안내 보강")
    wage = "아르바이트 임금체불"
    add(wage,"wage_after_exit","아르바이트를 그만뒀는데 마지막 급여는 언제까지 지급해야 하나요?",
        "퇴직 후 지급 기한;합의 연장 예외","supported","근로기준법 제36조 본문과 예외 확보.",law("근로기준법","제36조"))
    add(wage,"wage_minimum","올해 아르바이트 시급이 최저임금보다 낮은지 어떻게 확인하나요?",
        "해당 연도 적용 금액;산입범위;적용 예외","partial","근로기준법만으로 해당 연도 금액을 확정할 수 없음. 최저임금법·시행령·연도별 고시 원문은 8종 법령 목록에 없음.",[],"최저임금법·시행령과 해당 연도 고용노동부 고시 수집")
    add(wage,"wage_weekly","주 15시간 안팎으로 일하는 아르바이트의 주휴수당 조건은 무엇인가요?",
        "유급휴일;개근;소정근로시간 기준","supported","법 제18조·제55조와 시행령 제30조를 함께 확인. 실제 스케줄·근로자성 판단은 별도.",law("근로기준법","제18조")+law("근로기준법","제55조")+law("근로기준법 시행령","제30조"))
    add(wage,"wage_report","밀린 아르바이트 급여는 어디에 진정하거나 고소하나요?",
        "관할 노동관서;진정과 고소 차이;진행 경로","partial","안내서에 관할 노동관서와 진정·고소 구분 있음. 제출 자료·접수 화면·현행 처리 절차까지 검토한 것은 아님.",["guide_cad7b1de1e4dc64c_p0078_body_0000"],"고용노동부 임금체불 신고 절차·준비서류 안내 보강")
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
        "지역별 범위·금액;담보권과 개정 시점;경과 규정","partial","제8조와 안내서 지역 기준 있음. 시행령 원문·개정 이력·경과조치까지 수집 대조하지 않아 시점별 금액 확정 불가.",law("주택임대차보호법","제8조")+["guide_23ffe4c8fd5715cb_p0067_body_0000"],"주택임대차보호법 시행령과 필요한 과거 버전·부칙 보강")
    add(deposit,"deposit_delay","보증금 반환이 늦어지면 언제부터 어떤 이율로 지연손해금을 청구하나요?",
        "동시이행;이행 제공;소송 전후 이율","partial","동시이행 설명과 민법 기본 이율 조문은 있음. 소송촉진 관련 법 원문·사건별 이행 제공 검토가 미완료.",["guide_23ffe4c8fd5715cb_p0047_body_0000"]+law("민법","제379조"),"소송촉진 관련 법·시행령과 반환 지연 직접 사례 검토")
    add(deposit,"deposit_restore","집주인이 원상복구비를 보증금에서 공제하려는데 전액 부담해야 하나요?",
        "원상회복 범위;특약과 당시 상태;실제 비용 입증","partial","안내서가 원상회복의 개별 판단 요소를 설명. 통상손모·비용 공제의 구체 사례와 증거 기준은 추가 검토 필요.",["guide_23ffe4c8fd5715cb_p0045_body_0001","guide_23ffe4c8fd5715cb_p0046_body_0000"],"통상손모·원상복구비 공제의 직접 사례와 공식 안내 보강")
    add(deposit,"deposit_auction","집이 경매에 넘어갔을 때 보증금을 받으려면 배당요구를 해야 하나요?",
        "배당요구 종기;신청 방법;요구 없이 배당받는 예외","partial","안내서에 종기와 예외 설명 있음. 민사집행법·규칙 원문 및 개별 권리 순위 검토는 없음.",["guide_23ffe4c8fd5715cb_p0076_body_0001","guide_23ffe4c8fd5715cb_p0068_body_0000"],"민사집행법·규칙 관련 조문과 배당 신청 공식 안내 보강")
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
        "접수 창구;합의 권고와 조정;필요 자료·절차","partial","안내서에 소비자24·한국소비자원과 합의·조정 설명 있음. 소비자기본법 원문·신청서류·현행 절차 전체는 검토 부족.",["guide_383b62bce56bf1ce_p0035_body_0000"],"소비자기본법과 KCA 피해구제·조정 신청 공식 안내 보강")
    add(consumer,"consumer_repeat","수리를 반복해도 같은 하자가 생기는 제품은 모두 환불받을 수 있나요?",
        "품목별 기준;수리 횟수·보증기간;교환·환급 구분","partial","기존 반지 사례에 별표Ⅱ 전자제품·사무용기기 및 귀금속·보석 표를 추가 연결. 전자제품의 반복 하자·보증기간 조건과 귀금속의 분쟁 유형을 구분해 보존. 모든 품목의 기준을 확보한 것은 아니며 XML 표의 문자·병합 관계 대조 필요.",["kca_case_1003604873_case_reasoning_0000","consumer_dispute_standard_product_electronics_office_0000","consumer_dispute_standard_product_jewelry_0000"],"서비스 대상 품목 확정 후 나머지 품목의 원본 정제 확대 및 공식 별표 대조")
    add(consumer,"consumer_warranty","영수증이 없으면 제품 품질보증기간과 구매일을 어떻게 판단하나요?",
        "품목별 보증기간;구매일 증명;영수증 없는 경우","partial","제3조와 추가 정제한 별표Ⅲ의 품목별 품질보증·부품보유기간, 별표Ⅳ의 내용연수 기준 확보. 영수증 없는 구매일 판단 근거는 이 표에 없으며 문자 이상·병합 관계의 공식 파일 대조도 필요. 질문 전체는 여전히 부분 확보.",["consumer_dispute_standard_article_2_0000","consumer_dispute_standard_appendix_0003_0000","consumer_dispute_standard_appendix_0004_0000"],"일반적 해결기준·구매일 증빙 안내 보강 및 별표 문자·병합 관계 대조")
    add(consumer,"consumer_criteria","소비자분쟁해결기준은 바로 강제되는 법적 배상 기준인가요?",
        "합의·권고 기준 성격;별도 약정과 적용","supported","고시 목적 조문과 택배 안내서의 별도 약정·합의 권고 설명 확보. 실제 집행력·소송 효력 전반은 별도 검토.",["consumer_dispute_standard_article_0_0000","guide_94e9d4bd30eadf0b_p0005_body_0000"])
    return bank


# 최초 내용 검토 시 선정된 근거 본문 해시. 이후 변경 시 자동으로 재검토 표시.
REVIEWED_HASHES = {
    "001706_284415_20260317_art_140_0_chunk_0000": "e625b80f84ad1e2713a7205637a40c4969480cd81d4afa2b658f4875ce25f351",
    "001706_284415_20260317_art_141_0_chunk_0000": "fde13942b75b9a3d69476764322eac59b45f1687306fc19ee91e9625a5abacc6",
    "003058_270551_20251023_art_7_2_chunk_0000": "5ba43af3594ff65d53f1b262b7432e7950da66f3dca00b5713e6f7bafbedd0ab",
    "003058_270551_20251023_art_7_2_chunk_0001": "4c69a0fa29f1bc2b22ddc359d335adcd832cc04904668894f2b0657ad1926ea9",
    "consumer_dispute_standard_product_jewelry_0000": "46fde5d4a269104195acb57a13d345d5be5a34889577ed8c031603c162ac667c",
    "consumer_dispute_standard_product_electronics_office_0000": "21f18c3101e528e7be10912b0224df8d8cf6fb1ed15bb7d30488cffa8f1ea906",
    "consumer_dispute_standard_appendix_0003_0000": "34815ef7afae0e6874c166295c92fe4c34f3a692b75dc84828fc1bf75679067b",
    "consumer_dispute_standard_appendix_0004_0000": "ac434a2eab42e0eef204f3e4f6c31bab2618d7aad9ba2186eb7de2cc8ecb0279",
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
    return {"evaluation_type": "reviewed_evidence_and_offline_bm25_domain_audit",
            "reviewer": "Codex의 수집 자료 내용 검토. 전문가 법률 검증 아님",
            "chunk_count": len(chunks), "question_count": len(records),
            "input_fingerprint": source_fingerprint(), "search_policy":SEARCH_POLICY,"summary": summaries, "questions": records,
            "preprocessing_baseline":baseline,
            "preprocessing_comparison":{'basis':'수정 전 선정 청크 목록을 고정. 근거 목록 확대에 따른 자동 적중 증가와 분리.',
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
                            "BM25는 점수용 본문 구간과 셀 문자·용어 확장·주장 감점을 사용하며 문서 버전당 최대 2개 청크로 상위 5개 반환. 기존 문서당 1개 방식과 동일 자료·선정 근거로 비교.",
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
    lines += ["", "## 재실행 및 삭제", "", "`uv run python -m src.Scripts.tests.test_domain_coverage_pmh`로 보고서만 갱신합니다.",
              "이 파일과 두 보고서는 기존 수집·정제·청크 생성 코드의 의존성이 아닙니다. 삭제해도 데이터 생성에는 영향이 없지만 질문·검토 기록과 재검증 기능은 사라집니다."]
    return "\n".join(lines) + "\n"


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
        self.assertIsNone(missing["reviewed_source_hit_at_5"])
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
                    for p in sorted((ROOT / "data").rglob("*")) if p.is_file()]
        before = fingerprints()
        audit()
        self.assertEqual(before, fingerprints())

    def test_saved_reports_match_current_audit(self):
        saved = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(saved, self.report)
        self.assertEqual(REPORT_MD.read_text(encoding="utf-8"), render(self.report))

    def test_comparison_counts_and_new_appendix_review_do_not_upgrade_sufficiency(self):
        rows=self.report['questions']
        comparison=self.report['search_comparison']
        self.assertEqual(comparison['legacy_chunk_hits'],sum(r['legacy_chunk_hit_at_5'] is True for r in rows))
        self.assertEqual(comparison['improved_chunk_hits'],sum(r['reviewed_chunk_hit_at_5'] is True for r in rows))
        warranty=next(r for r in rows if r['question_id']=='consumer_warranty')
        self.assertEqual(warranty['review_status'],'partial')
        self.assertTrue({'consumer_dispute_standard_appendix_0003_0000','consumer_dispute_standard_appendix_0004_0000'}.issubset(warranty['reviewed_chunk_ids']))

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


if __name__ == "__main__":
    evaluate()
