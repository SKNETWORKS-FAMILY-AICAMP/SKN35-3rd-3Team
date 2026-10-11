"""무료 BGE 모델의 CPU 속도를 측정합니다. 검색 품질 평가는 아닙니다.

uv run --extra rerank python -B -m src.Scripts.rerank_benchmark
모델 최초 다운로드 후 캐시를 재사용합니다. 유료 API/원격 Qdrant는 호출하지 않습니다.
"""
import ctypes
import json
import os
from pathlib import Path
from time import perf_counter

from src.rerank import CrossEncoderScorer, DEFAULT_RERANK_MODEL


def memory_mib():
    """Windows 프로세스의 현재 작업 메모리와 최대 작업 메모리만 조회합니다."""
    if os.name != "nt":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        return None
    return {"working_set_mib":round(counter.WorkingSetSize / 2**20, 1),
            "peak_working_set_mib":round(counter.PeakWorkingSetSize / 2**20, 1)}


def main():
    import torch
    # 모든 CPU 스레드를 점유하지 않도록 벤치마크에서는 4개로 제한합니다.
    torch.set_num_threads(4)
    query = "전세 계약이 끝났는데 집주인이 보증금을 돌려주지 않아요."
    # 임의 작성된 테스트 문장입니다. 법률 정답이나 공식 출처로 사용하면 안 됩니다.
    documents = [
        "임대차 종료 후 보증금 반환에 관한 안내 자료입니다.",
        "택배가 분실되었을 때 확인할 배송 기록에 관한 자료입니다.",
        "근로자의 임금 지급과 체불 상담에 관한 자료입니다.",
        "임차인과 임대인의 보증금 반환 분쟁 사례입니다.",
        "온라인 쇼핑 주문 취소와 환불에 관한 안내 자료입니다.",
        "주택 임대차 계약 기간과 계약 종료에 관한 설명입니다.",
        "중고 물품 거래와 판매자 연락 기록에 관한 자료입니다.",
        "보증금을 받지 못한 임차인의 상황을 다루는 상담 자료입니다.",
        "헬스장 이용 계약 해지와 환불에 관한 자료입니다.",
        "근로 계약 종료와 해고에 관한 자료입니다.",
    ]
    start = perf_counter()
    scorer = CrossEncoderScorer.from_pretrained(batch_size=1, max_length=512)
    load_seconds = perf_counter() - start
    scorer.score_pairs(query, documents[:1])  # 최초 실행 준비 시간을 본 측정과 분리
    runs = []
    for _ in range(2):
        start = perf_counter()
        scores = list(scorer.score_pairs(query, documents))
        runs.append(perf_counter() - start)
    # 짧은 문장만으로 실제 청크 처리 속도를 낙관하지 않도록 긴 입력도 측정합니다.
    start = perf_counter()
    scorer.score_pairs(query, [text * 20 for text in documents])
    long_seconds = perf_counter() - start
    report = {
        "model":DEFAULT_RERANK_MODEL, "device":"cpu", "threads":4,
        "candidate_count":len(documents), "batch_size":1, "max_length":512,
        "load_seconds":round(load_seconds,3),
        "short_input_runs_seconds":[round(value,3) for value in runs],
        "long_input_seconds":round(long_seconds,3),
        "process_memory":memory_mib(),
        "ranked_sample_indices":sorted(range(len(scores)),key=lambda index:-scores[index]),
        "evaluation_type":"CPU benchmark with synthetic text; not legal retrieval quality",
    }
    output = Path(__file__).resolve().parents[2] / "docs" / "rerank-cpu-benchmark.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
