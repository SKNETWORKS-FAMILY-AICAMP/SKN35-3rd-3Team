# rerank 구현

def rerank(state):
    # state에 따라서 rerank를 수행하는 기능을 구현
    return {}


# 기존 그래프용 자리 함수는 유지합니다. 실제 재정렬 기능은 아래에 추가합니다.
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.vectorstore import VectorSearchHit, VectorSearchRequest


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "rerank"


class PairScorer(Protocol):
    """질문 하나와 문서 목록을 받아 문서별 관련성 점수를 반환하는 규칙."""

    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


@dataclass
class CrossEncoderScorer:
    """로컬 Cross-Encoder를 연결합니다. 유료 API는 사용하지 않습니다.

    Dense는 질문/문서 벡터를 따로 비교하지만, Cross-Encoder는 질문과 문서를
    한 쌍으로 읽습니다. 새로운 문서를 검색하는 것이 아니라 후보 순위를 바꿉니다.
    """

    model: object
    batch_size: int = 1

    @classmethod
    def from_pretrained(
        cls, model_name: str = DEFAULT_RERANK_MODEL, *,
        cache_folder: str | Path = DEFAULT_MODEL_CACHE,
        max_length: int = 512,
        batch_size: int = 1,
        cpu_threads: int = 4,
    ):
        """모델은 처음 한 번 다운로드하고 CPU에서 실행합니다.

        max_length는 질문+문서의 토큰 길이 제한입니다. 긴 청크의 뒷부분은
        잘릴 수 있으므로 실제 데이터로 길이/품질을 비교해야 합니다.
        batch_size=1은 한꺼번에 처리하는 문서를 줄여 RAM 부담을 낮춥니다.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name은 비어 있지 않은 문자열이어야 합니다.")
        for name, value in (("max_length", max_length), ("batch_size", batch_size), ("cpu_threads", cpu_threads)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}은 양의 정수여야 합니다.")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as error:
            raise RuntimeError("Rerank 패키지가 필요합니다. uv add --optional rerank sentence-transformers를 실행하세요.") from error
        # PyTorch 프로세스 전체의 CPU 연산 스레드 수를 제한합니다.
        # 스레드 제한은 CPU 사용률 상한/응답시간을 보장하는 것은 아닙니다.
        torch.set_num_threads(cpu_threads)
        model = CrossEncoder(
            model_name, device="cpu", max_length=max_length,
            cache_folder=str(cache_folder), trust_remote_code=False,
        )
        # 다중 클래스 분류 모델을 점수 모델로 잘못 사용하지 않도록 검사합니다.
        if model.num_labels != 1:
            raise ValueError("문서별 단일 점수를 반환하는 Reranker 모델이 필요합니다.")
        return cls(model, batch_size)

    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return ()
        pairs = [(query, text) for text in documents]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        # numpy 배열은 Python 숫자 목록으로 바꿔 공용 처리 코드에 전달합니다.
        return [float(score) for score in scores]


@dataclass
class RerankingRetriever:
    """기존 검색기를 감싸 후보 검색 → 재정렬 → 최종 선택을 수행합니다.

    searcher에는 Hybrid 또는 MultiQueryRetriever를 넣습니다.
    scorer는 실제 모델 또는 테스트 점수 함수를 주입할 수 있습니다.
    """

    searcher: object
    scorer: PairScorer
    candidate_k: int = 10

    def search_documents(
        self, query: str, *, top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[VectorSearchHit]:
        """원본 질문을 기준으로 후보를 재정렬합니다.

        score_threshold는 Reranker 점수 하한입니다. RRF/벡터 점수와 달라
        같은 기준값을 재사용하면 안 됩니다. 점수는 정답 확률/Precision도 아닙니다.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query는 비어 있지 않은 문자열이어야 합니다.")
        if filters is not None and not isinstance(filters, Mapping):
            raise TypeError("filters는 매핑 또는 None이어야 합니다.")
        for key, value in (filters or {}).items():
            if not isinstance(key, str) or not key.strip() or not isinstance(value, (str, int, bool)):
                raise ValueError("필터는 문자열 키와 문자열·정수·불리언 값이어야 합니다.")
        # 호출 전에 옵션을 검사합니다. 기존 검색기와 동일한 1~100개 규칙입니다.
        VectorSearchRequest.create([0.0], top_k=top_k, score_threshold=score_threshold)
        VectorSearchRequest.create([0.0], top_k=self.candidate_k)
        query = query.strip()
        candidates = self.searcher.search_documents(
            query, top_k=max(self.candidate_k, top_k), filters=filters,
        )
        if not candidates:
            return ()  # 빈 결과에 모델을 호출하지 않습니다.
        scores = list(self.scorer.score_pairs(query, [hit.text for hit in candidates]))
        if len(scores) != len(candidates):
            raise ValueError("Reranker 점수 개수가 후보 문서 개수와 다릅니다.")
        results = []
        for rank, (hit, score) in enumerate(zip(candidates, scores), start=1):
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Reranker는 문서별 유한한 숫자 점수를 반환해야 합니다.")
            if score_threshold is not None and score < score_threshold:
                continue
            # 이전 검색 점수/순위와 RRF 증거를 보존하여 적용 전후를 비교합니다.
            metadata = dict(hit.metadata)
            metadata["before_rerank"] = {
                "rank": rank, "score": float(hit.score),
                "method": metadata.get("retrieval_method", "unknown"),
            }
            metadata["retrieval_method"] = "rerank"
            results.append((rank, VectorSearchHit(hit.point_id, float(score), hit.text, metadata)))
        # 동점은 기존 검색 순위를 유지합니다. 본문과 출처는 변경하지 않습니다.
        results.sort(key=lambda item: (-item[1].score, item[0]))
        return tuple(hit for _, hit in results[:top_k])
