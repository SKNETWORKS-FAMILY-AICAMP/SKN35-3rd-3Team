"""Qdrant Client를 Vector Store 인터페이스에 연결합니다.

MCP SDK나 LangGraph를 import하지 않습니다. Qdrant Client도 실제 연결 객체를
만들 때만 지연 import하므로, 설정·변환·검색 결과 처리를 독립 테스트할 수
있습니다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from src.const.qdrant_config import QdrantSettings
from src.vector_db.ingestion import PreparedPoint
from src.vector_db.validation import StoredPointSnapshot
from src.vectorstore import (
    VectorCollectionInfo,
    VectorSearchHit,
    VectorSearchRequest,
    VectorStoreError,
)


class QdrantClientLike(Protocol):
    """실제 Qdrant Client와 테스트 대역이 따라야 하는 최소 API."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def get_collection(self, collection_name: str) -> object: ...

    def create_collection(self, **kwargs: object) -> bool: ...

    def query_points(self, **kwargs: object) -> object: ...

    def scroll(self, **kwargs: object) -> object: ...

    def upload_points(self, **kwargs: object) -> None: ...


PointFactory = Callable[[PreparedPoint, str | None], object]
FilterFactory = Callable[[Mapping[str, object]], object]
VectorParamsFactory = Callable[[int, str, str | None], object]


def create_qdrant_client(settings: QdrantSettings) -> QdrantClientLike:
    """설정으로 동기 Qdrant Client를 만듭니다. 서버 요청은 하지 않습니다."""

    settings.connection.validate()
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Qdrant 연결에는 qdrant-client 설치가 필요합니다."
        ) from error

    return QdrantClient(
        url=settings.connection.url,
        api_key=settings.connection.api_key,
        timeout=settings.connection.timeout_seconds,
        prefer_grpc=settings.connection.prefer_grpc,
    )


def _default_point_factory(
    point: PreparedPoint,
    vector_name: str | None,
) -> object:
    """저장 독립 Point를 Qdrant SDK PointStruct로 변환합니다."""

    try:
        from qdrant_client.models import PointStruct
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Qdrant Point 변환에는 qdrant-client 설치가 필요합니다."
        ) from error

    vector: object = list(point.vector)
    if vector_name is not None:
        vector = {vector_name: list(point.vector)}

    return PointStruct(
        id=point.point_id,
        vector=vector,
        payload=dict(point.payload),
    )


def _default_filter_factory(filters: Mapping[str, object]) -> object:
    """단순 동등 조건을 Qdrant Filter로 변환합니다."""

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Qdrant Filter 변환에는 qdrant-client 설치가 필요합니다."
        ) from error

    conditions = []
    for key, value in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("필터 키는 비어 있지 않은 문자열이어야 합니다.")
        if isinstance(value, float) or not isinstance(value, (str, int, bool)):
            raise TypeError(
                "현재 필터 값은 문자열·정수·불리언만 지원합니다: "
                f"{key}"
            )
        conditions.append(
            FieldCondition(key=key, match=MatchValue(value=value))
        )
    return Filter(must=conditions)


def _default_vector_params_factory(
    size: int,
    distance: str,
    vector_name: str | None,
) -> object:
    """팀 컬렉션 규격을 Qdrant SDK Vector 설정으로 변환합니다."""

    try:
        from qdrant_client.models import Distance, VectorParams
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Qdrant 컬렉션 생성에는 qdrant-client 설치가 필요합니다."
        ) from error

    distance_value = getattr(Distance, distance.upper())
    vector_params = VectorParams(size=size, distance=distance_value)
    if vector_name is not None:
        return {vector_name: vector_params}
    return vector_params


def _read_value(target: object, name: str, default: object = None) -> object:
    """SDK 객체와 테스트 사전에서 같은 방식으로 값을 읽습니다."""

    if isinstance(target, Mapping):
        return target.get(name, default)
    return getattr(target, name, default)


def _read_path(target: object, *names: str) -> object:
    value = target
    for name in names:
        value = _read_value(value, name)
        if value is None:
            return None
    return value


def _enum_text(value: object) -> str:
    raw_value = getattr(value, "value", value)
    return str(raw_value).lower()


@dataclass
class QdrantVectorStore:
    """Qdrant 검색·적재 구현체.

    컬렉션 삭제나 재생성 기능은 의도적으로 제공하지 않습니다.
    """

    client: QdrantClientLike
    settings: QdrantSettings
    point_factory: PointFactory = _default_point_factory
    filter_factory: FilterFactory = _default_filter_factory
    vector_params_factory: VectorParamsFactory = _default_vector_params_factory

    @classmethod
    def from_settings(cls, settings: QdrantSettings) -> QdrantVectorStore:
        """설정으로 Client를 만들되 컬렉션은 변경하지 않습니다."""

        return cls(client=create_qdrant_client(settings), settings=settings)

    @property
    def collection_name(self) -> str:
        name = self.settings.collection.name
        if name is None:
            raise ValueError("QDRANT_COLLECTION이 설정되지 않았습니다.")
        return name

    def health_check(self) -> bool:
        """서버에 접속하여 대상 컬렉션의 존재 여부를 확인합니다."""

        try:
            return bool(self.client.collection_exists(self.collection_name))
        except Exception as error:
            raise VectorStoreError(
                "connection_error",
                "Qdrant 연결 또는 컬렉션 확인에 실패했습니다.",
            ) from error

    def ensure_collection(self) -> tuple[VectorCollectionInfo, bool]:
        """컬렉션이 없으면 생성하고, 있으면 규격만 검사합니다.

        반환값의 두 번째 값은 이번 호출에서 새로 생성했는지 여부입니다.
        기존 컬렉션 삭제·재생성은 수행하지 않습니다.
        """

        self.settings.validate_for_vector_work()
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as error:
            raise VectorStoreError(
                "connection_error",
                "Qdrant 컬렉션 존재 여부를 확인하지 못했습니다.",
            ) from error

        if exists:
            return self.validate_collection_contract(), False

        vector_size = self.settings.collection.vector_size
        assert vector_size is not None
        vectors_config = self.vector_params_factory(
            vector_size,
            self.settings.collection.distance,
            self.settings.collection.vector_name,
        )
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )
        except Exception as error:
            raise VectorStoreError(
                "connection_error",
                "Qdrant 컬렉션 생성에 실패했습니다.",
            ) from error

        return self.validate_collection_contract(), True

    def _vector_params(self, collection: object) -> tuple[object, str | None]:
        vectors = _read_path(collection, "config", "params", "vectors")
        if vectors is None:
            raise VectorStoreError(
                "invalid_schema",
                "Qdrant 컬렉션의 벡터 설정을 읽을 수 없습니다.",
            )

        vector_name = self.settings.collection.vector_name
        if isinstance(vectors, Mapping):
            if vector_name is None:
                raise VectorStoreError(
                    "invalid_schema",
                    "Named Vector 컬렉션에는 QDRANT_VECTOR_NAME이 필요합니다.",
                )
            if vector_name not in vectors:
                raise VectorStoreError(
                    "invalid_schema",
                    "설정한 Named Vector가 컬렉션에 없습니다.",
                )
            return vectors[vector_name], vector_name

        if vector_name is not None:
            raise VectorStoreError(
                "invalid_schema",
                "Unnamed Vector 컬렉션에는 QDRANT_VECTOR_NAME을 사용할 수 없습니다.",
            )
        return vectors, None

    def get_collection_info(self) -> VectorCollectionInfo:
        """현재 컬렉션의 공개 가능한 상태와 벡터 규격을 반환합니다."""

        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception as error:
            raise VectorStoreError(
                "connection_error",
                "Qdrant 컬렉션 정보를 읽지 못했습니다.",
            ) from error

        vector_params, vector_name = self._vector_params(collection)
        vector_size = _read_value(vector_params, "size")
        distance = _read_value(vector_params, "distance")
        if isinstance(vector_size, bool) or not isinstance(vector_size, int):
            raise VectorStoreError(
                "invalid_schema",
                "Qdrant 컬렉션의 벡터 차원이 올바르지 않습니다.",
            )

        points_count = _read_value(collection, "points_count", 0)
        if not isinstance(points_count, int):
            points_count = 0

        return VectorCollectionInfo(
            name=self.collection_name,
            vector_size=vector_size,
            distance=_enum_text(distance),
            points_count=points_count,
            status=_enum_text(_read_value(collection, "status", "unknown")),
            vector_name=vector_name,
        )

    def validate_collection_contract(self) -> VectorCollectionInfo:
        """실제 컬렉션이 환경설정의 차원·거리와 같은지 확인합니다."""

        self.settings.validate_for_collection_creation()
        info = self.get_collection_info()
        expected_size = self.settings.collection.vector_size
        if info.vector_size != expected_size:
            raise VectorStoreError(
                "dimension_mismatch",
                "Qdrant 컬렉션과 임베딩 벡터 차원이 다릅니다.",
            )
        if info.distance != self.settings.collection.distance:
            raise VectorStoreError(
                "distance_mismatch",
                "Qdrant 컬렉션과 설정의 거리 함수가 다릅니다.",
            )
        return info

    def search(
        self,
        request: VectorSearchRequest,
    ) -> Sequence[VectorSearchHit]:
        """Qdrant ``query_points``로 유사 문서를 한 번 검색합니다."""

        query_filter = None
        if request.filters:
            query_filter = self.filter_factory(request.filters)

        query_kwargs: dict[str, object] = {
            "collection_name": self.collection_name,
            "query": list(request.query_vector),
            "limit": request.top_k,
            "with_payload": True,
            "with_vectors": False,
        }
        if self.settings.collection.vector_name is not None:
            query_kwargs["using"] = self.settings.collection.vector_name
        if request.score_threshold is not None:
            query_kwargs["score_threshold"] = request.score_threshold
        if query_filter is not None:
            query_kwargs["query_filter"] = query_filter

        try:
            response = self.client.query_points(**query_kwargs)
        except Exception as error:
            raise VectorStoreError(
                "connection_error",
                "Qdrant 문서 검색에 실패했습니다.",
            ) from error

        result_points = _read_value(response, "points", response)
        if not isinstance(result_points, Sequence) or isinstance(
            result_points,
            (str, bytes),
        ):
            raise VectorStoreError(
                "invalid_response",
                "Qdrant 검색 결과 형식이 올바르지 않습니다.",
            )

        hits: list[VectorSearchHit] = []
        for point in result_points:
            payload = _read_value(point, "payload", {})
            if not isinstance(payload, Mapping):
                payload = {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise VectorStoreError(
                    "invalid_response",
                    "Qdrant 검색 결과에 text Payload가 없습니다.",
                )
            hits.append(
                VectorSearchHit(
                    point_id=str(_read_value(point, "id", "")),
                    score=float(_read_value(point, "score", 0.0)),
                    text=text,
                    metadata={
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                )
            )
        return tuple(hits)

    def upsert_points(
        self,
        points: Sequence[PreparedPoint],
        *,
        batch_size: int = 64,
        parallel: int = 2,
        max_retries: int = 3,
    ) -> int:
        """Qdrant 대량 적재 기능으로 Point를 병렬 Upsert합니다."""

        upload_options = {
            "batch_size": batch_size,
            "parallel": parallel,
            "max_retries": max_retries,
        }
        for option_name, option_value in upload_options.items():
            if isinstance(option_value, bool) or not isinstance(
                option_value,
                int,
            ):
                raise TypeError(f"{option_name}는 정수여야 합니다.")
            minimum = 0 if option_name == "max_retries" else 1
            if option_value < minimum:
                raise ValueError(
                    f"{option_name}는 {minimum} 이상이어야 합니다."
                )

        if not points:
            return 0

        sdk_points = (
            self.point_factory(point, self.settings.collection.vector_name)
            for point in points
        )
        try:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=sdk_points,
                batch_size=batch_size,
                parallel=parallel,
                max_retries=max_retries,
                wait=True,
            )
        except Exception as error:
            raise VectorStoreError(
                "connection_error",
                "Qdrant Point 적재에 실패했습니다.",
            ) from error

        return len(points)

    def read_document_points(
        self,
        document_ids: Sequence[str],
        *,
        page_size: int = 256,
    ) -> Sequence[StoredPointSnapshot]:
        """검증용으로 지정 문서의 Point ID와 Payload만 읽습니다.

        벡터는 내려받지 않으며, 저장된 Point를 삭제하거나 수정하지 않습니다.
        """

        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise TypeError("page_size는 정수여야 합니다.")
        if page_size < 1:
            raise ValueError("page_size는 1 이상이어야 합니다.")

        normalized_ids = sorted(
            {
                document_id.strip()
                for document_id in document_ids
                if isinstance(document_id, str) and document_id.strip()
            }
        )
        snapshots: list[StoredPointSnapshot] = []
        for document_id in normalized_ids:
            query_filter = self.filter_factory({"document_id": document_id})
            offset: object = None
            while True:
                scroll_kwargs: dict[str, object] = {
                    "collection_name": self.collection_name,
                    "scroll_filter": query_filter,
                    "limit": page_size,
                    "with_payload": True,
                    "with_vectors": False,
                }
                if offset is not None:
                    scroll_kwargs["offset"] = offset

                try:
                    response = self.client.scroll(**scroll_kwargs)
                except Exception as error:
                    raise VectorStoreError(
                        "connection_error",
                        "Qdrant 적재 검증용 Point 조회에 실패했습니다.",
                    ) from error

                if not isinstance(response, tuple) or len(response) != 2:
                    raise VectorStoreError(
                        "invalid_response",
                        "Qdrant Scroll 결과 형식이 올바르지 않습니다.",
                    )
                records, offset = response
                if not isinstance(records, Sequence) or isinstance(
                    records,
                    (str, bytes),
                ):
                    raise VectorStoreError(
                        "invalid_response",
                        "Qdrant Scroll Point 목록 형식이 올바르지 않습니다.",
                    )

                for record in records:
                    payload = _read_value(record, "payload", {})
                    if not isinstance(payload, Mapping):
                        payload = {}
                    snapshots.append(
                        StoredPointSnapshot(
                            point_id=str(_read_value(record, "id", "")),
                            payload=dict(payload),
                        )
                    )
                if offset is None:
                    break

        return tuple(snapshots)
