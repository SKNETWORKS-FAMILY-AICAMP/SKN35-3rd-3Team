# 실행 파일을 둘 예정

# 순서


# 1. API 키를 통한 Collecting
# 2. 전처리 = Processing
# 3. 청킹 = Chunking
# 4. 임베딩 = Embedding
# 5. 벡터 DB에 저장 = VectorStore
# 6. 검색 및 유사도 높은 순으로 정렬 = Retrieval
# 7. 대답 생성 = Generation


# 각 폴더에서 __init__.py 파일에 역할을 주석으로 작성해두었으니 참고하면 좋습니다.
# 또한 각 폴더에서 데이터를 처리하거나 생성하는 과정에서 무조건 출력을 해보는 것을 추천드립니다.
# 따로 print를 구현하는 py를 하나 만들어서 해도 좋고, 각 폴더에서 구현하는 py에 print를 넣어도 좋습니다.