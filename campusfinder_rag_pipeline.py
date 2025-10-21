
import os
import sys
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# LangChain / Upstage
from langchain_upstage import UpstageEmbeddings, ChatUpstage, UpstageDocumentParseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# ==============================
# 🔧 0. 환경설정
# ==============================
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise ValueError("❌ Upstage API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 루트 데이터 디렉터리 (사용자 폴더 트리의 최상위)
ROOT_DATA_DIR = os.getenv("CAMPUSFINDER_DATA_DIR", "CampusFinder_Data")

# Chroma 벡터 DB 경로
DB_PATH = os.getenv("CF_CHROMA_DIR", "chroma_cf")

# 청크 기본 설정
CHUNK_SIZE = int(os.getenv("CF_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CF_CHUNK_OVERLAP", "120"))

# 검색 기본값
TOP_K = int(os.getenv("CF_TOP_K", "8"))


# ==============================
# 📁 1. 폴더 순회: PDF 수집
# ==============================
def iter_pdfs(root_dir: str) -> List[Tuple[str, str]]:
    """
    루트 폴더 하위의 모든 PDF 경로를 찾아서 반환합니다.
    반환: [(absolute_pdf_path, category_path_rel), ...]
      - category_path_rel: 루트로부터의 상대 폴더 경로 (예: '대학_공통정보/학사안내/성적')
    """
    results: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        rel_dir = "" if rel_dir == "." else rel_dir.replace("\\", "/")
        for fname in filenames:
            if fname.lower().endswith(".pdf"):
                results.append((os.path.join(dirpath, fname), rel_dir))
    results.sort()
    return results


# ==============================
# 📄 2. PDF → Documents (Upstage Document Parse)
# ==============================
def parse_pdf_to_documents(pdf_path: str) -> List[str]:
    """
    UpstageDocumentParseLoader를 사용하여 PDF를 페이지 단위 텍스트로 변환합니다.
    반환: 페이지 텍스트 리스트
    """
    loader = UpstageDocumentParseLoader(pdf_path, split="page")
    docs = loader.load()
    return [d.page_content for d in docs]


# ==============================
# ✂️ 3. 페이지 텍스트 → 청크 분할
# ==============================
def split_pages_to_chunks(pages: List[str]) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks: List[str] = []
    for page_text in pages:
        chunks.extend(splitter.split_text(page_text))
    return chunks


# ==============================
# 🧠 4. 벡터 스토어 구축/적재
# ==============================
def build_or_load_vectorstore(root_dir: str, db_path: str) -> Chroma:
    """
    이미 DB가 존재하면 로드, 없으면 생성합니다.
    각 청크에는 다음 메타데이터를 저장합니다.
      - source: 파일명 (확장자 제외)
      - category: 루트로부터의 폴더 경로 (예: '대학_공통정보/학사안내/성적')
      - file_path: 루트로부터의 파일 상대경로 (카테고리/파일.pdf)
    """
    # 이미 DB가 있으면 로드
    if os.path.isdir(db_path) and len(os.listdir(db_path)) > 0:
        embedding = UpstageEmbeddings(model="solar-embedding-1-large")
        return Chroma(persist_directory=db_path, embedding_function=embedding)

    print("📚 신규 임베딩 시작… (처음 1회만 실행됩니다)\n")
    pdfs = iter_pdfs(root_dir)
    if not pdfs:
        raise FileNotFoundError(f"❌ '{root_dir}' 에서 PDF를 찾지 못했습니다. 폴더 경로를 확인해주세요.")

    all_texts: List[str] = []
    all_metadatas: List[Dict[str, str]] = []

    for idx, (abs_pdf, rel_category) in enumerate(pdfs, 1):
        rel_file_path = os.path.relpath(abs_pdf, root_dir).replace("\\", "/")
        file_stem = os.path.splitext(os.path.basename(abs_pdf))[0]

        print(f"[{idx}/{len(pdfs)}] 🔍 파싱: {rel_file_path}")
        try:
            pages = parse_pdf_to_documents(abs_pdf)
            chunks = split_pages_to_chunks(pages)
            # 메타데이터 부착
            for ch in chunks:
                all_texts.append(ch)
                all_metadatas.append({
                    "source": file_stem,
                    "category": rel_category,
                    "file_path": rel_file_path,
                })
        except Exception as e:
            print(f"   ⚠️ 파싱 실패: {rel_file_path} -> {e}")
            continue

    print(f"\n🧠 벡터 임베딩 변환 중… (총 {len(all_texts)}개 청크)")
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embedding,
        metadatas=all_metadatas,
        persist_directory=db_path,
    )
    print("✅ 벡터 저장 완료!")
    return vectorstore


# ==============================
# 🧭 5. 질의 → 카테고리 후보 자동 매핑 (룰 기반 1차 버전)
# ==============================
CATEGORY_RULES: List[Tuple[str, str]] = [
    # 학사안내 하위
    ("수업",               "대학_공통정보/학사안내/수업"),
    ("교과과정",           "대학_공통정보/학사안내/교과과정"),
    ("학적",               "대학_공통정보/학사안내/학적"),
    ("성적",               "대학_공통정보/학사안내/성적"),
    ("졸업",               "대학_공통정보/학사안내/졸업"),
    ("다전공",             "대학_공통정보/학사안내/다전공안내"),
    ("복수전공",           "대학_공통정보/학사안내/다전공안내"),
    ("부전공",             "대학_공통정보/학사안내/다전공안내"),
    ("학점인정",           "대학_공통정보/학사안내/학점인정_신청"),
    ("장학",               "대학_공통정보/학사안내/장학"),
    ("등록금",             "대학_공통정보/학사안내/등록"),
    ("등록",               "대학_공통정보/학사안내/등록"),
    ("예비군",             "대학_공통정보/학사안내/예비군"),
    ("학사일정",           "대학_공통정보/학사일정_동아광장"),

    # 동아광장
    ("FAQ",               "대학_공통정보/동아광장/FAQ"),
    ("생활정보",           "대학_공통정보/동아광장/생활정보"),
    ("학생자치",           "대학_공통정보/동아광장/학생자치활동"),

    # 경영대학 하위 (필요시 확장)
    ("경영학과",           "경영대학_학과별정보/경영학과"),
    ("관광경영학과",       "경영대학_학과별정보/관광경영학과"),
    ("국제무역학과",       "경영대학_학과별정보/국제무역학과"),
    ("경영정보학과",       "경영대학_학과별정보/경영정보학과"),
    ("금융학과",           "경영대학_학과별정보/금융학과"),
    ("경영대학",           "경영대학_학과별정보/공통_경영대학소개"),

    # 비교과 & 취업
    ("비교과",             "비교과_DECO시스템"),
    ("DECO",              "비교과_DECO시스템"),
    ("취업",               "취업지원_채용정보"),
    ("채용",               "취업지원_채용정보"),
]

def map_query_to_categories(query: str) -> List[str]:
    """
    간단한 규칙 기반으로 질의에서 카테고리 후보를 도출합니다.
    - '폴더:카테고리/하위' 형태가 있으면 그대로 사용 (사용자 강제 지정)
    - 규칙과 매칭되는 항목을 수집 (중복 제거)
    """
    query = (query or "").strip()
    cats: List[str] = []

    # 사용자가 직접 지정: 예) 폴더: 학사안내/성적  또는  folder: 경영대학_학과별정보/경영학과
    for token in ["폴더:", "folder:"]:
        if token in query:
            after = query.split(token, 1)[1].strip()
            # 공백/문장부호 앞까지만
            after = after.split()[0].strip().strip(" ,.;)」』]")
            # 상대경로를 표준화
            if not after.startswith(("대학_공통정보", "경영대학_학과별정보", "취업지원_채용정보", "비교과_DECO시스템")):
                # 사용자가 '학사안내/성적' 같은 상대만 적었을 수 있음 → 대학_공통정보/ 접두어 보정
                after = f"대학_공통정보/{after}".replace("//", "/")
            cats.append(after.replace("\\", "/"))

    q = query.replace(" ", "")
    for key, cat in CATEGORY_RULES:
        if key in q:
            cats.append(cat)

    # 중복 제거 & 최대 3개까지만 (너무 넓어지는 걸 방지)
    uniq = []
    for c in cats:
        if c not in uniq:
            uniq.append(c)
    return uniq[:3]


# ==============================
# 🔎 6. 카테고리 필터링 가능한 Retriever
# ==============================
def make_retriever(vs: Chroma, top_k: int = TOP_K, categories: Optional[List[str]] = None):
    """
    categories가 주어지면 해당 카테고리(들)만 대상으로 검색합니다.
    """
    search_kwargs = {"k": top_k}
    if categories:
        # Chroma 필터 문법: {"category": {"$in": [...]}}  또는 정확일치 {"category": "..."}
        if len(categories) == 1:
            search_kwargs["filter"] = {"category": categories[0]}
        else:
            search_kwargs["filter"] = {"category": {"$in": categories}}
    return vs.as_retriever(search_kwargs=search_kwargs)


# ==============================
# 🧵 7. 대화형 RAG 체인
# ==============================
def build_chain(vs: Chroma, categories: Optional[List[str]] = None):
    retriever = make_retriever(vs, TOP_K, categories)
    llm = ChatUpstage(model="solar-pro")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "당신은 동아대학교 관련 공식 문서를 기반으로 답변하는 RAG 챗봇입니다.\n"
            "아래의 참고 문서 조각을 근거로 질문에 정확하고 간결하게 답하세요.\n\n"
            "참고 문서 내용:\n{context}\n\n"
            "질문: {question}\n\n"
            "응답 규칙:\n"
            "- 문서에 없는 내용은 추측하지 말고 '해당 내용은 제공된 문서에 명시되어 있지 않습니다.'라고 답하세요.\n"
            "- 핵심만 정리하여 한국어로 답변하세요.\n"
            "- 필요하면 항목(•)으로 깔끔하게 정리하세요."
        ),
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
            "output_key": "answer",
        },
        output_key="answer",
    )
    return chain


# ==============================
# 🚀 8. 메인 루프
# ==============================
def main():
    print("📚 CampusFinder RAG 초기화 중…\n")
    vectorstore = build_or_load_vectorstore(ROOT_DATA_DIR, DB_PATH)
    print("\n✅ 시스템 준비 완료! 질문을 입력하세요. (종료: exit/quit/종료)\n")

    # 카테고리 모드 안내
    print("💡 힌트) 특정 폴더만 검색하려면 질의에 '폴더: 학사안내/성적' 처럼 적어주세요.")
    print("   예) '졸업요건 알려줘 폴더: 학사안내/졸업'")

    qa_chain = build_chain(vectorstore)  # 최초엔 전체 검색

    while True:
        try:
            query = input("\n❓ 질문: ").strip()
        except EOFError:
            break

        if query.lower() in {"exit", "quit", "종료"}:
            print("👋 프로그램을 종료합니다.")
            break

        # 질의 → 카테고리 추론
        cats = map_query_to_categories(query)
        if cats:
            print(f"🔎 카테고리 우선 검색 적용: {cats}")
            qa_chain = build_chain(vectorstore, categories=cats)
        else:
            qa_chain = build_chain(vectorstore, categories=None)

        try:
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "").strip()
            print(f"\n🤖 답변:\n{answer}\n")

            # 참고 문서 출력
            if "source_documents" in result:
                seen = set()
                refs = []
                for d in result["source_documents"]:
                    meta = d.metadata or {}
                    ref = (meta.get("file_path") or meta.get("source") or "unknown", meta.get("category", ""))
                    if ref not in seen:
                        seen.add(ref)
                        refs.append(ref)

                if refs:
                    print("📎 참고 문서:")
                    for fp, cat in refs:
                        cat_disp = f" ({cat})" if cat else ""
                        print(f" - {fp}{cat_disp}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    # 옵션: 루트 폴더/DB 경로를 커맨드라인에서 전달 가능
    # 사용: python campusfinder_rag_pipeline.py [DATA_DIR] [DB_PATH]
    if len(sys.argv) >= 2:
        ROOT_DATA_DIR = sys.argv[1]
    if len(sys.argv) >= 3:
        DB_PATH = sys.argv[2]
    main()
