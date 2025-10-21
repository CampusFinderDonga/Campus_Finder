import os
import re
import shutil
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.chains import ConversationalRetrievalChain


# ======================================
# 1️⃣ 환경설정 및 상수
# ======================================
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    raise ValueError("❌ .env 파일에 UPSTAGE_API_KEY가 없습니다.")

PDF_FOLDER = r"C:\Users\seogu\Documents\CampusFinder\PDFs"
DB_PATH = "pdf_chroma_db"

# ======================================
# 2️⃣ PDF 텍스트 추출 함수
# ======================================
def extract_text_from_pdfs(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("❌ PDFs 폴더에 PDF 파일이 없습니다.")

    print(f"📄 총 {len(pdf_files)}개의 PDF 파일을 감지했습니다.\n")
    all_texts = []

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        # 전처리: 공백, 줄바꿈, 특수문자 정리
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\u200b\xa0]", " ", text)

        if text.strip():
            all_texts.append(f"[문서: {filename}]\n{text.strip()}")
        else:
            print(f"⚠️ {filename}에서 텍스트를 추출하지 못했습니다.")
    return all_texts

# ======================================
# 3️⃣ 벡터 DB 생성
# ======================================
def build_vector_db(texts):
    print("\n🧠 텍스트 분할 및 임베딩 생성 중...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = []
    for doc in texts:
        chunks.extend(splitter.split_text(doc))
    print(f"✂️ 총 {len(chunks)}개의 청크 생성 완료")

    # 기존 DB 삭제
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("🗑 기존 DB 삭제 완료")

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=DB_PATH)
    vectorstore.persist()
    print("✅ 새로운 벡터 DB 생성 완료!")
    return vectorstore

# ======================================
# 4️⃣ 사용자 질문 의도 확장 (semantic reformulation)
# ======================================
def refine_query(query: str) -> str:
    # 의도 기반 질문 보정
    query = query.strip().lower()
    replacements = {
        "이게 뭐야": "정의와 개념을 설명해줘",
        "이건 뭐야": "정의와 개념을 알려줘",
        "어떻게": "절차나 방법을 설명해줘",
        "왜": "이유와 목적을 알려줘",
        "비교": "차이점을 설명해줘",
        "같은가": "유사점과 차이점을 알려줘"
    }
    for key, val in replacements.items():
        if key in query:
            query += f" ({val})"
    return query

# ======================================
# 5️⃣ 챗봇 실행
# ======================================
def run_conversational_rag():
    texts = extract_text_from_pdfs(PDF_FOLDER)
    vectorstore = build_vector_db(texts)

    llm = ChatUpstage(model="solar-pro")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    chat_history = []
    print("\n🎓 캠퍼스 파인더 대화형 RAG 챗봇 시작!")
    print("💬 질문을 입력하세요. (종료하려면 exit 입력)\n")

    while True:
        query = input("❓ 질문: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 챗봇을 종료합니다.")
            break

        # 질문 의도 보정
        refined = refine_query(query)
        try:
            result = qa_chain.invoke({"question": refined, "chat_history": chat_history})
            answer = result["answer"].strip()
            print(f"\n🤖 답변:\n{answer}\n")
            chat_history.append((query, answer))
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

# ======================================
# 🚀 메인 실행
# ======================================
if __name__ == "__main__":
    run_conversational_rag()
