import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import shutil

# ======================================
# 🔹 1. 환경 설정
# ======================================

load_dotenv()  # .env 파일 로드
api_key = os.getenv("UPSTAGE_API_KEY")

if not api_key:
    raise ValueError("❌ .env 파일에 'UPSTAGE_API_KEY'가 없습니다.")

PDF_FOLDER = r"C:\Users\seogu\Documents\CampusFinder\PDFs"
DB_PATH = "pdf_chroma_db"

# ======================================
# 🔹 2. PDF 읽기 함수
# ======================================

def extract_text_from_pdfs(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("❌ PDFs 폴더 안에 PDF 파일이 없습니다.")

    print(f"📄 총 {len(pdf_files)}개의 PDF 파일을 찾았습니다.\n")

    all_texts = []

    for filename in pdf_files:
        file_path = os.path.join(pdf_folder, filename)
        print(f"🔍 {filename} 텍스트 추출 중...")

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        # PDF 이름 태그 추가
        if text.strip():
            all_texts.append(f"[문서: {filename}]\n{text.strip()}")
        else:
            print(f"⚠️ {filename}에서 텍스트를 추출하지 못했습니다.")

    return all_texts

# ======================================
# 🔹 3. 벡터스토어 구축
# ======================================

def build_vector_db(texts):
    print("\n🧠 텍스트 분할 및 임베딩 생성 중...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = []
    for doc in texts:
        chunks.extend(splitter.split_text(doc))

    print(f"✂️ 총 {len(chunks)}개의 청크 생성 완료")

    # 기존 DB 삭제
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("🗑 기존 Chroma DB 삭제 완료")

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    vectorstore.persist()
    print("✅ 벡터 DB 생성 완료!")

    return vectorstore

# ======================================
# 🔹 4. RAG 기반 질의응답 실행
# ======================================

def run_rag_chatbot(vectorstore):
    llm = ChatUpstage(model="solar-pro")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\n🎓 캠퍼스 파인더 PDF RAG 챗봇 시작!")
    print("💬 질문을 입력하세요 (종료하려면 exit 입력)\n")

    while True:
        query = input("❓ 질문: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 챗봇을 종료합니다.")
            break
        try:
            result = qa_chain.invoke({"query": query})
            print(f"\n🤖 답변:\n{result['result']}\n")
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

# ======================================
# 🚀 실행
# ======================================

if __name__ == "__main__":
    texts = extract_text_from_pdfs(PDF_FOLDER)
    vectorstore = build_vector_db(texts)
    run_rag_chatbot(vectorstore)
