import os

pdf_path = r"동아대학교 _ 전공 마이크로모듈제 안내 _ 교과과정 _ 학사안내.pdf"

print("현재 작업 경로:", os.getcwd())
print("파일 목록:", os.listdir())
print("PDF 경로:", pdf_path)
print("PDF 존재 여부:", os.path.exists(pdf_path))


import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 🔹 API 키 설정
os.environ["UPSTAGE_API_KEY"] = "up_ZqiarO6eDupuUop9IMEkRoe9WA7Al"

# 🔹 PDF 파일 경로 (직접 복사한 경로 넣기)
pdf_path = pdf_path = r"C:\Users\user\OneDrive\문서\바탕 화면\Campus Finder\동아대학교 _ 전공 마이크로모듈제 안내 _ 교과과정 _ 학사안내.pdf"

# ✅ 경로 확인용 출력 (여기서 False 나오면 경로 인식 안 된 것)
print("현재 작업 경로:", os.getcwd())
print("PDF 경로:", pdf_path)
print("PDF 존재 여부:", os.path.exists(pdf_path))

# 🔹 PDF 파일 읽기
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# 🔹 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = splitter.split_text(text)
print(f"총 분할된 문서 수: {len(docs)}")

# 🔹 업스테이지 임베딩
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

# 🔹 벡터 DB 생성
vectorstore = Chroma.from_texts(
    texts=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("✅ 벡터 저장 완료!")

# 🔹 업스테이지 LLM 로드
llm = ChatUpstage(model="solar-1-mini-chat")

# 🔹 RAG 체인 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🔹 질문 반복 입력 모드
while True:
    query = input("\n❓ 질문을 입력하세요 (종료하려면 exit 입력): ")
    if query.lower() == "exit":
        print("프로그램을 종료합니다.")
        break
    answer = qa.run(query)
    print(f"\n💬 답변: {answer}\n")

