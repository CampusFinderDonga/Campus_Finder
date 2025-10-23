import os
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader # PDF 로더 사용
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. 설정 ---

# Gemini API 키를 설정합니다. (반드시 자신의 키로 교체!)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC3H_EHCl4lAEtmo-OmoJ1dooDWSD_xD5k"

# PDF 파일 경로 설정
# 현재 스크립트 위치(donga_rag_project)에서 한 단계 상위 폴더(Campus Finder)를 가리킴
pdf_path = "../동아대학교 _ 전공 마이크로모듈제 안내 _ 교과과정 _ 학사안내.pdf"

# --- 2. 데이터 로딩 (Load) ---

# PyPDFLoader를 사용하여 지정된 PDF 파일을 불러옵니다.
print(f"'{pdf_path}' 파일 로딩 시작...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"PDF에서 {len(documents)}개 페이지를 불러왔습니다.")

# --- 3. 데이터 분할 (Split) ---

# 불러온 문서를 AI가 처리하기 좋은 크기로 자릅니다.
print("데이터 분할 시작...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = text_splitter.split_documents(documents)
print(f"분할된 텍스트 청크 개수: {len(texts)}개")

# --- 4. 데이터 저장 (Store) ---

# 분할된 텍스트를 Gemini 임베딩 모델을 사용해 벡터로 변환하고,
# ChromaDB (벡터 데이터베이스)에 저장합니다.
print("임베딩 및 벡터 저장 시작...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
print("벡터 저장 완료.")

# --- 5. RAG 체인 생성 ---

# 답변 생성을 위한 Gemini LLM을 준비합니다.
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 벡터 저장소에서 질문과 관련된 정보를 찾아주는 검색기(Retriever)를 생성합니다.
retriever = vectorstore.as_retriever()

# 검색된 정보(context)와 질문을 함께 LLM에 전달하여 답변을 생성하는 RAG 체인을 만듭니다.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\n✅ RAG 시스템이 준비되었습니다. 질문을 입력하세요.")

# --- 6. 질문 및 답변 (무한 반복) ---

while True:
    question = input("\n[질문] (종료하려면 'exit' 입력): ")
    if question.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break
    
    # RAG 체인을 실행하여 답변 생성
    result = qa_chain.invoke({"query": question})
    
    print(f"\n[답변] {result['result']}")
    
    # (선택 사항) 답변의 근거가 된 소스 문서의 일부를 보고 싶다면 아래 주석을 해제하세요.
    # print("\n[참고 소스]")
    # for doc in result['source_documents']:
    #     print(f"📄 페이지 {doc.metadata.get('page', '?')}: '{doc.page_content[:150]}...'")