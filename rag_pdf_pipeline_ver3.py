import os
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage, UpstageDocumentParseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# ==============================
# 🔧 1. 환경 설정
# ==============================
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    raise ValueError("❌ Upstage API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

PDF_FOLDER = "PDFs"
TXT_FOLDER = "PDFs_text"
DB_PATH = "chroma_db_v2"

os.makedirs(TXT_FOLDER, exist_ok=True)


# ==============================
# 📄 2. PDF → TXT 변환
# ==============================
def convert_pdfs_to_text():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ PDFs 폴더에 PDF 파일이 없습니다.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        txt_name = pdf_file.replace(".pdf", ".txt")
        txt_path = os.path.join(TXT_FOLDER, txt_name)

        if os.path.exists(txt_path):
            print(f"📘 이미 변환됨: {pdf_file}")
            continue

        print(f"🔍 PDF 변환 중: {pdf_file}")
        try:
            loader = UpstageDocumentParseLoader(pdf_path, split="page")
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            full_text = "\n".join(texts)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"✅ 변환 완료: {txt_name}")
        except Exception as e:
            print(f"❌ 변환 실패 ({pdf_file}): {e}")


# ==============================
# 📘 3. 텍스트 로드
# ==============================
def load_documents():
    docs = []
    for filename in os.listdir(TXT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(TXT_FOLDER, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                docs.append({"source": filename.replace(".txt", ""), "text": text})
    if not docs:
        raise ValueError("⚠️ PDFs_text 폴더에 .txt 파일이 없습니다.")
    return docs


# ==============================
# ✂️ 4. 텍스트 분할
# ==============================
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = []
    for doc in docs:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append({"text": chunk, "source": doc["source"]})
    return chunks


# ==============================
# 🧠 5. 벡터스토어 생성
# ==============================
def build_vectorstore(chunks):
    print("\n🧠 벡터스토어 생성 중...")
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=DB_PATH
    )
    print(f"✅ 벡터 저장 완료! ({len(chunks)}개의 청크 저장됨)")
    return vectorstore


# ==============================
# 💬 6. RAG 대화 체인
# ==============================
def create_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatUpstage(model="solar-pro")

    # ✅ PromptTemplate 객체
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "당신은 동아대학교 학사안내 문서를 기반으로 답변하는 RAG 챗봇입니다.\n"
            "아래 문서 내용을 참고하여 질문에 답하세요.\n\n"
            "문서 내용:\n{context}\n\n"
            "질문: {question}\n\n"
            "규칙:\n"
            "- 문서에 없는 내용은 추론하지 말고 '해당 내용은 문서에 명시되어 있지 않습니다.'라고 답하세요.\n"
            "- 문서에 근거한 정확하고 간결한 답변을 작성하세요."
        )
    )

    # ✅ memory에도 output_key 지정
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # ⚡ 추가됨 (핵심 수정)
    )

    # ✅ chain 구성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context",
            "output_key": "answer"
        },
        output_key="answer"  # 여기도 유지
    )
    return chain


# ==============================
# 🚀 7. 메인 실행
# ==============================
def main():
    print("📚 동아대 RAG 시스템 초기화 중...\n")

    convert_pdfs_to_text()
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    qa_chain = create_chain(vectorstore)

    print("\n✅ 시스템 준비 완료! 질문을 입력하세요. (종료하려면 exit)\n")

    while True:
        query = input("❓ 질문: ").strip()
        if query.lower() in ["exit", "quit", "종료"]:
            print("👋 프로그램을 종료합니다.")
            break

        try:
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "답변 생성 실패")
            print(f"\n🤖 답변:\n{answer}\n")

            if "source_documents" in result:
                print("📎 참고 문서:")
                unique_sources = sorted(set([doc.metadata.get("source", "unknown") for doc in result["source_documents"]]))
                for src in unique_sources:
                    print(f" - {src}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
