import pandas as pd
from langchain_upstage import ChatUpstage # 👈 OpenAI에서 Upstage로 변경
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os

# --- 1. Upstage API 키 설정 ---
# (upstage_api_key.txt 파일에서 키를 읽어옵니다)
try:
    with open("upstage_api_key.txt", "r") as f:
        # 업스테이지는 'UPSTAGE_API_KEY'라는 이름의 환경 변수를 사용합니다.
        os.environ["UPSTAGE_API_KEY"] = f.read().strip()
    print("✅ Upstage API 키를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("🛑 [오류] 'upstage_api_key.txt' 파일을 찾을 수 없습니다.")
    print("이 코드 파일과 같은 폴더에 API 키를 담은 텍스트 파일을 만들어주세요.")
    exit()

# --- 2. Upstage 'Solar' 모델 준비 ---
# 👈 AI 모델을 ChatOpenAI에서 ChatUpstage로 변경
# 한국어에 강력한 'solar-1-mini-chat' 모델을 사용합니다.
llm = ChatUpstage(model="solar-1-mini-chat", temperature=0)

# --- 3. CSV 파일을 Pandas DataFrame으로 읽기 ---
file_path = "../donga_job_postings.csv"
try:
    df = pd.read_csv(file_path, encoding='cp949')
    print(f"✅ '{file_path}' 파일을 성공적으로 불러왔습니다.")
    df.columns = df.columns.str.replace(' ', '_')
except FileNotFoundError:
    print(f"🛑 [오류] '{file_path}' 파일을 찾을 수 없습니다.")
    exit()

# --- 4. LangChain 에이전트 생성 ---
# (이 부분은 OpenAI 때와 완전히 동일합니다)
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True, # AI가 생각하는 과정을 보여줌
    allow_dangerous_code=True # AI가 Pandas 코드를 직접 실행하도록 허용
)

print("\n" + "="*50)
print("🤖 업스테이지 AI 에이전트가 준비되었습니다.")
print("CSV 파일에 대해 무엇이든 물어보세요! (종료하려면 'exit' 입력)")
print("="*50)

# --- 5. AI에게 질문하기 (무한 루프) ---
while True:
    try:
        query = input("질문: ")
        if query.lower() == "exit":
            print("🤖 테스트를 종료합니다.")
            break
        
        response = agent.invoke(query)
        print("\nAI 답변:", response['output'], "\n" + "-"*50)
    
    except Exception as e:
        print(f"\n[오류 발생] {e}")
        continue