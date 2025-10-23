# 필요한 라이브러리 설치
# pip install selenium webdriver-manager

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from datetime import datetime
import time

# -------------------------------
# 1️⃣ 로그인 정보 입력
# -------------------------------
LOGIN_ID = "2215165"
LOGIN_PW = "Dd2215165!"

# -------------------------------
# 2️⃣ 웹드라이버 실행
# -------------------------------
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# -------------------------------
# 3️⃣ 로그인 페이지 접속 및 대기
# -------------------------------
driver.get("https://job.donga.ac.kr/login")
wait = WebDriverWait(driver, 20)
wait.until(EC.presence_of_element_located((By.ID, "login_id")))

# -------------------------------
# 4️⃣ 로그인 입력 및 버튼 클릭
# -------------------------------
driver.find_element(By.ID, "login_id").send_keys(LOGIN_ID)
driver.find_element(By.ID, "login_pw").send_keys(LOGIN_PW)
driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

# -------------------------------
# 5️⃣ 추천 채용정보 페이지 이동
# -------------------------------
time.sleep(1)
driver.get("https://job.donga.ac.kr/jobinfo/recommend")

# -------------------------------
# 6️⃣ 채용공고 데이터 추출 (✨ 페이지 넘기기 & 마감일 필터링 추가)
# -------------------------------
today = datetime.now().date() # 오늘 날짜 (YYYY-MM-DD 형식)
scraped_jobs_count = 0 # 수집한 공고 수를 세기 위한 변수

print(f"오늘 날짜: {today}\n마감일이 지나지 않은 공고만 수집합니다.\n" + "="*60)

# ✨ [페이지 넘기기] 마지막 페이지까지 반복하기 위한 while문
while True:
    try:
        # 현재 페이지의 채용 공고 목록이 나타날 때까지 기다립니다.
        jobs = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, ".list-employment tbody > tr")
        ))
        
        # 현재 페이지의 모든 공고를 하나씩 확인
        for job in jobs:
            try:
                deadline_str = job.find_element(By.CSS_SELECTOR, "td.td_deadline").text
                
                # ✨ [마감일 필터링] 마감일을 날짜 형식으로 변환하고 오늘과 비교
                deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
                if deadline_date >= today:
                    scraped_jobs_count += 1
                    company = job.find_element(By.CSS_SELECTOR, "td.td_company").text
                    title = job.find_element(By.CSS_SELECTOR, "td.td_subject").text
                    job_type = job.find_element(By.CSS_SELECTOR, "td.td_jobtype").text
                    area = job.find_element(By.CSS_SELECTOR, "td.td_area").text
                    hits = job.find_element(By.CSS_SELECTOR, "td.td_hit").text

                    print(f"[{scraped_jobs_count}] {company} | {title} | {job_type} | {area} | 마감: {deadline_str} | 조회수: {hits}")
                    print("-"*60)
            
            # 빈 행(tr)이 있거나 구조가 다른 행은 건너뜁니다.
            except Exception:
                continue
                
        # ✨ [페이지 넘기기] '다음' 버튼을 찾아 클릭합니다.
        # 버튼이 없으면 NoSuchElementException이 발생하고 루프가 종료됩니다.
        next_button = driver.find_element(By.LINK_TEXT, "다음")
        next_button.click()
        
        # 다음 페이지가 로딩될 때까지 잠시 기다립니다.
        time.sleep(2)

    # '다음' 버튼을 더 이상 찾을 수 없을 때 (마지막 페이지일 때)
    except NoSuchElementException:
        print("\n✅ 마지막 페이지입니다. 모든 공고 수집을 완료했습니다.")
        break
    # 페이지 로딩 중 오류 등 예외 상황 처리
    except TimeoutException:
        print("\n😭 페이지 로딩 시간이 초과되었습니다. 크롤링을 중단합니다.")
        break

# -------------------------------
# 7️⃣ 종료
# -------------------------------
driver.quit()
print(f"🎉 총 {scraped_jobs_count}개의 유효한 공고를 찾았습니다.")
print("🔚 크롤링 완료! 브라우저 닫음.")