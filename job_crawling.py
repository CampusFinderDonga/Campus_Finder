# 필요한 라이브러리
import csv
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
# 1️⃣ 로그인 및 페이지 이동
# -------------------------------
LOGIN_ID = "아이디"
LOGIN_PW = "비밀번호"

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://job.donga.ac.kr/login")
wait = WebDriverWait(driver, 15)
wait.until(EC.presence_of_element_located((By.ID, "login_id")))

driver.find_element(By.ID, "login_id").send_keys(LOGIN_ID)
driver.find_element(By.ID, "login_pw").send_keys(LOGIN_PW)
driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
time.sleep(1)
driver.get("https://job.donga.ac.kr/jobinfo/recommend")

# ----------------------------------------------------
# ✨ '내용'을 기준으로 테이블을 역추적하는 최종 크롤러
# ----------------------------------------------------
today = datetime.now().date()
all_jobs_details = []
scraped_jobs_count = 0

print("="*60)
print("마감되지 않은 공고의 상세 정보를 수집합니다...")
print("="*60)

page_num = 1
while True:
    print(f"--- {page_num} 페이지 수집 시작 ---")
    
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".list-employment tbody > tr")))
    job_rows = driver.find_elements(By.CSS_SELECTOR, ".list-employment tbody > tr")
    
    num_valid_jobs_on_page = 0
    for row in job_rows:
        try:
            deadline_str = row.find_element(By.CSS_SELECTOR, "td.td_deadline").text
            deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            if deadline_date >= today:
                num_valid_jobs_on_page += 1
        except (NoSuchElementException, ValueError):
            continue
            
    for i in range(num_valid_jobs_on_page):
        try:
            current_job_rows = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".list-employment tbody > tr")))
            valid_rows = [r for r in current_job_rows if r.find_elements(By.CSS_SELECTOR, "td.td_deadline")]
            job_to_click = valid_rows[i].find_element(By.CSS_SELECTOR, "td.td_subject a")
            
            scraped_jobs_count += 1
            print(f"[{scraped_jobs_count}] '{job_to_click.text}' 정보 수집 중...")
            
            job_to_click.click()
            
            detail_table = wait.until(EC.presence_of_element_located(
                (By.XPATH, "//*[contains(text(), '모집내용')]/ancestor::table[1]")
            ))
            
            rows = detail_table.find_elements(By.TAG_NAME, "tr")
            job_details = {}
            for row in rows:
                try:
                    header = row.find_element(By.TAG_NAME, "th").text
                    value = row.find_element(By.TAG_NAME, "td").text
                    job_details[header] = value.strip()
                except NoSuchElementException:
                    continue
            all_jobs_details.append(job_details)
            print(f"  -> 수집 완료!")

        except Exception as e:
            print(f"  -> 오류 발생: {type(e).__name__}. 다음 공고로 넘어갑니다.")
        
        finally:
            driver.back()
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".list-employment")))

    # --- 다음 페이지로 이동 ---
    try:
        next_button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "다음")))
        next_button.click()
        page_num += 1
    except (NoSuchElementException, TimeoutException):
        print("\n✅ 마지막 페이지입니다. 모든 공고 수집을 완료했습니다.")
        break

# -------------------------------
# 7️⃣ 수집된 데이터를 CSV 파일로 저장
# -------------------------------
print("\n🎉 모든 상세 정보 크롤링 완료!")
print(f"총 {len(all_jobs_details)}개의 공고 정보를 수집했습니다.")

if all_jobs_details:
    all_keys = set()
    for job in all_jobs_details:
        all_keys.update(job.keys())
    
    file_path = "donga_job_postings.csv"
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(all_keys))
        writer.writeheader()
        writer.writerows(all_jobs_details)
    print(f"✅ 데이터가 성공적으로 '{file_path}' 파일에 저장되었습니다.")

# -------------------------------
# 8️⃣ 종료
# -------------------------------
driver.quit()
print("🔚 브라우저 닫음.")