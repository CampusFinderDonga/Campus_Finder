<!-- 로고/배너 삽입 예정 -->
<p align="center">
  <!-- 예: <img src="이미지경로" alt="Campus Finder Logo" width="300"/> -->
</p>

<h1 align="center">🎓 Campus Finder | 동아대학교 학사정보 통합 플랫폼</h1>

<p align="center">
  <b>동아대학교 재학생을 위한 RAG 기반 학사정보 통합 & 질의응답 웹 서비스</b><br>
  분산된 학사정보를 한눈에, 질문은 AI에게 쉽게.
</p>

---

## 📚 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 🎯 목적 | 동아대학교 학사정보(수강신청, 학점확인, 연구학점제 등)를 <br>웹에서 통합 조회 + RAG 기반 질의응답 제공 |
| 👥 대상 | 동아대학교 재학생 |
| 📆 개발 기간 | 2025년 2학기 (9월 ~ 12월) |
| 🛠 수행 방식 | 팀 프로젝트 |
| 💻 담당 역할 | 기획 및 전체 웹 개발 |
| 📦 GitHub | https://github.com/Seogaeun03/Campus_Finder.git |

---

## 🧠 프로젝트 목적 및 필요성

✅ 학내 학사정보가 통합정보서비스, 학과 홈페이지, PDF 문서 등에 **분산**  
✅ “휴학 신청 언제야?” 같은 단순 질문에도 **여러 페이지 탐색 필요**  
✅ 이를 해결하기 위해:

```bash
📌 정보 수집 자동화 + RAG 기반 Q&A
📌 학사 서비스 UI 대시보드화
📌 직관적 접근성과 검색성 제공
```

---

## 🏗 시스템 아키텍처

```
[사용자] → [Frontend (HTML/CSS/JS)] → [Django Backend]
           → [Upstage API (RAG + Document Parse)]
           → [학교 웹페이지 / PDF 데이터]
```

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🧠 RAG 기반 챗봇 | Upstage API 기반 학사문서 질의응답 |
| 📊 통합 대시보드 | 핵심 학사메뉴 UI 템플릿 제공 |
| 📅 학사일정 캘린더 | JSON 기반 일정 + 사용자 커스텀 일정 지원 |
| 🔐 로그인 기능 (선택) | Django Auth 기반 즐겨찾기 및 일정 저장 |
| 🧾 데이터 수집 자동화 | Selenium + BeautifulSoup + Upstage Parse 사용 |

---

## ⚙ 기술 스택

| 분류 | 기술 |
|------|------|
| 🎛 Backend | Django, Python |
| 🎨 Frontend | HTML, CSS, JS, Bootstrap |
| 🗃 DB | SQLite(개발) / PostgreSQL(예정) |
| 🤖 AI/RAG | Upstage Solar RAG, Document Parse |
| 📄 데이터 처리 | JSON, CSV, Excel, PDF |
| 🕸 크롤링 | Selenium, BeautifulSoup |
| ☁ 배포 예정 | AWS / Docker |

---

## 💻 설치 및 실행 방법 (Django 기준)

```bash
# 1. 저장소 클론
git clone https://github.com/CampusFinderDonga/Campus_Finder.git
cd Campus_Finder

# 2. 가상환경 생성 및 활성화 (예: venv 사용)
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 마이그레이션 적용
python manage.py migrate

# 5. 서버 실행
python manage.py runserver
```

🔗 기본 접속: `http://127.0.0.1:8000/`

---

## 🖼 페이지 / 화면 미리보기 (추후 삽입)

| 화면 | 설명 |
|------|------|
| 📍 메인 대시보드 | 이미지 예정 |
| 📍 챗봇 UI | 이미지 예정 |
| 📍 일정 캘린더 | 이미지 예정 |

---

## 📊 GitHub 활동 통계

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Seogaeun03&show_icons=true&theme=dracula" width="48%">
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=Seogaeun03&layout=compact&theme=dracula" width="40%">
</p>
