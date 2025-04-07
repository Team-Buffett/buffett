# AI 기반 암호화폐 선물 트레이딩 봇

이 프로젝트는 AI를 활용한 암호화폐 선물 트레이딩 봇으로, 멀티 타임프레임 분석, 워렌 버핏의 투자 철학, AI 기반 포지션 관리, 실시간 모니터링, 클라우드 배포를 목표로 개발되었습니다.

## 프로젝트 개요

코인 선물 거래를 자동화하고 AI로 최적화된 투자 결정을 내리는 봇 구현 `ai-autotrade`

### 목적

> 숨만 쉬어도 자동으로 돈이 되는 것을 만들자

### 인사이트

- 선물 투자는 위험하다
- AI가 몇년 전보다 많이 발전했다
- 이미 너무 많은 사람이 트레이딩봇을 고도화 해놨다. 전체 코인 거래 중 많은 비율이 봇일수도 있다.

### 스크린샷

<img width="715" alt="image" src="https://github.com/user-attachments/assets/fe5c4e49-b547-418b-a728-3c38e221dd90" />
<img width="734" alt="image" src="https://github.com/user-attachments/assets/f2e221fe-5a60-45f0-b623-2561c9c44aff" />
<img width="734" alt="image" src="https://github.com/user-attachments/assets/ac78f91d-b032-4eed-81fc-f4aa7fc58546" />

<img width="736" alt="image" src="https://github.com/user-attachments/assets/6edb1123-b3b9-4784-8eb9-bd83d7a4a73b" />
<img width="712" alt="image" src="https://github.com/user-attachments/assets/ae5c1a10-08c6-4383-9fba-e1d796fa0d00" />


### 특징

실시간 데이터 분석, 동적 손절/익절, 거래 기록 DB, Streamlit 대시보드, AWS 배포

### 주요 기능

- 멀티 타임프레임 분석: 15분, 1시간, 4시간 차트를 활용한 시장 분석
- AI 기반 의사결정: OpenAI API로 포지션 크기(켈리 공식), 레버리지, SL/TP를 동적 조절
- 거래 기록 및 학습: SQLite DB에 거래 내역 저장, 과거 데이터를 반영한 전략 개선
- 실시간 모니터링: Streamlit 대시보드로 현재 포지션 및 성과 시각화
- 백테스트: 과거 데이터를 활용한 전략 검증 및 자기반성
- 클라우드 배포: AWS에서 24/7 실행 가능

### 기술 스택

- 언어: Python
- 라이브러리: CCXT (Binance API), Pandas, OpenAI, SQLite, Streamlit
- 배포: AWS (EC2)
- 데이터: Binance OHLCV, 과거 거래 기록

### 성과 분석

- DB 기록: 모든 거래와 AI 분석이 SQLite에 저장
- 메트릭스: 승률, 평균 손익률, 최대 손실/이익 등 제공

### 백테스트

모델 성능 비교를 위한 과거 데이터를 활용하여 백테스트
`ai-crypto-backtest`

### 개선 계획

- 코인 제외 다른 투자 구현 (현물, 주식 등)
- 로컬 LLM 으로 실행 및 성능비교
- 파인튜닝 (Fine-tuning) 또는 로컬 LLM 학습

`2025.04.04 ~ 2025.04.05`
