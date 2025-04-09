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
- 프롬포트에서 언급하는 것들을 AI가 얼마나 중요하게 보는지 설정하는 것이 중요하다. 의도와 동일하게 동작하지 않을 확률이 높다.
- 종목에 따라 여러 데이터를 조절해줘야 한다.

### 스크린샷

<img width="730" alt="image" src="https://github.com/user-attachments/assets/0e769756-10ae-4162-b170-461ff1a17553" />
<img width="735" alt="image" src="https://github.com/user-attachments/assets/8e25de2b-e9f2-4c99-8f7c-fbadd8779a4f" />
<img width="732" alt="image" src="https://github.com/user-attachments/assets/3754bcb1-e36e-4c7f-bc5b-ab1a2670e30e" />
<img width="736" alt="image" src="https://github.com/user-attachments/assets/6f7f6f43-0d9b-4c23-b63f-8541c0733e73" />
<img width="737" alt="image" src="https://github.com/user-attachments/assets/54064b90-5989-4300-8e66-b491d22b3b7c" />
<img width="737" alt="image" src="https://github.com/user-attachments/assets/8c1b1d85-c57b-4027-ba4c-52371eaf8895" />

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
