import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ccxt
import numpy as np
import os
import time

# 현재 기본 코인명 (파일 기준)
with open("txt/coinName.txt", "r", encoding="utf-8") as f:
    default_coin = f.read().strip().replace(" ", "")

# Streamlit 페이지 설정
st.set_page_config(
    page_title=f"Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Coin 리스트 가져오기
@st.cache_data(ttl=5)
def get_available_coin_names():
    db_files = [f for f in os.listdir("db") if f.endswith("_trading.db")]
    return [f.replace("_trading.db", "") for f in db_files]

def get_db_connection(coin_name: str):
    conn = sqlite3.connect(f"db/{coin_name}_trading.db", timeout=10)
    conn.execute("PRAGMA query_only=ON;")
    return conn

# --- 사이드바 구성 ---
st.sidebar.title("⚙️ 대시보드 설정")
auto_refresh = st.sidebar.checkbox("자동 새로고침 (5초)", value=True)
refresh_interval_sec = 5

available_coins = get_available_coin_names()
coin_options = ["ALL (XRP+ETH)"] + available_coins
default_idx = coin_options.index(default_coin) if default_coin in coin_options else 0
selected_coin = st.sidebar.selectbox("코인 선택:", coin_options, index=default_idx)

# 선택된 코인명 전역 설정
_coinName = selected_coin

# --- 스타일 ---
st.markdown("""
<style>
    .header { font-size: 2.5rem; color: #FF9900; text-align: center; margin-bottom: 1.5rem; }
    .metrics-container { display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between; margin-bottom: 2rem; }
    .metric-card { background-color: #262730; border-radius: 5px; padding: 1rem; text-align: center; width: calc(25% - 10px); box-sizing: border-box; }
    .metric-title { font-size: 1rem; color: #888888; margin-bottom: 0.5rem; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #FFFFFF; }
    .positive { color: #00CC96; }
    .negative { color: #EF553B; }
    .neutral { color: #FFD700; }
    .subheader { font-size: 1.5rem; color: #FF9900; margin-top: 2rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# 데이터 로딩 함수들
def get_trades_data(coin_name):
    conn = get_db_connection(coin_name)
    query = """
    SELECT id, timestamp, action, entry_price, exit_price, amount, leverage,
           status, profit_loss, profit_loss_percentage, exit_timestamp
    FROM trades
    ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'exit_timestamp' in df.columns:
        df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
    return df

def get_ai_analysis_data(coin_name):
    conn = get_db_connection(coin_name)
    query = """
    SELECT id, timestamp, current_price, direction,
           recommended_leverage, reasoning, trade_id
    FROM ai_analysis
    ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data(ttl=300)
def get_Coin_price_data(coin_name, timeframe='1d', limit=90):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(f'{coin_name}/USDT', timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 트레이딩 성과 지표 계산 함수
def calculate_trading_metrics(trades_df, Coin_price_df=None, time_filter=None, filter_time=None):
    if trades_df.empty:
        return {
            'total_return': 0,
            'market_return': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'avg_profit_loss': 0,
            'avg_holding_time': 0
        }

    # 종료된 거래만 필터링
    closed_trades = trades_df[trades_df['status'] == 'CLOSED']
    if closed_trades.empty:
        return {
            'total_return': 0,
            'market_return': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'avg_profit_loss': 0,
            'avg_holding_time': 0
        }

    # 총 수익률 (초기 투자 금액은 추정)
    total_profit_loss = closed_trades['profit_loss'].sum()
    # 초기 투자 금액 추정
    initial_investment = closed_trades.sort_values('timestamp').head(3)['entry_price'].mean() * \
                         closed_trades.sort_values('timestamp').head(3)['amount'].mean()
    if initial_investment < 100:  # 너무 작은 경우 합리적인 값으로 설정
        initial_investment = 10000
    total_return = (total_profit_loss / initial_investment) * 100

    # 시장 수익률 계산 (Buy & Hold 전략)
    market_return = 0
    if Coin_price_df is not None and not Coin_price_df.empty:
        if time_filter != "전체" and filter_time is not None:
            # 필터링된 기간에 해당하는 Coin 가격 데이터
            filtered_Coin = Coin_price_df[Coin_price_df['timestamp'] >= filter_time]
            if not filtered_Coin.empty:
                start_price = filtered_Coin.iloc[0]['close']
                end_price = filtered_Coin.iloc[-1]['close']
                market_return = ((end_price - start_price) / start_price) * 100
        else:
            # 거래 기간에 맞춰 Coin 가격 데이터
            first_trade_time = closed_trades.sort_values('timestamp').iloc[0]['timestamp']
            last_trade_time = closed_trades.sort_values('timestamp').iloc[-1]['timestamp' if 'exit_timestamp' not in closed_trades.columns else 'exit_timestamp']

            relevant_Coin = Coin_price_df[
                (Coin_price_df['timestamp'] >= first_trade_time) &
                (Coin_price_df['timestamp'] <= last_trade_time)
                ]

            if not relevant_Coin.empty:
                start_price = relevant_Coin.iloc[0]['close']
                end_price = relevant_Coin.iloc[-1]['close']
                market_return = ((end_price - start_price) / start_price) * 100

    # 승률
    winning_trades = len(closed_trades[closed_trades['profit_loss'] > 0])
    total_trades = len(closed_trades)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    # 손익비 (Profit Factor)
    total_profit = closed_trades[closed_trades['profit_loss'] > 0]['profit_loss'].sum()
    total_loss = abs(closed_trades[closed_trades['profit_loss'] < 0]['profit_loss'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else 0

    # 최대 낙폭 (Maximum Drawdown) - 계좌 잔고 기준으로 변경
    closed_trades_sorted = closed_trades.sort_values('timestamp')

    # 계좌 잔고 추적 - 초기 계좌 잔고를 initial_investment로 설정
    account_balance = initial_investment
    balances = []

    # 각 거래 후의 계좌 잔고 계산
    for _, trade in closed_trades_sorted.iterrows():
        account_balance += trade['profit_loss']  # 이익/손실 반영
        balances.append(account_balance)

    if balances:
        # NumPy 배열로 변환
        balances = np.array(balances)
        # 각 시점까지의 최고 잔고
        peak_balances = np.maximum.accumulate(balances)
        # 각 시점의 드로다운 계산 (최고 잔고 대비 현재 잔고의 하락률)
        drawdowns = (peak_balances - balances) / peak_balances
        # 최대 드로다운
        max_drawdown = np.max(drawdowns) * 100 if drawdowns.size > 0 else 0
    else:
        max_drawdown = 0

    # 샤프 비율 (일일 수익률 기반)
    if 'profit_loss_percentage' in closed_trades.columns and len(closed_trades) > 1:
        returns = closed_trades['profit_loss_percentage'] / 100  # 비율로 변환
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0

    # 평균 손익
    avg_profit_loss = closed_trades['profit_loss'].mean()

    # 평균 보유 시간
    if 'exit_timestamp' in closed_trades.columns and 'timestamp' in closed_trades.columns:
        # timestamp와 exit_timestamp가 모두 datetime인지 확인
        valid_timestamps = closed_trades.dropna(subset=['exit_timestamp'])
        if not valid_timestamps.empty:
            holding_times = (valid_timestamps['exit_timestamp'] - valid_timestamps['timestamp']).dt.total_seconds() / 3600  # 시간 단위
            avg_holding_time = holding_times.mean()
        else:
            avg_holding_time = 0
    else:
        avg_holding_time = 0

    return {
        'total_return': total_return,
        'market_return': market_return,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'avg_profit_loss': avg_profit_loss,
        'avg_holding_time': avg_holding_time
    }

try:
    if selected_coin == "ALL (XRP+ETH)":
        st.markdown("<h1 class='header'>XRP + ETH Trading Dashboard</h1>", unsafe_allow_html=True)
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")
        cols = st.columns(2)
        for i, coin in enumerate(["XRP", "ETH"]):
            with cols[i]:
                if coin.lower() not in [c.lower() for c in available_coins]:
                    st.warning(f"{coin} DB가 없습니다.")
                    continue
                c_trades = get_trades_data(coin)
                c_ai = get_ai_analysis_data(coin)
                c_price = get_Coin_price_data(coin)
                c_metrics = calculate_trading_metrics(c_trades, c_price, None, None)
                c_open = c_trades[c_trades['status'] == 'OPEN']
                c_has_open = len(c_open) > 0
                c_pos = c_open.iloc[0] if c_has_open else None
                c_now_price = c_ai.iloc[0]['current_price'] if not c_ai.empty else c_price.iloc[-1]['close']

                st.markdown(f"<h2 class='subheader'>{coin}</h2>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Return", f"{c_metrics['total_return']:.2f}%")
                m2.metric("Win Rate", f"{c_metrics['win_rate']:.1f}%")
                m3.metric("Total Trades", f"{c_metrics['total_trades']}")
                m4, m5, m6 = st.columns(3)
                m4.metric("Current Price", f"${c_now_price:,.4f}")
                m5.metric("Position", c_pos['action'].upper() if c_has_open else "NO_POSITION")
                latest_dir = c_ai.iloc[0]['direction'] if not c_ai.empty else "N/A"
                m6.metric("AI Direction", latest_dir)

                show_df = c_trades[['timestamp', 'action', 'entry_price', 'status', 'profit_loss']].head(8).copy()
                show_df['timestamp'] = show_df['timestamp'].dt.strftime('%m-%d %H:%M:%S')
                st.dataframe(show_df, use_container_width=True, height=260)
        st.stop()

    # 데이터 로드
    trades_df = get_trades_data(_coinName)
    ai_analysis_df = get_ai_analysis_data(_coinName)
    Coin_price_df = get_Coin_price_data(_coinName)
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")

    st.sidebar.title(f"Currently invested coins are: {default_coin}")
    # 시간 필터
    st.sidebar.title(f"{_coinName} Trading Bot")
    time_filter = st.sidebar.selectbox(
        "기간 선택:",
        ["전체", "최근 24시간", "최근 7일", "최근 30일", "최근 90일"]
    )

    # 시간 필터 적용
    now = datetime.now()
    if time_filter == "최근 24시간":
        filter_time = now - timedelta(days=1)
        filtered_trades = trades_df[trades_df['timestamp'] > filter_time]
        chart_days = 1
    elif time_filter == "최근 7일":
        filter_time = now - timedelta(days=7)
        filtered_trades = trades_df[trades_df['timestamp'] > filter_time]
        chart_days = 7
    elif time_filter == "최근 30일":
        filter_time = now - timedelta(days=30)
        filtered_trades = trades_df[trades_df['timestamp'] > filter_time]
        chart_days = 30
    elif time_filter == "최근 90일":
        filter_time = now - timedelta(days=90)
        filtered_trades = trades_df[trades_df['timestamp'] > filter_time]
        chart_days = 90
    else:
        filtered_trades = trades_df
        filter_time = None
        chart_days = 90

    # 트레이딩 지표 계산
    metrics = calculate_trading_metrics(filtered_trades, Coin_price_df, time_filter, filter_time)

    # 현재 오픈 포지션
    open_trades = trades_df[trades_df['status'] == 'OPEN']
    has_open_position = len(open_trades) > 0
    current_position = open_trades.iloc[0] if has_open_position else None

    # 현재 Coin 가격
    current_Coin_price = ai_analysis_df.iloc[0]['current_price'] if not ai_analysis_df.empty else Coin_price_df.iloc[-1]['close']

    # 대시보드 메인
    st.markdown(f"<h1 class='header'>{_coinName} Trading Dashboard</h1>", unsafe_allow_html=True)

    # 주요 트레이딩 지표 표시
    st.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-title">Total Return</div>
            <div class="metric-value {'positive' if metrics['total_return'] > 0 else 'negative' if metrics['total_return'] < 0 else ''}">
                {metrics['total_return']:.2f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Market Return</div>
            <div class="metric-value {'positive' if metrics['market_return'] > 0 else 'negative' if metrics['market_return'] < 0 else ''}">
                {metrics['market_return']:.2f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Sharpe Ratio</div>
            <div class="metric-value {'positive' if metrics['sharpe_ratio'] > 1 else 'negative' if metrics['sharpe_ratio'] < 0 else ''}">
                {metrics['sharpe_ratio']:.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Win Rate</div>
            <div class="metric-value {'positive' if metrics['win_rate'] >= 50 else 'negative' if metrics['win_rate'] < 40 else ''}">
                {metrics['win_rate']:.1f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Profit Factor</div>
            <div class="metric-value {'positive' if metrics['profit_factor'] > 1 else 'negative' if metrics['profit_factor'] < 1 else ''}">
                {metrics['profit_factor']:.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Max Drawdown</div>
            <div class="metric-value {'negative' if metrics['max_drawdown'] > 0 else ''}">
                {metrics['max_drawdown']:.2f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Total Trades</div>
            <div class="metric-value">
                {metrics['total_trades']}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Avg P/L</div>
            <div class="metric-value {'positive' if metrics['avg_profit_loss'] > 0 else 'negative' if metrics['avg_profit_loss'] < 0 else ''}">
                {metrics['avg_profit_loss']:.2f} USDT
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Avg Holding</div>
            <div class="metric-value">
                {metrics['avg_holding_time']:.1f}h
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 현재 Coin 가격 및 포지션 정보
    position_cols = st.columns(2)

    with position_cols[0]:
        st.markdown(f"""
        <div class="metric-card" style="width: 100%">
            <div class="metric-title">Current {_coinName} Price</div>
            <div class="metric-value">${current_Coin_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with position_cols[1]:
        position_status = "No Position" if not has_open_position else f"{current_position['action'].upper()}"
        position_color = "" if not has_open_position else ("positive" if current_position['action'] == 'long' else "negative")
        st.markdown(f"""
        <div class="metric-card" style="width: 100%">
            <div class="metric-title">Current Position</div>
            <div class="metric-value {position_color}">{position_status}</div>
        </div>
        """, unsafe_allow_html=True)

    # Coin 가격 차트와 거래 시점 표시
    st.markdown(f"<h2 class='subheader'>{_coinName} Price Chart & Trade Entries</h2>", unsafe_allow_html=True)

    # Coin 차트 기간 필터링
    filtered_price_df = Coin_price_df[Coin_price_df['timestamp'] > (now - timedelta(days=chart_days))]

    # 비트코인 차트 + 거래 시점 차트 생성
    fig = go.Figure()

    # Coin 가격 라인
    fig.add_trace(go.Scatter(
        x=filtered_price_df['timestamp'],
        y=filtered_price_df['close'],
        mode='lines',
        name=f'{_coinName} Price',
        line=dict(color='gray', width=2),
        hovertemplate='<b>Price</b>: $%{y:,.2f}<br>'
    ))

    # 롱(매수) 포인트
    long_points = filtered_trades[filtered_trades['action'] == 'long']
    if not long_points.empty:
        fig.add_trace(go.Scatter(
            x=long_points['timestamp'],
            y=long_points['entry_price'],
            mode='markers',
            name='Long Entry',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            hovertemplate='<b>Long Entry</b><br>' +
                          'Price: $%{y:,.2f}<br>' +
                          'Date: %{x}<br>' +
                          '<extra></extra>'
        ))

    # 숏(매도) 포인트
    short_points = filtered_trades[filtered_trades['action'] == 'short']
    if not short_points.empty:
        fig.add_trace(go.Scatter(
            x=short_points['timestamp'],
            y=short_points['entry_price'],
            mode='markers',
            name='Short Entry',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            hovertemplate='<b>Short Entry</b><br>' +
                          'Price: $%{y:,.2f}<br>' +
                          'Date: %{x}<br>' +
                          '<extra></extra>'
        ))

    # 청산 포인트
    exit_points = filtered_trades[(filtered_trades['status'] == 'CLOSED') & (filtered_trades['exit_price'].notna())]
    if not exit_points.empty:
        fig.add_trace(go.Scatter(
            x=exit_points['exit_timestamp'] if 'exit_timestamp' in exit_points.columns else exit_points['timestamp'],
            y=exit_points['exit_price'],
            mode='markers',
            name='Exit',
            marker=dict(color='yellow', size=8, symbol='circle'),
            hovertemplate='<b>Exit</b><br>' +
                          'Price: $%{y:,.2f}<br>' +
                          'Date: %{x}<br>' +
                          '<extra></extra>'
        ))

    # 차트 레이아웃 설정
    fig.update_layout(
        title=f'{_coinName} Price & Trading Points',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # 거래 성과 차트
    st.markdown("<h2 class='subheader'>Trading Performance</h2>", unsafe_allow_html=True)
    chart_cols = st.columns(2)

    with chart_cols[0]:
        closed_trades = filtered_trades[filtered_trades['status'] == 'CLOSED']
        if not closed_trades.empty:
            # 누적 수익 차트
            trades_sorted = closed_trades.sort_values('timestamp')
            trades_sorted['cumulative_pl'] = trades_sorted['profit_loss'].cumsum()

            fig = px.line(
                trades_sorted,
                x='timestamp',
                y='cumulative_pl',
                title='Cumulative Profit/Loss',
                labels={'timestamp': 'Date', 'cumulative_pl': 'P/L (USDT)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades to display.")

    with chart_cols[1]:
        total_trades = len(closed_trades)
        if total_trades > 0:
            # 거래 결정 분포
            decisions = filtered_trades['action'].value_counts().reset_index()
            decisions.columns = ['Direction', 'Count']

            fig = px.pie(
                decisions,
                values='Count',
                names='Direction',
                title='Trade Direction Distribution',
                color='Direction',
                color_discrete_map={'long': '#00CC96', 'short': '#EF553B'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades to display.")

    # 거래 내역
    st.markdown("<h2 class='subheader'>Recent Trades</h2>", unsafe_allow_html=True)
    if not filtered_trades.empty:
        # 표시용 데이터 준비
        display_df = filtered_trades[['id', 'timestamp', 'action', 'entry_price', 'exit_price', 'status', 'profit_loss']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.rename(columns={
            'id': 'ID',
            'timestamp': 'Date',
            'action': 'Direction',
            'entry_price': 'Entry Price',
            'exit_price': 'Exit Price',
            'status': 'Status',
            'profit_loss': 'P/L'
        })

        # 데이터프레임 표시
        st.dataframe(
            display_df,
            height=400,
            use_container_width=True
        )
    else:
        st.info("No trades in the selected time period.")


    # 오픈 포지션 정보
    if has_open_position:
        st.markdown("<h2 class='subheader'>Current Open Position</h2>", unsafe_allow_html=True)

        position_cols = st.columns(2)
        with position_cols[0]:
            entry_time = current_position['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(f"""
            ### Position Details
            - **Direction**: {current_position['action'].upper()}
            - **Entry Time**: {entry_time}
            - **Entry Price**: ${current_position['entry_price']:,.2f}
            - **Leverage**: {current_position['leverage']}x
            - **Amount**: {current_position['amount']} {_coinName}
            """)

        with position_cols[1]:
            # 현재가와 진입가 비교 차트
            if isinstance(current_Coin_price, (int, float)):
                price_diff = current_Coin_price - current_position['entry_price']
                price_diff_pct = (price_diff / current_position['entry_price']) * 100
                price_color = "green" if (current_position['action'] == 'long' and price_diff > 0) or (current_position['action'] == 'short' and price_diff < 0) else "red"

                st.markdown(f"""
                ### Current Performance
                - **Current Price**: ${current_Coin_price:,.2f}
                - **Price Change**: ${price_diff:,.2f} ({price_diff_pct:.2f}%)
                - **Estimated P/L**: <span style='color:{price_color};'>${price_diff * current_position['amount'] * current_position['leverage']:,.2f}</span>
                """, unsafe_allow_html=True)

    # AI 분석 섹션
    st.markdown("<h2 class='subheader'>Latest AI Analysis</h2>", unsafe_allow_html=True)
    if not ai_analysis_df.empty:
        latest_analysis = ai_analysis_df.iloc[0]

        analysis_cols = st.columns(2)
        with analysis_cols[0]:
            direction_color = "green" if latest_analysis['direction'] == 'LONG' else "red" if latest_analysis['direction'] == 'SHORT' else "orange"
            st.markdown(f"""
            ### AI Analysis Summary
            - **Time**: {latest_analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            - **Direction Recommendation**: <span style='color:{direction_color};'>{latest_analysis['direction']}</span>
            - **Recommended Leverage**: {latest_analysis['recommended_leverage']}x
            """, unsafe_allow_html=True)

        with analysis_cols[1]:
            st.markdown("### Analysis Reasoning")
            # 분석 내용 일부만 표시
            reasoning_preview = latest_analysis['reasoning'][:200] + "..." if len(latest_analysis['reasoning']) > 200 else latest_analysis['reasoning']
            st.write(reasoning_preview)

            if st.button("View Full Analysis"):
                st.write(latest_analysis['reasoning'])
    else:
        st.info("No AI analysis data available.")


    # 추가 섹션: 거래 상세 정보 (Trade Details)
    st.markdown("<h2 class='subheader'>Trade Details</h2>", unsafe_allow_html=True)

    # filtered_trades 데이터프레임에서 거래 ID 목록을 selectbox 옵션으로 사용합니다.
    if not filtered_trades.empty:
        trade_ids = filtered_trades['id'].unique()
        selected_trade_id = st.selectbox("Select Trade ID", options=trade_ids, format_func=lambda x: f"Trade {x}")

        # 선택한 거래 id에 해당하는 상세 정보 가져오기 (첫번째 행 사용)
        selected_trade = filtered_trades[filtered_trades['id'] == selected_trade_id].iloc[0]

        # 거래 기본 정보 표시 (날짜, 방향, 진입가, 청산가 등)
        st.markdown("#### Trade Information")
        trade_info = f"""
        - **Trade ID**: {selected_trade['id']}
        - **Date**: {selected_trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        - **Direction**: {selected_trade['action'].upper()}
        - **Entry Price**: ${selected_trade['entry_price']:,.2f}
        - **Exit Price**: {"N/A" if pd.isna(selected_trade['exit_price']) else f"${selected_trade['exit_price']:,.2f}"}
        - **Amount**: {selected_trade['amount']} {_coinName}
        - **Leverage**: {selected_trade['leverage']}x
        - **Status**: {selected_trade['status']}
        - **Profit/Loss**: ${selected_trade['profit_loss']:,.2f}
        """
        st.markdown(trade_info)

        # AI 분석 reasoning 정보 가져오기 (ai_analysis의 trade_id와 매칭)
        trade_analysis = ai_analysis_df[ai_analysis_df['trade_id'] == selected_trade_id]
        if not trade_analysis.empty:
            st.markdown("#### AI Analysis Reasoning")
            # 여러 건인 경우 첫 번째 건을 표시합니다.
            analysis_reasoning = trade_analysis.iloc[0]['reasoning']
            st.write(analysis_reasoning)
        else:
            st.info("No AI analysis data available for this trade.")
    else:
        st.info("No trades available to select.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()

if auto_refresh:
    time.sleep(refresh_interval_sec)
    st.rerun()
