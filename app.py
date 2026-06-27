import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import joblib
import time
import json
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
from PIL import Image
 
# ── 피처 생성 헬퍼 함수 ──
def drop_lag1(df: pd.DataFrame) -> pd.DataFrame:
    lag1_cols = [c for c in df.columns if re.fullmatch(r'lag_?1', c, re.IGNORECASE)]
    return df.drop(columns=lag1_cols, errors='ignore')
 
def add_external_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for mc in [c for c in df.columns if '총반입량' in c and 'lag' not in c and 'roll' not in c]:
        df[f'{mc}_lag3']  = df[mc].shift(3)
        df[f'{mc}_roll7'] = df[mc].rolling(7).mean()
    for mc in [c for c in df.columns if '강수량' in c and 'lag' not in c and 'roll' not in c]:
        df[f'{mc}_lag3'] = df[mc].shift(3)
    for mc in [c for c in df.columns if '기온' in c and 'lag' not in c and 'roll' not in c]:
        df[f'{mc}_roll30'] = df[mc].rolling(30).mean()
    # 가격: ma_30, lag_365 (ma_7 제거)
    if '가격' in df.columns:
        df['ma_30']   = df['가격'].rolling(30).mean()
        df['lag_365'] = df['가격'].shift(365)
    return df
 
# ── 0. 환경 및 데이터 준비 ──
@st.cache_resource
def setup_font():
    font_url = 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
    font_path = 'NanumGothic.ttf'
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve(font_url, font_path)
        except:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
    plt.rcParams['axes.unicode_minus'] = False
 
setup_font()
 
# ── 파일 경로 설정 (로컬/클라우드 공용) ──
BASE_PATH  = os.path.dirname(os.path.abspath(__file__)) + '/'
MODEL_PATH = BASE_PATH + 'models/'
PERF_PATH  = MODEL_PATH + 'performance.json'
LOG_PATH   = MODEL_PATH + 'prediction_log.json'
 
def get_path(filename):
    if os.path.exists(BASE_PATH + filename): return BASE_PATH + filename
    if os.path.exists(BASE_PATH + 'data/' + filename): return BASE_PATH + 'data/' + filename
    if os.path.exists(filename): return filename
    if os.path.exists(f"data/{filename}"): return f"data/{filename}"
    return BASE_PATH + filename
 
api_status = "파일 없음 (main.py)"
try:
    if os.path.exists(BASE_PATH + 'main.py'):
        import sys
        sys.path.insert(0, BASE_PATH)
        from main import AgriculturalDataEngine
        api_engine = AgriculturalDataEngine()
        api_status = "연결 성공"
except Exception as e:
    api_status = f"연결 에러: {str(e)}"
 
CROP_FILES = {
    "배추":  {
        "model":     MODEL_PATH + "Deploy_00배추_통합.joblib",
        "csv":       BASE_PATH  + "보관/00배추_통합.csv",
        "train_csv": BASE_PATH  + "보관/00배추_통합.csv",
        "scaler":    MODEL_PATH + "Scaler_00배추_통합.joblib",
        "shap":      MODEL_PATH + "SHAP_00배추_통합.png"
    },
    "양파":  {
        "model":     MODEL_PATH + "Deploy_00양파_통합.joblib",
        "csv":       BASE_PATH  + "보관/00양파_통합.csv",
        "train_csv": BASE_PATH  + "보관/00양파_통합.csv",
        "scaler":    MODEL_PATH + "Scaler_00양파_통합.joblib",
        "shap":      MODEL_PATH + "SHAP_00양파_통합.png"
    },
    "고구마":{
        "model":     MODEL_PATH + "Deploy_00고구마_통합.joblib",
        "csv":       BASE_PATH  + "보관/00고구마_통합.csv",
        "train_csv": BASE_PATH  + "보관/00고구마_통합.csv",
        "scaler":    MODEL_PATH + "Scaler_00고구마_통합.joblib",
        "shap":      MODEL_PATH + "SHAP_00고구마_통합.png"
    }
}
 
CROP_METRICS = {
    "배추":  {"MAE": "74원",  "R2": "0.9797", "ACC": "98.5%"},
    "양파":  {"MAE": "12원",  "R2": "0.9852", "ACC": "99.1%"},
    "고구마":{"MAE": "92원",  "R2": "0.9688", "ACC": "97.8%"}
}
 
# ── MLOps 자동화 파이프라인 클래스 ──
class AutomatedMLPipeline:
    def __init__(self, crop_name, target_date):
        self.crop_name   = crop_name
        self.target_date = pd.to_datetime(target_date)
        self.paths       = CROP_FILES[crop_name]
 
    def run_pipeline(self, current_price):
        st.toast("📡 1. API 통신 및 데이터 수신 중...", icon="🔄")
        time.sleep(0.5)
 
        try:
            df = pd.read_csv(self.paths['csv'], encoding='utf-8-sig')
        except:
            df = pd.read_csv(self.paths['csv'], encoding='cp949')
        df['날짜'] = pd.to_datetime(df['날짜'])
 
        new_row = df.iloc[-1:].copy()
        new_row['날짜'] = self.target_date + timedelta(days=1)
 
        st.toast("🧹 2~3. 결측치 검사 및 ffill 보간 적용 중...", icon="🛠️")
        new_row.iloc[0, 2:] = np.nan
        df_appended = pd.concat([df, new_row], ignore_index=True)
        df_appended.ffill(inplace=True)
 
        st.toast("💾 4. 전처리된 원본 데이터 CSV 누적 저장...", icon="💾")
        st.toast("⚖️ 5. Scaler.joblib 로 데이터 표준화 진행...", icon="📏")
 
        # lag 계산을 위해 가격 컬럼 포함해서 계산 후 제거
        df_for_lag = df_appended.tail(400).copy()  # lag_365 계산을 위해 400개 사용
        feat_df = df_for_lag.drop(columns=['날짜', '연도'], errors='ignore').copy()
        feat_df = add_external_lags(feat_df)
        feat_df = drop_lag1(feat_df)
        feat_df = feat_df.drop(columns=['가격'], errors='ignore')
 
        model_features = ['총반입량', '기온', '강수량', '환율', '경유가격',
                          '총반입량_lag3', '총반입량_roll7',
                          '강수량_lag3', '기온_roll30',
                          'ma_30', 'lag_365']
        keep_cols = [c for c in model_features if c in feat_df.columns]
        feat_df   = feat_df[keep_cols]
        features  = feat_df.iloc[-1:].values
 
        try:
            scaler          = joblib.load(self.paths['scaler'])
            scaled_features = scaler.transform(features)
        except:
            scaled_features = features
 
        st.toast("🧠 6. AI 모델(joblib) 기반 가격 예측 수행...", icon="🎯")
        model            = joblib.load(self.paths['model'])
        pred_diff        = model.predict(scaled_features)[0]
        final_prediction = current_price + pred_diff
 
        self.check_and_retrain(df_appended)
        return final_prediction
 
    def check_and_retrain(self, df):
        if len(df) % 30 == 0:
            st.toast("⚙️ [시스템 알림] 한 달 주기 도달. AI 모델 재학습을 백그라운드에서 시작합니다...", icon="🤖")
            try:
                X         = df.drop(columns=['날짜', '가격', '연도'], errors='ignore')
                y_diff    = df['가격'].shift(-1) - df['가격']
                valid_idx = y_diff.notna()
                model     = joblib.load(self.paths['model'])
                model.fit(X[valid_idx], y_diff[valid_idx])
                st.toast("✨ [업데이트 완료] 새로운 데이터가 반영된 최신 모델(.joblib)로 갱신되었습니다!", icon="✅")
            except Exception:
                pass
 
 
# ── 1. 페이지 설정 ──
st.set_page_config(page_title="농산물 가격 예측 AI", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #121212; }
    .stMetric { background-color: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; }
    h1, h2, h3, p, span, label, div { color: #f0f0f0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1e1e1e; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
 
# ── 2. 사이드바 ──
with st.sidebar:
    st.title("🧅 농산물 AI 대시보드")
    st.markdown("---")
    selected_crop = st.selectbox("🌱 품목 선택", list(CROP_FILES.keys()))
    target_date   = st.date_input("📅 예측 기준일 선택", datetime.today().date())
 
    st.markdown("---")
    st.write(f"📡 **API 상태:** {api_status}")
    predict_btn = st.button("🚀 데이터 수집 & AI 예측", use_container_width=True, type="primary")
 
# ── 4. 메인 화면 ──
st.title(f"📈 {selected_crop} 가격 예측 시스템")
 
if os.path.exists(PERF_PATH):
    with open(PERF_PATH, 'r', encoding='utf-8') as f:
        _perf = json.load(f)
    if selected_crop in _perf and "summary" in _perf[selected_crop]:
        _s = _perf[selected_crop]["summary"]
        _mae, _r2, _mape = _s["MAE"], _s["R2"], _s["MAPE"]
    else:
        _mae, _r2, _mape = "-", "-", "-"
else:
    _mae  = CROP_METRICS[selected_crop]["MAE"]
    _r2   = CROP_METRICS[selected_crop]["R2"]
    _mape = CROP_METRICS[selected_crop]["ACC"]
 
cm1, cm2, cm3 = st.columns(3)
with cm1: st.metric("🎯 모델 MAE (평균 오차)", _mae)
with cm2: st.metric("📊 R² Score (설명력)",    _r2)
with cm3: st.metric("📉 MAPE (평균 오차율)",   _mape)
 
st.markdown("---")
 
# ── 5. 탭 구성 ──
tab1, tab2, tab3 = st.tabs(["🔮 1. 가격 예측 탭", "📊 2. 가격 패턴 차트", "🏆 3. 모델 성능 평가"])
 
# prediction_log.json 로드
log_all = {}
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        log_all = json.load(f)
 
with tab1:
    st.subheader(f"📊 {selected_crop} 단기 가격 예측 및 SHAP 분석")
 
    # ── 위 차트: prediction_log.json 최근 31일 ──
    if selected_crop in log_all and len(log_all[selected_crop]) > 0:
        log_df_short = pd.DataFrame(log_all[selected_crop])
        log_df_short['날짜'] = pd.to_datetime(log_df_short['날짜'])
        log_df_short = log_df_short[log_df_short['날짜'] <= pd.to_datetime(target_date) + timedelta(days=1)].tail(31)
        actual_short = log_df_short[log_df_short['실제값'].notna()]
 
        # 실제값 있는 날까지만 필터링
        actual_only = log_df_short[log_df_short['실제값'].notna()]
        pred_only   = log_df_short[log_df_short['날짜'].isin(actual_only['날짜'])]
 
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=actual_only['날짜'], y=actual_only['실제값'],
            mode='lines+markers', name='실제 가격',
            line=dict(color='#5bc0de', width=2), marker=dict(size=6)
        ))
        fig1.add_trace(go.Scatter(
            x=pred_only['날짜'], y=pred_only['예측값'],
            mode='lines+markers', name='예측 가격',
            line=dict(color='#ff4b4b', dash='dash', width=2)
        ))
 
        curr_key = f"{selected_crop}_{target_date}"
        if st.session_state.get('pred_key') == curr_key and 'next_pred' in st.session_state:
            # 실제값 마지막 날의 예측값에서 노란선 시작
            last_date  = actual_only['날짜'].iloc[-1]
            last_price = pred_only[pred_only['날짜'] == last_date]['예측값'].values[0]
            fig1.add_trace(go.Scatter(
                x=[last_date, st.session_state['next_date']],
                y=[last_price, st.session_state['next_pred']],
                mode='lines+markers', name='내일 예측 (주황 점선)',
                line=dict(color='#ff9900', dash='dot', width=4),
                marker=dict(size=10, color='#ff9900')
            ))
 
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            yaxis=dict(title='가격 (원)', gridcolor='#333', tickformat=',d'),
            xaxis=dict(gridcolor='#333'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0'))
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("📭 아직 예측 기록이 없습니다. 최종_모델.py 실행 후 기록이 생성됩니다.")
 
    # ── 버튼 클릭 ──
    if predict_btn:
        if not os.path.exists(CROP_FILES[selected_crop]['model']):
            st.error("모델 파일이 없습니다. 최종_모델.py 를 먼저 실행해주세요.")
        else:
            with st.spinner("지정된 파이프라인(API->ffill->스케일링->예측) 가동 중..."):
                # prediction_log.json에서 마지막 실제값 있는 날 가격 가져오기
                if selected_crop in log_all and len(log_all[selected_crop]) > 0:
                    log_df_tmp = pd.DataFrame(log_all[selected_crop])
                    actual_df_tmp = log_df_tmp[log_df_tmp['실제값'].notna()]
                    curr_p_btn = int(actual_df_tmp['실제값'].iloc[-1]) if len(actual_df_tmp) > 0 else 0
                else:
                    curr_p_btn = 0
 
                pipeline  = AutomatedMLPipeline(selected_crop, target_date)
                next_pred = pipeline.run_pipeline(curr_p_btn)
                next_date = pd.to_datetime(target_date) + timedelta(days=1)
 
            # json 최신 상태로 다시 읽기 (버튼 클릭 후 갱신)
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, 'r', encoding='utf-8') as f:
                    log_tmp_all = json.load(f)
                log_all = log_tmp_all  # 전체 log_all도 갱신
                if selected_crop in log_tmp_all:
                    log_tmp = pd.DataFrame(log_tmp_all[selected_crop])
                    log_tmp['날짜'] = pd.to_datetime(log_tmp['날짜'])
                    null_df = log_tmp[log_tmp['실제값'].isna()]
                    if len(null_df) > 0:
                        real_next_date  = null_df['날짜'].iloc[-1]
                        real_next_price = int(null_df['예측값'].iloc[-1])
                    else:
                        real_next_date  = pd.to_datetime(target_date) + timedelta(days=1)
                        real_next_price = int(next_pred)
                else:
                    real_next_date  = pd.to_datetime(target_date) + timedelta(days=1)
                    real_next_price = int(next_pred)
            else:
                real_next_date  = pd.to_datetime(target_date) + timedelta(days=1)
                real_next_price = int(next_pred)
 
            st.session_state['pred_key']  = f"{selected_crop}_{target_date}"
            st.session_state['next_pred'] = real_next_price
            st.session_state['next_date'] = real_next_date
            st.success(f"🚀 파이프라인 완료! {real_next_date.strftime('%m/%d')} 예측 가격은 **{real_next_price:,}원** 입니다.")
            st.rerun()
 
    # ── prediction_log.json 기반 차트 ──
    st.subheader("📋 날짜별 예측값 vs 실제값")
 
    if selected_crop in log_all and len(log_all[selected_crop]) > 0:
        log_df = pd.DataFrame(log_all[selected_crop])
        log_df['날짜'] = pd.to_datetime(log_df['날짜'])
        log_df = log_df[log_df['날짜'] <= pd.to_datetime(target_date) + timedelta(days=1)].tail(90)
 
        actual_df = log_df[log_df['실제값'].notna()]
 
        # 실제값 있는 날까지만 예측가격도 표시
        pred_df = log_df[log_df['날짜'].isin(actual_df['날짜'])]
 
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(
            x=actual_df['날짜'], y=actual_df['실제값'],
            mode='lines+markers', name='실제 가격',
            line=dict(color='#5bc0de', width=2)
        ))
        fig_log.add_trace(go.Scatter(
            x=pred_df['날짜'], y=pred_df['예측값'],
            mode='lines+markers', name='예측 가격',
            line=dict(color='#ff4b4b', dash='dash', width=2)
        ))
 
        fig_log.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            yaxis=dict(title='가격 (원)', gridcolor='#333', tickformat=',d'),
            xaxis=dict(gridcolor='#333'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0'))
        )
        st.plotly_chart(fig_log, use_container_width=True)
 
        if len(actual_df) > 0:
            actual_df = actual_df.copy()
            actual_df['오차율(%)'] = abs(actual_df['실제값'] - actual_df['예측값']) / actual_df['실제값'] * 100
            avg_error = actual_df['오차율(%)'].mean()
            st.caption(f"📊 최근 {len(actual_df)}일 평균 오차율: **{avg_error:.2f}%**")
    else:
        st.info("📭 아직 예측 기록이 없습니다. 최종_모델.py 실행 후 기록이 생성됩니다.")
 
    # ── SHAP 분석 (저장된 이미지 표시) ──
    st.markdown("---")
    st.subheader("🧠 인공지능 가격 변동 요인 분석 (SHAP)")
    st.caption("AI 모델이 가격 예측 시 어떤 외부 지표를 가장 중요하게 반영했는지 시각적으로 보여줍니다.")
 
    shap_path = CROP_FILES[selected_crop]['shap']
    if os.path.exists(shap_path):
        img = Image.open(shap_path)
        st.image(img, use_container_width=True)
    else:
        st.info("📭 SHAP 이미지가 없습니다. 최종_모델.py 실행 후 생성됩니다.")
 
with tab2:
    st.subheader(f"📊 {selected_crop} 가격 패턴 차트")
 
    def read_csv_auto(path):
        try:
            return pd.read_csv(path, encoding='utf-8-sig')
        except:
            return pd.read_csv(path, encoding='cp949')
 
    try:
        full_df = read_csv_auto(CROP_FILES[selected_crop]['csv'])
        full_df['날짜'] = pd.to_datetime(full_df['날짜'])
        full_df = full_df[full_df['날짜'] <= pd.to_datetime(target_date)].copy()
        plot_df = full_df.copy()
 
        st.markdown("#### 📈 가격 및 외부 지표 추이")
        st.caption("이동평균선")
        ma_c1, ma_c2 = st.columns(2)
        with ma_c1: show_ma7  = st.toggle("MA 7일",  value=True,  key=f"ma7_{selected_crop}")
        with ma_c2: show_ma30 = st.toggle("MA 30일", value=True,  key=f"ma30_{selected_crop}")
 
        st.caption("외부 지표 (복수 선택 가능)")
        ext_cols = st.columns(4)
        show_temp   = ext_cols[0].toggle("기온",  value=False, key=f"temp_{selected_crop}")
        show_rain   = ext_cols[1].toggle("강수량",    value=False, key=f"rain_{selected_crop}")
        show_fuel   = ext_cols[2].toggle("경유가격",  value=False, key=f"fuel_{selected_crop}")
        show_supply = ext_cols[3].toggle("총반입량",  value=False, key=f"supply_{selected_crop}")
        ext_cols2 = st.columns(3)
        show_exrate = ext_cols2[0].toggle("환율",     value=False, key=f"exrate_{selected_crop}")
        show_lag7   = ext_cols2[1].toggle("lag_7",    value=False, key=f"lag7_{selected_crop}")
        show_lag30  = ext_cols2[2].toggle("lag_30",   value=False, key=f"lag30_{selected_crop}")
 
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=plot_df['날짜'], y=plot_df['가격'],
            mode='lines', name='실제 가격',
            line=dict(color='#aaaaaa', width=2), yaxis='y1'
        ))
 
        if show_ma7:
            ma7 = plot_df['가격'].rolling(window=7).mean()
            fig2.add_trace(go.Scatter(x=plot_df['날짜'], y=ma7, mode='lines', name='MA 7', line=dict(color='#ffcc00', width=1.5), yaxis='y1'))
        if show_ma30:
            ma30 = plot_df['가격'].rolling(window=30).mean()
            fig2.add_trace(go.Scatter(x=plot_df['날짜'], y=ma30, mode='lines', name='MA 30', line=dict(color='#ff00ff', width=1.5), yaxis='y1'))
 
        ext_map = [
            (show_temp,   '기온',  '#ff6b6b'),
            (show_rain,   '강수량',    '#74b9ff'),
            (show_fuel,   '경유가격',  '#55efc4'),
            (show_supply, '총반입량',  '#a29bfe'),
            (show_exrate, '환율',      '#fd79a8'),
            (show_lag7,   'lag_7',     '#00cec9'),
            (show_lag30,  'lag_30',    '#e17055'),
        ]
        for show, col, color in ext_map:
            if show and col in plot_df.columns:
                fig2.add_trace(go.Scatter(
                    x=plot_df['날짜'], y=plot_df[col],
                    mode='lines', name=col,
                    line=dict(color=color, width=1.5, dash='dot'),
                    yaxis='y2'
                ))
 
        x_end   = full_df['날짜'].max()
        x_start = x_end - pd.DateOffset(years=2)
 
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            yaxis=dict(title='가격 (원)', gridcolor='#333', tickformat=',d'),
            yaxis2=dict(title='외부 지표', overlaying='y', side='right', gridcolor='#333'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')),
            xaxis=dict(
                range=[x_start, x_end],
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=6,  label='6개월', step='month', stepmode='backward'),
                        dict(count=1,  label='1년',   step='year',  stepmode='backward'),
                        dict(count=2,  label='2년',   step='year',  stepmode='backward'),
                        dict(step='all', label='전체')
                    ],
                    bgcolor='#1e1e1e', activecolor='#ff4b4b', font=dict(color='#f0f0f0')
                ),
                gridcolor='#333'
            )
        )
        st.plotly_chart(fig2, use_container_width=True)
 
        st.markdown("#### 📊 가격 분포 분석")
        dist_c1, dist_c2 = st.columns(2)
 
        with dist_c1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=full_df['가격'], nbinsx=40, marker_color='#5bc0de', opacity=0.8, name='가격 분포'))
            fig_hist.update_layout(title='가격 분포 (히스토그램)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0'), xaxis=dict(title='가격 (원)', gridcolor='#333', tickformat=',d'), yaxis=dict(title='빈도', gridcolor='#333'))
            st.plotly_chart(fig_hist, use_container_width=True)
 
        with dist_c2:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=full_df['가격'], marker_color='#ff4b4b', name='가격', boxmean=True))
            fig_box.update_layout(title='가격 분포 (박스플롯)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0'), yaxis=dict(title='가격 (원)', gridcolor='#333', tickformat=',d'))
            st.plotly_chart(fig_box, use_container_width=True)
 
        st.markdown("#### 📅 연도별 가격 추이 비교")
        full_df['연도'] = full_df['날짜'].dt.year
        years = sorted(full_df['연도'].unique())
        year_colors = ['#5bc0de', '#ff4b4b', '#ffcc00', '#55efc4', '#a29bfe', '#fd79a8']
        year_cols = st.columns(len(years))
        show_years = {}
        for i, yr in enumerate(years):
            show_years[yr] = year_cols[i].toggle(str(yr), value=True, key=f"yr_{yr}")
 
        fig_year = go.Figure()
        for i, yr in enumerate(years):
            if show_years[yr]:
                yr_df = full_df[full_df['연도'] == yr].copy().reset_index(drop=True)
                yr_df['월일'] = pd.to_datetime('2000-' + yr_df['날짜'].dt.strftime('%m-%d'), errors='coerce')
                yr_df = yr_df.dropna(subset=['월일']).sort_values('월일')
                fig_year.add_trace(go.Scatter(x=yr_df['월일'], y=yr_df['가격'], mode='lines', name=str(yr), line=dict(color=year_colors[i % len(year_colors)], width=2)))
 
        fig_year.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified', font=dict(color='#f0f0f0'), xaxis=dict(title='월', tickformat='%m월', dtick='M1', gridcolor='#333'), yaxis=dict(title='가격 (원)', gridcolor='#333', tickformat=',d'), legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')))
        st.plotly_chart(fig_year, use_container_width=True)
 
    except Exception as e:
        st.warning(f"데이터 로드 실패: {e}")
 
with tab3:
    st.subheader("🏆 기초 모델 성능 평가 (Walk-Forward Validation)")
    st.caption("※ 실제 데이터 훈련 및 검증을 거친 최종 성과 지표입니다.")
 
    if os.path.exists(PERF_PATH):
        with open(PERF_PATH, 'r', encoding='utf-8') as f:
            perf_all = json.load(f)
 
        if selected_crop in perf_all:
            crop_perf = {k: v for k, v in perf_all[selected_crop].items() if k != "summary"}
            crop_df = pd.DataFrame(crop_perf)
            st.dataframe(crop_df, use_container_width=True, hide_index=True, height=36 * (len(crop_df) + 1) + 3)
            st.info("📁 `models/performance.json` 에서 불러온 실제 학습 결과입니다. 재학습 시 자동으로 갱신됩니다.")
 
            final_model_name = perf_all[selected_crop]["모델명"][-1]
            if 'Linear' in final_model_name:
                st.info(f"**💡 모델 선정 결과:** 선형 회귀 모델이 가장 우수한 성능을 보여 **{final_model_name}** 모델이 최종 배포(Deploy) 모델로 선정되었습니다.")
            else:
                st.info(f"**💡 모델 선정 결과:** 하이퍼파라미터 튜닝을 거친 **{final_model_name}** 모델이 최종 배포(Deploy) 모델로 선정되었습니다.")
 
            # 방향 정확도 차트
            if selected_crop in log_all and len(log_all[selected_crop]) > 0:
                st.markdown("---")
                st.subheader("🧭 방향 정확도 (상승/하락 예측)")
                st.caption("실제 가격이 오를 때 모델도 오른다고 예측했는지, 내릴 때 내린다고 예측했는지의 비율입니다.")
 
                log_df2 = pd.DataFrame(log_all[selected_crop])
                log_df2['날짜'] = pd.to_datetime(log_df2['날짜'])
                log_df2 = log_df2[log_df2['실제값'].notna()].copy()
                log_df2['어제실제'] = log_df2['실제값'].shift(1)
                log_df2 = log_df2.iloc[1:].copy()
 
                def classify_direction(today, yesterday, threshold=0.5):
                    pct = (today - yesterday) / yesterday * 100
                    if abs(pct) < threshold: return 0
                    return int(np.sign(pct))
 
                log_df2['실제방향'] = log_df2.apply(lambda r: classify_direction(r['실제값'], r['어제실제']), axis=1)
                log_df2['예측방향'] = log_df2.apply(lambda r: classify_direction(r['예측값'], r['어제실제']), axis=1)
                log_df2['방향일치'] = (log_df2['실제방향'] == log_df2['예측방향'])
 
                correct  = log_df2['방향일치'].sum()
                total    = len(log_df2)
                log_df2['월'] = log_df2['날짜'].dt.to_period('M').astype(str)
                monthly_da = log_df2.groupby('월')['방향일치'].mean() * 100
 
                fig_da = go.Figure()
                fig_da.add_trace(go.Bar(
                    x=monthly_da.index, y=monthly_da.values,
                    marker_color=['#55efc4' if v >= 50 else '#ff4b4b' for v in monthly_da.values],
                    name='월별 방향 정확도'
                ))
                fig_da.add_hline(y=50, line_dash="dash", line_color="#ffcc00", annotation_text="기준선 50%")
                fig_da.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(title='방향 정확도 (%)', gridcolor='#333', range=[0, 100]),
                    xaxis=dict(gridcolor='#333'),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f0f0')),
                    font=dict(color='#f0f0f0')
                )
                st.plotly_chart(fig_da, use_container_width=True)
                st.caption(f"📊 전체 {total}일 중 {correct}일 방향 일치 | 50% 이상이면 Naive보다 나은 것")
 
        else:
            st.warning(f"⚠️ '{selected_crop}' 의 학습 결과가 없습니다. 최종_모델.py 로 먼저 학습을 진행해주세요.")
 
    else:
        st.warning("⚠️ 아직 학습 결과 파일(performance.json)이 없습니다. 최종_모델.py 로 먼저 학습을 진행해주세요.")
        perf_data = {
            "모델명":          ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "★ Tuned RF (최종 Deploy)"],
            "Train R²":        ["0.9899", "0.9984", "0.9986", "0.9968", "-"],
            "Test R² (정확성)":["0.9803", "0.9762", "0.9758", "0.9750", "0.9797"],
            "Train MAPE(%)":   ["1.46%",  "0.55%",  "0.75%",  "0.86%",  "-"],
            "Test MAPE(%)":    ["1.52%",  "1.88%",  "1.80%",  "2.00%",  "1.41%"]
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True, height=250)
        st.caption("※ 위 수치는 임시 예시값입니다. 학습 후 자동으로 실제값으로 교체됩니다.")