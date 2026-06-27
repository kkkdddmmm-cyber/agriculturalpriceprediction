"""
농산물 가격 예측 모델 학습 코드 (로컬 환경)
실행 방법: python 최종_모델.py
 
변경사항:
- Train/Test CSV 대신 원본 CSV에서 1913개 추출 후 분리
- prediction_log.json 기준으로 예측 날짜 결정
- 표준화 추가 (Train fit → Test transform)
- 결측치 처리 (강수량→0, 나머지→ffill)
"""
 
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings, os, re, json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import clone
import joblib
import shap
 
warnings.filterwarnings('ignore')
 
# ── 0. 폰트 설정 (로컬/클라우드 공용) ────────────────────────
import matplotlib.font_manager as fm
import urllib.request

def _setup_font():
    # 1) 시스템에 한글 폰트가 이미 있으면 사용 (윈도우=Malgun Gothic 등)
    for name in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']:
        if any(name in f.name for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
    # 2) 없으면 NanumGothic 다운로드 (리눅스/클라우드)
    font_path = 'NanumGothic.ttf'
    if not os.path.exists(font_path):
        try:
            urllib.request.urlretrieve(
                'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf',
                font_path
            )
        except Exception:
            pass
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False

_setup_font()
 
# ── 1. 파일 경로 설정 (로컬/클라우드 공용) ──────────────────
# 이 스크립트가 있는 폴더를 기준으로 동작 → 윈도우/리눅스 어디서든 OK
FOLDER_PATH   = os.path.dirname(os.path.abspath(__file__)) + '/'
OUTPUT_FOLDER = FOLDER_PATH + 'models/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
LOG_PATH  = OUTPUT_FOLDER + 'prediction_log.json'
PERF_PATH = OUTPUT_FOLDER + 'performance.json'
 
TOTAL_SIZE  = 1913   # 슬라이딩 윈도우 전체 크기
TRAIN_RATIO = 0.8    # Train 비율
 
# ★ 학습할 품목 목록 - 원본 CSV 사용
CROP_LIST = [
    {"name": "양파",   "csv": "보관/00양파_통합.csv"},
    {"name": "배추",   "csv": "보관/00배추_통합.csv"},
    {"name": "고구마", "csv": "보관/00고구마_통합.csv"},
]
 
# ── 2. 헬퍼 함수 ─────────────────────────────────────────────
 
def read_csv_auto(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        return pd.read_csv(file_path, encoding='cp949')
 
def handle_missing(df):
    """강수량 → 0, 나머지 → ffill"""
    df = df.copy()
    rain_cols = [c for c in df.columns if '강수량' in c]
    for col in rain_cols:
        df[col] = df[col].fillna(0)
    df = df.ffill()
    return df
 
def build_features(df_combined, price_col):
    """
    Train+Test를 합친 DataFrame에서 피처를 생성합니다.
    - lag_1 제거 (echo 현상 방지)
    - 기상/수급 시차 피처 생성 (lag3/7/14, roll7)
    - Train/Test 경계에서 NaN 없이 연속 계산 보장
    """
    X = df_combined.drop(columns=[price_col, '연도'], errors='ignore').select_dtypes(include=[np.number])
    X = X.rename(columns=lambda x: re.sub(r'[\[\]<>,]', '_', str(x)))
 
    # lag_1 제거
    lag1_cols = [c for c in X.columns if re.fullmatch(r'lag_?1', c, re.IGNORECASE)]
    if lag1_cols:
        X = X.drop(columns=lag1_cols)
        print(f"   ℹ️  lag_1 피처 제거됨: {lag1_cols}")
 
    added_cols = []
 
    # 총반입량: lag3, roll7 (lag7, lag14 제거 - 중복 정보)
    for mc in [c for c in X.columns if '총반입량' in c]:
        X[f'{mc}_lag3']  = X[mc].shift(3);       added_cols.append(f'{mc}_lag3')
        X[f'{mc}_roll7'] = X[mc].rolling(7).mean(); added_cols.append(f'{mc}_roll7')
 
    # 강수량: lag3
    for mc in [c for c in X.columns if '강수량' in c and 'lag' not in c and 'roll' not in c]:
        X[f'{mc}_lag3'] = X[mc].shift(3); added_cols.append(f'{mc}_lag3')
 
    # 기온: roll30
    for mc in [c for c in X.columns if '기온' in c and 'lag' not in c and 'roll' not in c]:
        X[f'{mc}_roll30'] = X[mc].rolling(30).mean(); added_cols.append(f'{mc}_roll30')
 
    # 가격: ma_30, lag_365 (ma_7 제거 - ma_30에 포함, lag_1 제외 - echo 현상)
    if price_col in df_combined.columns:
        X['ma_30']   = df_combined[price_col].rolling(30).mean(); added_cols.append('ma_30')
        X['lag_365'] = df_combined[price_col].shift(365);         added_cols.append('lag_365')
 
    if added_cols:
        print(f"   ✅ 기상/수급 시차 피처 추가됨 ({len(added_cols)}개)")
 
    return X
 
def load_and_prep(csv_path, target_date=None):
    """
    원본 CSV에서 TOTAL_SIZE개 추출 후 Train/Test 분리 + 표준화
    target_date: 해당 날짜 이전 데이터 기준 (None이면 전체 최근 데이터)
    """
    df = read_csv_auto(csv_path)
 
    date_col  = next((c for c in df.columns if '날짜' in c or 'date' in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if '가격' in c or 'price' in c.lower()), df.columns[1])
 
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
 
    # target_date 이전 데이터만 사용
    if target_date is not None:
        df = df[df[date_col] < pd.to_datetime(target_date)]
 
    # 최근 TOTAL_SIZE개 추출
    df = df.tail(TOTAL_SIZE).copy()
    df = df.set_index(date_col)
 
    print(f"   📅 전체 기간: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}행)")
 
    # 결측치 처리
    df = handle_missing(df)
 
    # Train+Test 합쳐서 피처 생성 (경계 NaN 방지)
    X_all = build_features(df, price_col)
    y_all = df[price_col]
    if isinstance(y_all, pd.DataFrame): y_all = y_all.iloc[:, 0]
 
    # NaN 제거 (앞부분 최대 14행 손실)
    X_all = X_all.dropna()
    y_all = y_all.loc[X_all.index]
 
    # Train/Test 분리 (80/20)
    n_train = int(len(X_all) * TRAIN_RATIO)
    X_train = X_all.iloc[:n_train]
    y_train = y_all.iloc[:n_train]
    X_test  = X_all.iloc[n_train:]
    y_test  = y_all.iloc[n_train:]
 
    # ── 표준화 (Train fit → Test transform) ──
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
 
    col_mapping = dict(zip(X_all.columns, X_all.columns))
 
    return X_train_scaled, y_train, X_test_scaled, y_test, col_mapping, scaler
 
 
# ── 3. prediction_log.json 로드 ──────────────────────────────
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        existing_log = json.load(f)
else:
    existing_log = {}
 
 
# ── 4. 품목별 반복 학습 ──────────────────────────────────────
for crop in CROP_LIST:
    CROP_NAME = crop["name"]
    CSV_FILE  = crop["csv"]
    csv_path  = FOLDER_PATH + CSV_FILE
    base_name = CSV_FILE.replace('보관/', '').replace('.csv', '')
 
    print("\n" + "=" * 70)
    print(f"🌱 [{CROP_NAME}] 학습 시작")
    print("=" * 70)
 
    if not os.path.exists(csv_path):
        print(f"⚠️ 파일을 찾을 수 없습니다: {csv_path} → 건너뜀")
        continue
 
    # ── prediction_log.json 기준으로 예측 시작 날짜 결정 ──
    existing_entries = existing_log.get(CROP_NAME, [])
    existing_dates   = {e["날짜"] for e in existing_entries}
 
    # 원본 CSV 날짜 목록
    df_all = read_csv_auto(csv_path)
    date_col  = next((c for c in df_all.columns if '날짜' in c), df_all.columns[0])
    price_col = next((c for c in df_all.columns if '가격' in c), df_all.columns[1])
    df_all[date_col] = pd.to_datetime(df_all[date_col])
    df_all = df_all.sort_values(date_col).reset_index(drop=True)
    all_csv_dates = df_all[date_col].dt.strftime('%Y-%m-%d').tolist()
    csv_max_date  = df_all[date_col].max()
 
    # ── 실제값(null) 백필: CSV에 실제 가격이 들어온 과거 예측 항목을 채움 ──
    price_by_date = dict(zip(
        df_all[date_col].dt.strftime('%Y-%m-%d'),
        df_all[price_col]
    ))
    backfilled = 0
    for e in existing_entries:
        v = e.get("실제값")
        is_null = (v is None) or (isinstance(v, float) and pd.isna(v))
        if is_null and e["날짜"] in price_by_date:
            real = price_by_date[e["날짜"]]
            if pd.notna(real):
                e["실제값"] = round(float(real), 1)
                backfilled += 1
    if backfilled:
        existing_log[CROP_NAME] = existing_entries   # 즉시 반영
        print(f"   🔄 [{CROP_NAME}] 실제값 백필: {backfilled}건 채움")

    # json 마지막 예측 날짜 다음부터 예측
    if existing_dates:
        last_pred_date = max(existing_dates)
        if last_pred_date in all_csv_dates:
            start_idx = all_csv_dates.index(last_pred_date) + 1
        else:
            start_idx = TOTAL_SIZE + 1
        predict_dates = [d for d in all_csv_dates[start_idx:] if d not in existing_dates]
    else:
        predict_dates = []
 
    # ── json 비어있으면 한번에 Test 전체 예측 ──
    if not existing_dates:
        print(f"   ℹ️ prediction_log.json 비어있음 → Test 기간 전체 한번에 예측")
        try:
            X_train, y_train, X_test, y_test, col_map, scaler = load_and_prep(csv_path)
            print(f"✅ Train: {len(X_train)}행 | Test: {len(X_test)}행")
 
            full_y      = pd.concat([y_train, y_test])
            full_y_prev = full_y.shift(1)
            full_y_diff = full_y.diff()
 
            y_train_prev = full_y_prev.loc[y_train.index]
            y_test_prev  = full_y_prev.loc[y_test.index]
            y_train_diff = full_y_diff.loc[y_train.index]
            y_test_diff  = full_y_diff.loc[y_test.index]
 
            X_train      = X_train.iloc[1:]
            y_train      = y_train.iloc[1:]
            y_train_prev = y_train_prev.iloc[1:]
            y_train_diff = y_train_diff.iloc[1:]
 
            if pd.isna(y_test_prev.iloc[0]):
                y_test_prev.iloc[0] = y_train.iloc[-1]
 
            # 4개 모델 학습
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'XGBoost':           XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, n_jobs=-1),
                'LightGBM':          LGBMRegressor(n_estimators=100, random_state=42, learning_rate=0.1, n_jobs=-1, verbose=-1)
            }
            results = {}
            print("\n⏳ 1단계: 4개 기본 모델 학습...\n")
            for name, model in models.items():
                model.fit(X_train, y_train_diff.values.ravel())
                train_pred = y_train_prev.values.ravel() + model.predict(X_train)
                test_pred  = y_test_prev.values.ravel()  + model.predict(X_test)
                train_r2   = r2_score(y_train.values.ravel(), train_pred)
                test_r2    = r2_score(y_test.values.ravel(), test_pred)
                train_mape = mean_absolute_percentage_error(y_train.values.ravel(), train_pred) * 100
                test_mape  = mean_absolute_percentage_error(y_test.values.ravel(), test_pred) * 100
                print(f"🤖 [{name}] Train R²={train_r2:.4f} MAPE={train_mape:.2f}% | Test R²={test_r2:.4f} MAPE={test_mape:.2f}%")
                results[name] = {'Train R²': train_r2, 'Test R²': test_r2, 'Train MAPE(%)': train_mape, 'Test MAPE(%)': test_mape}
 
            results_df = pd.DataFrame(results).T
 
            # 튜닝
            tree_results   = results_df.drop('Linear Regression', errors='ignore')
            best_tree_name = tree_results['Test R²'].idxmax()
            tscv = TimeSeriesSplit(n_splits=5)
            if 'Random Forest' in best_tree_name:
                base_model = RandomForestRegressor(random_state=42, n_jobs=1)
                param_grid = {'n_estimators': [100, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [5, 10]}
            elif 'XGBoost' in best_tree_name:
                base_model = XGBRegressor(random_state=42, n_jobs=1)
                param_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
            else:
                base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=1)
                param_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
 
            grid_search = GridSearchCV(base_model, param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=1)
            grid_search.fit(X_train, y_train_diff.values.ravel())
            best_tuned_model = grid_search.best_estimator_
 
            tuned_pred = y_test_prev.values.ravel() + best_tuned_model.predict(X_test)
            tuned_mape = mean_absolute_percentage_error(y_test.values.ravel(), tuned_pred) * 100
 
            if tuned_mape < results_df.loc['Linear Regression', 'Test MAPE(%)']:
                final_model      = best_tuned_model
                final_model_name = f"Tuned {best_tree_name}"
            else:
                final_model      = models['Linear Regression']
                final_model_name = "Linear Regression"
            print(f"   최종 선택 모델: {final_model_name}")
 
            # Walk-Forward로 Test 전체 예측
            print(f"\n🚀 Walk-Forward 시작... ({len(X_test)}개)")
            WINDOW_SIZE    = len(X_train)
            history_X      = X_train.copy()
            history_y_diff = y_train_diff.copy()
            history_y_real = y_train.copy()
            rolling_model  = clone(final_model)
            wf_predictions = []
 
            for i in range(len(X_test)):
                window_X = history_X.iloc[-WINDOW_SIZE:]
                window_yd = history_y_diff.iloc[-WINDOW_SIZE:]
                rolling_model.fit(window_X, window_yd.values.ravel())
                next_X     = X_test.iloc[[i]]
                pred_diff  = rolling_model.predict(next_X)[0]
                pred_price = history_y_real.iloc[-1] + pred_diff
                wf_predictions.append(pred_price)
                history_X      = pd.concat([history_X, next_X])
                history_y_diff = pd.concat([history_y_diff, y_test_diff.iloc[[i]]])
                history_y_real = pd.concat([history_y_real, y_test.iloc[i:i+1]])
                if (i + 1) % 50 == 0:
                    print(f"   진행 중: {i+1}/{len(X_test)}")
 
            # Test 전체 예측값 json에 저장
            for i in range(len(X_test)):
                date_str = y_test.index[i].strftime('%Y-%m-%d')
                real_val = round(float(y_test.values[i]), 1)
                pred_val = round(float(wf_predictions[i]), 1)
                existing_entries.append({"날짜": date_str, "예측값": pred_val, "실제값": real_val})
                existing_dates.add(date_str)
            print(f"   ✅ Test 전체 {len(X_test)}개 예측값 json 저장 완료")
 
            # 모델/Scaler 저장
            final_model.fit(history_X, history_y_diff.values.ravel())
            joblib.dump(final_model, OUTPUT_FOLDER + f"Deploy_{base_name}.joblib")
            joblib.dump(scaler,      OUTPUT_FOLDER + f"Scaler_{base_name}.joblib")
            print(f"   ✅ 모델/Scaler 저장 완료")
 
            # 성능 지표 계산
            wf_predictions_arr = np.array(wf_predictions)
            wf_mape = mean_absolute_percentage_error(y_test.values.ravel(), wf_predictions_arr) * 100
            wf_r2   = r2_score(y_test.values.ravel(), wf_predictions_arr)
            wf_mae  = np.mean(np.abs(y_test.values.ravel() - wf_predictions_arr))
 
            y_prev_arr = np.concatenate([[y_train.iloc[-1]], y_test.values[:-1]])
            def classify_dir(today, yesterday, threshold=0.5):
                pct = (today - yesterday) / yesterday * 100
                if abs(pct) < threshold: return 0
                return int(np.sign(pct))
            true_dir = np.array([classify_dir(y_test.values[i], y_prev_arr[i]) for i in range(len(y_test))])
            pred_dir = np.array([classify_dir(wf_predictions_arr[i], y_prev_arr[i]) for i in range(len(wf_predictions_arr))])
            wf_da    = np.mean(true_dir == pred_dir) * 100
 
            # performance.json 저장
            new_perf = {
                CROP_NAME: {
                    "summary": {
                        "MAE":      f"{round(wf_mae, 1)}원",
                        "R2":       str(round(wf_r2, 4)),
                        "MAPE":     f"{round(wf_mape, 2)}%",
                        "방향정확도": f"{round(wf_da, 1)}%",
                        "MASE":     "0.0"
                    },
                    "모델명":           ["Linear Regression", "Random Forest", "XGBoost", "LightGBM",
                                        f"★ {final_model_name} (최종 Deploy)"],
                    "Train R²":         [str(round(results_df.loc['Linear Regression', 'Train R²'], 4)),
                                         str(round(results_df.loc['Random Forest', 'Train R²'], 4)),
                                         str(round(results_df.loc['XGBoost', 'Train R²'], 4)),
                                         str(round(results_df.loc['LightGBM', 'Train R²'], 4)), "-"],
                    "Test R² (정확성)": [str(round(results_df.loc['Linear Regression', 'Test R²'], 4)),
                                         str(round(results_df.loc['Random Forest', 'Test R²'], 4)),
                                         str(round(results_df.loc['XGBoost', 'Test R²'], 4)),
                                         str(round(results_df.loc['LightGBM', 'Test R²'], 4)),
                                         str(round(wf_r2, 4))],
                    "Train MAPE(%)":    [f"{round(results_df.loc['Linear Regression', 'Train MAPE(%)'], 2)}%",
                                         f"{round(results_df.loc['Random Forest', 'Train MAPE(%)'], 2)}%",
                                         f"{round(results_df.loc['XGBoost', 'Train MAPE(%)'], 2)}%",
                                         f"{round(results_df.loc['LightGBM', 'Train MAPE(%)'], 2)}%", "-"],
                    "Test MAPE(%)":     [f"{round(results_df.loc['Linear Regression', 'Test MAPE(%)'], 2)}%",
                                         f"{round(results_df.loc['Random Forest', 'Test MAPE(%)'], 2)}%",
                                         f"{round(results_df.loc['XGBoost', 'Test MAPE(%)'], 2)}%",
                                         f"{round(results_df.loc['LightGBM', 'Test MAPE(%)'], 2)}%",
                                         f"{round(wf_mape, 2)}%"]
                }
            }
            if os.path.exists(PERF_PATH):
                with open(PERF_PATH, 'r', encoding='utf-8') as f:
                    existing_perf = json.load(f)
                existing_perf.update(new_perf)
                new_perf = existing_perf
            with open(PERF_PATH, 'w', encoding='utf-8') as f:
                json.dump(new_perf, f, ensure_ascii=False, indent=2)
            print(f"   ✅ performance.json 저장 완료")
 
            # SHAP 분석
            print("   ⏳ SHAP 분석 중...")
            history_X_korean = history_X.rename(columns=col_map)
            if 'Linear' in final_model_name:
                explainer   = shap.LinearExplainer(final_model, history_X_korean)
                shap_values = explainer.shap_values(history_X_korean)
            else:
                explainer   = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(history_X_korean)
            plt.figure(figsize=(10, 8))
            plt.title(f'SHAP 분석: [{CROP_NAME}] {final_model_name}의 가격 등락 결정 요인',
                      fontsize=14, fontweight='bold', pad=20)
            shap.summary_plot(shap_values, history_X_korean, show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_FOLDER + f'SHAP_{base_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ SHAP 이미지 저장 완료")
 
        except Exception as e:
            print(f"   ⚠️ 초기 예측 실패: {e}")
 
    # 예측할 날짜가 없으면 내일 예측만
    if not predict_dates:
        print(f"   ✅ 이미 최신 예측 완료 → 내일 예측만 수행")
    else:
        print(f"   예측할 날짜: {predict_dates[0]} ~ {predict_dates[-1]} ({len(predict_dates)}일)")
 
    # ── 날짜별 순서대로 예측 ──
    for date_str in predict_dates:
        target_date = pd.to_datetime(date_str)
        print(f"\n   📅 [{date_str}] 예측 시작...")
 
        # 해당 날짜 이전 TOTAL_SIZE개로 Train/Test 분리
        print("📂 데이터 로드 중...")
        try:
            X_train, y_train, X_test, y_test, col_map, scaler = load_and_prep(csv_path, target_date)
        except Exception as e:
            print(f"   ⚠️ 데이터 준비 실패: {e}")
            continue
 
        print(f"✅ Train: {len(X_train)}행 | Test: {len(X_test)}행")
 
        # ── 차분 적용 ──
        full_y      = pd.concat([y_train, y_test])
        full_y_prev = full_y.shift(1)
        full_y_diff = full_y.diff()
 
        y_train_prev = full_y_prev.loc[y_train.index]
        y_test_prev  = full_y_prev.loc[y_test.index]
        y_train_diff = full_y_diff.loc[y_train.index]
        y_test_diff  = full_y_diff.loc[y_test.index]
 
        X_train      = X_train.iloc[1:]
        y_train      = y_train.iloc[1:]
        y_train_prev = y_train_prev.iloc[1:]
        y_train_diff = y_train_diff.iloc[1:]
 
        if pd.isna(y_test_prev.iloc[0]):
            y_test_prev.iloc[0] = y_train.iloc[-1]
 
        # ── 1단계: 기본 모델 학습 ──
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost':           XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, n_jobs=-1),
            'LightGBM':          LGBMRegressor(n_estimators=100, random_state=42, learning_rate=0.1, n_jobs=-1, verbose=-1)
        }
 
        results = {}
        print("\n⏳ 1단계: 4개 기본 모델 학습...\n")
 
        for name, model in models.items():
            model.fit(X_train, y_train_diff.values.ravel())
 
            train_pred_diff = model.predict(X_train)
            train_pred      = y_train_prev.values.ravel() + train_pred_diff
            train_r2        = r2_score(y_train.values.ravel(), train_pred)
            train_mape      = mean_absolute_percentage_error(y_train.values.ravel(), train_pred) * 100
 
            test_pred_diff = model.predict(X_test)
            test_pred      = y_test_prev.values.ravel() + test_pred_diff
            test_r2        = r2_score(y_test.values.ravel(), test_pred)
            test_mape      = mean_absolute_percentage_error(y_test.values.ravel(), test_pred) * 100
 
            mape_gap = test_mape - train_mape
            status = "🚨 과대적합" if mape_gap > 2.0 else "⚠️ 주의" if mape_gap > 1.0 else "✅ 안정"
            print(f"🤖 [{name}] Train R²={train_r2:.4f} MAPE={train_mape:.2f}% | Test R²={test_r2:.4f} MAPE={test_mape:.2f}% {status}")
 
            results[name] = {
                'Train R²': train_r2, 'Test R²': test_r2,
                'Train MAPE(%)': train_mape, 'Test MAPE(%)': test_mape
            }
 
        results_df = pd.DataFrame(results).T
 
        # ── 2단계: 최고 모델 튜닝 ──
        tree_results   = results_df.drop('Linear Regression', errors='ignore')
        best_tree_name = tree_results['Test R²'].idxmax()
        print(f"\n⚙️ 2단계: [{best_tree_name}] 튜닝 시작...")
 
        tscv = TimeSeriesSplit(n_splits=5)
 
        if 'Random Forest' in best_tree_name:
            base_model = RandomForestRegressor(random_state=42, n_jobs=1)
            param_grid = {'n_estimators': [100, 300], 'max_depth': [5, 10, 15], 'min_samples_split': [5, 10]}
        elif 'XGBoost' in best_tree_name:
            base_model = XGBRegressor(random_state=42, n_jobs=1)
            param_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
        else:
            base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=1)
            param_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
 
        grid_search = GridSearchCV(base_model, param_grid, cv=tscv,
                                   scoring='neg_mean_absolute_percentage_error', n_jobs=1)
        grid_search.fit(X_train, y_train_diff.values.ravel())
 
        best_tuned_model = grid_search.best_estimator_
        print(f"   최적 파라미터: {grid_search.best_params_}")
 
        tuned_pred_diff = best_tuned_model.predict(X_test)
        tuned_pred_bulk = y_test_prev.values.ravel() + tuned_pred_diff
        tuned_mape_bulk = mean_absolute_percentage_error(y_test.values.ravel(), tuned_pred_bulk) * 100
        tuned_r2_bulk   = r2_score(y_test.values.ravel(), tuned_pred_bulk)
 
        if tuned_mape_bulk < results_df.loc['Linear Regression', 'Test MAPE(%)']:
            final_model      = best_tuned_model
            final_model_name = f"Tuned {best_tree_name}"
        else:
            final_model      = models['Linear Regression']
            final_model_name = "Linear Regression"
 
        print(f"   최종 선택 모델: {final_model_name}")
 
        # ── 3단계: Walk-Forward (슬라이딩 윈도우 고정) ──
        print(f"\n🚀 3단계: Walk-Forward... ({final_model_name})")
 
        WINDOW_SIZE    = len(X_train)
        history_X      = X_train.copy()
        history_y_diff = y_train_diff.copy()
        history_y_real = y_train.copy()
 
        wf_predictions = []
        rolling_model  = clone(final_model)
        total_days     = len(X_test)
 
        for i in range(total_days):
            window_X      = history_X.iloc[-WINDOW_SIZE:]
            window_y_diff = history_y_diff.iloc[-WINDOW_SIZE:]
 
            rolling_model.fit(window_X, window_y_diff.values.ravel())
            next_day_X = X_test.iloc[[i]]
 
            pred_diff  = rolling_model.predict(next_day_X)[0]
            pred_price = history_y_real.iloc[-1] + pred_diff
            wf_predictions.append(pred_price)
 
            history_X      = pd.concat([history_X, next_day_X])
            history_y_diff = pd.concat([history_y_diff, y_test_diff.iloc[[i]]])
            history_y_real = pd.concat([history_y_real, y_test.iloc[i:i+1]])
 
            if (i + 1) % 50 == 0:
                print(f"   진행 중: {i+1}/{total_days}")
 
        wf_predictions = np.array(wf_predictions)
        wf_mape = mean_absolute_percentage_error(y_test.values.ravel(), wf_predictions) * 100
        wf_r2   = r2_score(y_test.values.ravel(), wf_predictions)
        wf_mae  = np.mean(np.abs(y_test.values.ravel() - wf_predictions))
 
        # 방향 정확도 (±0.5% 이내 보합 처리)
        y_prev_arr = np.concatenate([[y_train.iloc[-1]], y_test.values[:-1]])
 
        def classify_direction(today, yesterday, threshold=0.5):
            pct = (today - yesterday) / yesterday * 100
            if abs(pct) < threshold:
                return 0
            return int(np.sign(pct))
 
        true_dir = np.array([classify_direction(y_test.values[i], y_prev_arr[i]) for i in range(len(y_test))])
        pred_dir = np.array([classify_direction(wf_predictions[i], y_prev_arr[i]) for i in range(len(wf_predictions))])
        wf_da    = np.mean(true_dir == pred_dir) * 100
        mase_val = 0.0
 
        print(f"   방향 정확도: {wf_da:.1f}%")
 
        # ── 최종 성능 비교 ──
        comparison_dict = {
            '1. 선형 회귀 (기준)':              {'Test MAPE(%)': results_df.loc['Linear Regression', 'Test MAPE(%)'], 'Test R²': results_df.loc['Linear Regression', 'Test R²']},
            f'2. {best_tree_name} (튜닝 전)':   {'Test MAPE(%)': results_df.loc[best_tree_name, 'Test MAPE(%)'],      'Test R²': results_df.loc[best_tree_name, 'Test R²']},
            f'3. {best_tree_name} (튜닝 후)':   {'Test MAPE(%)': tuned_mape_bulk,                                     'Test R²': tuned_r2_bulk},
            '4. Walk-Forward (최종 실전 예측)': {'Test MAPE(%)': wf_mape,                                             'Test R²': wf_r2}
        }
        print("\n" + "=" * 70)
        print(f"🏆 [{CROP_NAME}] 최종 성능 비교")
        print("=" * 70)
        print(pd.DataFrame(comparison_dict).T.round(4))
        print("=" * 70)
 
        # ── 모델 저장 ──
        final_model.fit(history_X, history_y_diff.values.ravel())
        save_filename = OUTPUT_FOLDER + f"Deploy_{base_name}.joblib"
        joblib.dump(final_model, save_filename)
        print(f"\n✅ 모델 저장: {save_filename}")
 
        # Scaler 저장
        scaler_filename = OUTPUT_FOLDER + f"Scaler_{base_name}.joblib"
        joblib.dump(scaler, scaler_filename)
        print(f"✅ Scaler 저장: {scaler_filename}")
 
        # ── performance.json 저장 ──
        new_perf = {
            CROP_NAME: {
                "summary": {
                    "MAE":      f"{round(wf_mae, 1)}원",
                    "R2":       str(round(wf_r2, 4)),
                    "MAPE":     f"{round(wf_mape, 2)}%",
                    "방향정확도": f"{round(wf_da, 1)}%",
                    "MASE":     str(round(mase_val, 4))
                },
                "모델명":           ["Linear Regression", "Random Forest", "XGBoost", "LightGBM",
                                    f"★ {final_model_name} (최종 Deploy)"],
                "Train R²":         [str(round(results_df.loc['Linear Regression', 'Train R²'], 4)),
                                     str(round(results_df.loc['Random Forest', 'Train R²'], 4)),
                                     str(round(results_df.loc['XGBoost', 'Train R²'], 4)),
                                     str(round(results_df.loc['LightGBM', 'Train R²'], 4)), "-"],
                "Test R² (정확성)": [str(round(results_df.loc['Linear Regression', 'Test R²'], 4)),
                                     str(round(results_df.loc['Random Forest', 'Test R²'], 4)),
                                     str(round(results_df.loc['XGBoost', 'Test R²'], 4)),
                                     str(round(results_df.loc['LightGBM', 'Test R²'], 4)),
                                     str(round(wf_r2, 4))],
                "Train MAPE(%)":    [f"{round(results_df.loc['Linear Regression', 'Train MAPE(%)'], 2)}%",
                                     f"{round(results_df.loc['Random Forest', 'Train MAPE(%)'], 2)}%",
                                     f"{round(results_df.loc['XGBoost', 'Train MAPE(%)'], 2)}%",
                                     f"{round(results_df.loc['LightGBM', 'Train MAPE(%)'], 2)}%", "-"],
                "Test MAPE(%)":     [f"{round(results_df.loc['Linear Regression', 'Test MAPE(%)'], 2)}%",
                                     f"{round(results_df.loc['Random Forest', 'Test MAPE(%)'], 2)}%",
                                     f"{round(results_df.loc['XGBoost', 'Test MAPE(%)'], 2)}%",
                                     f"{round(results_df.loc['LightGBM', 'Test MAPE(%)'], 2)}%",
                                     f"{round(wf_mape, 2)}%"]
            }
        }
 
        if os.path.exists(PERF_PATH):
            with open(PERF_PATH, 'r', encoding='utf-8') as f:
                existing_perf = json.load(f)
            existing_perf.update(new_perf)
            new_perf = existing_perf
 
        with open(PERF_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_perf, f, ensure_ascii=False, indent=2)
        print(f"✅ 성능 지표 저장: {PERF_PATH}")
 
        # ── prediction_log.json 저장 ──
        # Walk-Forward 완료 후 target_date 예측값 저장
        # target_date의 실제값
        real_row = df_all[df_all[date_col] == target_date]
        real_val = round(float(real_row[price_col].values[0]), 1) if len(real_row) > 0 else None
 
        # Walk-Forward 마지막 예측값 = target_date 예측값
        next_day_X     = history_X.iloc[[-1]]
        window_X_last  = history_X.iloc[-WINDOW_SIZE:]
        window_yd_last = history_y_diff.iloc[-WINDOW_SIZE:]
        rolling_model.fit(window_X_last, window_yd_last.values.ravel())
        next_pred_diff  = rolling_model.predict(next_day_X)[0]
        next_pred_price = history_y_real.iloc[-1] + next_pred_diff
 
        existing_entries.append({
            "날짜":   date_str,
            "예측값": round(float(next_pred_price), 1),
            "실제값": real_val
        })
        existing_dates.add(date_str)
        print(f"✅ {date_str} 예측값: {round(float(next_pred_price), 1)}원 | 실제값: {real_val}원")
 
        # ── SHAP 분석 ──
        print("\n⏳ SHAP 분석 중...")
        history_X_korean = history_X.rename(columns=col_map)
 
        if 'Linear' in final_model_name:
            explainer   = shap.LinearExplainer(final_model, history_X_korean)
            shap_values = explainer.shap_values(history_X_korean)
        else:
            explainer   = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(history_X_korean)
 
        plt.figure(figsize=(10, 8))
        plt.title(f'SHAP 분석: [{CROP_NAME}] {final_model_name}의 가격 등락 결정 요인',
                  fontsize=14, fontweight='bold', pad=20)
        shap.summary_plot(shap_values, history_X_korean, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER + f'SHAP_{base_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP 이미지 저장: {OUTPUT_FOLDER}SHAP_{base_name}.png")
 
        print(f"\n🎉 [{CROP_NAME}] {date_str} 완료!")
 
    # ── 내일 예측값 저장 (CSV 마지막 다음날) ──
    next_pred_date     = csv_max_date + pd.Timedelta(days=1)
    next_pred_date_str = next_pred_date.strftime('%Y-%m-%d')
 
    if next_pred_date_str not in existing_dates:
        print(f"\n   📅 내일({next_pred_date_str}) 예측 중...")
        try:
            X_train, y_train, X_test, y_test, col_map, scaler = load_and_prep(csv_path)
 
            full_y      = pd.concat([y_train, y_test])
            full_y_diff = full_y.diff()
            y_train_diff = full_y_diff.loc[y_train.index].iloc[1:]
            y_test_diff  = full_y_diff.loc[y_test.index]
            X_train      = X_train.iloc[1:]
            y_train      = y_train.iloc[1:]
 
            WINDOW_SIZE    = len(X_train)
            history_X      = X_train.copy()
            history_y_diff = y_train_diff.copy()
            history_y_real = y_train.copy()
 
            # final_model이 없으면 joblib에서 로드, 그것도 없으면 LightGBM 사용
            try:
                _fm = final_model
            except NameError:
                try:
                    _fm = joblib.load(OUTPUT_FOLDER + f"Deploy_{base_name}.joblib")
                    print(f"   ✅ 저장된 모델 로드: Deploy_{base_name}.joblib")
                except:
                    _fm = LGBMRegressor(n_estimators=100, random_state=42, learning_rate=0.1, n_jobs=-1, verbose=-1)
                    print(f"   ℹ️ LightGBM 기본 모델 사용")
            rolling_model = clone(_fm)
 
            for i in range(len(X_test)):
                window_X      = history_X.iloc[-WINDOW_SIZE:]
                window_y_diff = history_y_diff.iloc[-WINDOW_SIZE:]
                rolling_model.fit(window_X, window_y_diff.values.ravel())
                next_X     = X_test.iloc[[i]]
                pred_diff  = rolling_model.predict(next_X)[0]
                pred_price = history_y_real.iloc[-1] + pred_diff
                history_X      = pd.concat([history_X, next_X])
                history_y_diff = pd.concat([history_y_diff, y_test_diff.iloc[[i]]])
                history_y_real = pd.concat([history_y_real, y_test.iloc[i:i+1]])
 
            next_day_X     = history_X.iloc[[-1]]
            window_X_last  = history_X.iloc[-WINDOW_SIZE:]
            window_yd_last = history_y_diff.iloc[-WINDOW_SIZE:]
            rolling_model.fit(window_X_last, window_yd_last.values.ravel())
            next_pred_diff  = rolling_model.predict(next_day_X)[0]
            next_pred_price = history_y_real.iloc[-1] + next_pred_diff
 
            existing_entries.append({
                "날짜":   next_pred_date_str,
                "예측값": round(float(next_pred_price), 1),
                "실제값": None
            })
            print(f"   ✅ 내일({next_pred_date_str}) 예측값: {round(float(next_pred_price), 1)}원")
 
        except Exception as e:
            print(f"   ⚠️ 내일 예측 실패: {e}")
 
    # prediction_log 업데이트
    existing_log[CROP_NAME] = sorted(existing_entries, key=lambda x: x["날짜"])
 
# ── prediction_log.json 최종 저장 ────────────────────────────
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    json.dump(existing_log, f, ensure_ascii=False, indent=2)
print(f"\n✅ prediction_log.json 저장 완료: {LOG_PATH}")
 
print("\n" + "=" * 70)
print("🏆 전체 품목 학습 완료!")
print(f"   성능 JSON: {PERF_PATH}")
print("=" * 70)