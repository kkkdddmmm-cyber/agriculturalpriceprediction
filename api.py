"""
농산물 가격 예측 시스템 - 데이터 수집 및 CSV 업데이트
실행 방법: python api.py (매일 오후 11시 자동 실행)
"""
 
import requests
import pandas as pd
import numpy as np
import urllib.parse
import os
import json
from datetime import datetime, timedelta
 
# ── API 키 설정 (환경변수 / .env 에서 읽기 — 코드에 비밀값 저장 금지) ──
# 로컬: 같은 폴더에 .env 파일을 두면 자동 로드 (이 파일은 .gitignore 로 제외)
# 클라우드: GitHub Secrets → 워크플로 env 로 주입
try:
    from dotenv import load_dotenv          # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

KAMIS_KEY    = os.environ.get('KAMIS_KEY',    '')
KAMIS_ID     = os.environ.get('KAMIS_ID',     '')
WEATHER_KEY  = os.environ.get('WEATHER_KEY',  '')
OPINET_KEY   = os.environ.get('OPINET_KEY',   '')
GARAK_ID     = os.environ.get('GARAK_ID',     '')
GARAK_PASSWD = os.environ.get('GARAK_PASSWD', '')

# 키 누락 시 조기 경고 (어떤 키가 비었는지 알려줌)
_missing = [k for k, v in {
    'KAMIS_KEY': KAMIS_KEY, 'KAMIS_ID': KAMIS_ID, 'WEATHER_KEY': WEATHER_KEY,
    'OPINET_KEY': OPINET_KEY, 'GARAK_ID': GARAK_ID, 'GARAK_PASSWD': GARAK_PASSWD,
}.items() if not v]
if _missing:
    print(f"⚠️ 환경변수 누락: {', '.join(_missing)} → .env 또는 GitHub Secrets 를 확인하세요.")
 
# ── 파일 경로 설정 (로컬/클라우드 공용) ──────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
 
CROP_FILES = {
    '양파':   BASE_PATH + '보관/00양파_통합.csv',
    '배추':   BASE_PATH + '보관/00배추_통합.csv',
    '고구마': BASE_PATH + '보관/00고구마_통합.csv',
}
 
# ── 기상청 관측소 코드 ────────────────────────────────────────
CROP_STATIONS = {
    '양파':   {'기온': ['288', '264'],        '강수량': ['288', '264', '165']},
    '배추':   {'기온': ['105', '261'],        '강수량': ['105', '261']},
    '고구마': {'기온': ['203', '261', '140'], '강수량': ['203', '261', '140']},
}
 
# ════════════════════════════════════════════════════════════
# 1. KAMIS 가격 수집
# ════════════════════════════════════════════════════════════
def get_kamis_price(date_str):
    url = 'http://www.kamis.or.kr/service/price/xml.do'
    params = {
        'action':          'dailyPriceByCategoryList',
        'p_cert_key':      KAMIS_KEY,
        'p_cert_id':       KAMIS_ID,
        'p_returntype':    'json',
        'p_convert_kg_yn': 'N',
        'p_country_code':  '',
        'p_regday':        date_str,
    }
 
    result = {}
 
    try:
        # 채소류 (양파, 배추)
        res  = requests.get(url, params={**params, 'p_product_cls_code': '01', 'p_item_category_code': '200'}, timeout=10)
        data = res.json().get('data', {})
        if isinstance(data, dict):
            df = pd.DataFrame(data.get('item', []))
 
            # 양파
            onion = df[(df['item_name'] == '양파') & (df['rank'] == '상품')]
            if len(onion) > 0:
                result['양파'] = float(onion.iloc[0]['dpr1'].replace(',', ''))
 
            # 배추 (현재 유통 품종 단순 평균)
            cabbage = df[(df['item_name'] == '배추') & (df['rank'] == '상품')]
            valid   = cabbage[cabbage['dpr1'] != '-'].copy()
            if len(valid) > 0:
                valid['가격'] = valid['dpr1'].str.replace(',', '').astype(float)
                result['배추'] = round(valid['가격'].mean(), 0)
 
        # 식량작물 (고구마)
        res2  = requests.get(url, params={**params, 'p_product_cls_code': '01', 'p_item_category_code': '100'}, timeout=10)
        data2 = res2.json().get('data', {})
        if isinstance(data2, dict):
            df2 = pd.DataFrame(data2.get('item', []))
            sp  = df2[(df2['item_name'] == '고구마') & (df2['kind_name'].str.contains('밤')) & (df2['rank'] == '상품')]
            if len(sp) > 0:
                result['고구마'] = float(sp.iloc[0]['dpr1'].replace(',', ''))
 
    except Exception as e:
        print(f"   ⚠️ KAMIS API 오류: {e}")
 
    return result
 
 
# ════════════════════════════════════════════════════════════
# 2. 기상청 ASOS 기온/강수량 수집
# ════════════════════════════════════════════════════════════
def get_weather(date_str, crop):
    url     = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
    dec_key = urllib.parse.unquote(WEATHER_KEY)
    stations = CROP_STATIONS[crop]
    temps, rains = [], []
 
    for stn_id in stations['기온']:
        params = {
            'serviceKey': dec_key, 'pageNo': '1', 'numOfRows': '1',
            'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
            'startDt': date_str.replace('-', ''),
            'endDt':   date_str.replace('-', ''),
            'stnIds':  stn_id
        }
        try:
            item = requests.get(url, params=params, timeout=10).json()['response']['body']['items']['item'][0]
            temps.append(float(item['avgTa']))
        except: pass
 
    for stn_id in stations['강수량']:
        params = {
            'serviceKey': dec_key, 'pageNo': '1', 'numOfRows': '1',
            'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
            'startDt': date_str.replace('-', ''),
            'endDt':   date_str.replace('-', ''),
            'stnIds':  stn_id
        }
        try:
            item = requests.get(url, params=params, timeout=10).json()['response']['body']['items']['item'][0]
            rain = item.get('sumRn', '0')
            rains.append(float(rain) if rain else 0.0)
        except: pass
 
    return {
        '기온': round(sum(temps) / len(temps), 4) if temps else None,
        '강수량':   round(sum(rains) / len(rains), 4) if rains else None
    }
 
 
# ════════════════════════════════════════════════════════════
# 3. 환율 수집
# ════════════════════════════════════════════════════════════
def get_exchange_rate(date_str):
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader('USD/KRW', date_str, date_str)
        if not df.empty:
            return round(float(df['Close'].iloc[0]), 2)
    except: pass
    return None
 
 
# ════════════════════════════════════════════════════════════
# 4. 오피넷 경유가격 수집 (최근 7일치)
# ════════════════════════════════════════════════════════════
def get_opinet_prices():
    """최근 7일치 경유가격 딕셔너리 반환 {날짜: 가격}"""
    try:
        url = f'http://www.opinet.co.kr/api/avgRecentPrice.do?out=json&code={OPINET_KEY}'
        res = requests.get(url, timeout=10)
        df  = pd.DataFrame(res.json()['RESULT']['OIL'])
        diesel = df[df['PRODCD'] == 'D047'].copy()
        diesel['DATE'] = pd.to_datetime(diesel['DATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        return dict(zip(diesel['DATE'], diesel['PRICE'].astype(float)))
    except Exception as e:
        print(f"   ⚠️ 오피넷 API 오류: {e}")
        return {}
 
 
# ════════════════════════════════════════════════════════════
# 5. 가락시장 총반입량 수집
# ════════════════════════════════════════════════════════════
def get_supply(date_str):
    url     = 'http://www.garak.co.kr/homepage/publicdata/dataJsonOpen.do'
    all_dfs = []
    try:
        for page in ['1', '2']:
            params = {
                'id': GARAK_ID, 'passwd': GARAK_PASSWD,
                'dataid': 'data22', 'pagesize': '100',
                'pageidx': page, 'portal.templet': 'false',
                'date': date_str.replace('-', '')
            }
            res = requests.get(url, params=params, timeout=10)
            all_dfs.append(pd.DataFrame(res.json()['resultData']))
 
        full_df = pd.concat(all_dfs, ignore_index=True)
        result  = {}
        for crop in ['양파', '배추', '고구마']:
            row = full_df[full_df['PUM_NM'] == crop]
            if len(row) > 0:
                result[crop] = round(float(row.iloc[0]['SUM_TOT']), 2)
        return result
    except Exception as e:
        print(f"   ⚠️ 가락시장 API 오류: {e}")
        return {}
 
 
# ════════════════════════════════════════════════════════════
# CSV 읽기/쓰기
# ════════════════════════════════════════════════════════════
def read_csv(crop):
    path = CROP_FILES[crop]
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='cp949')
    df['날짜'] = pd.to_datetime(df['날짜'])
    return df.sort_values('날짜').reset_index(drop=True)
 
def save_csv(df, crop):
    path = CROP_FILES[crop]
    df.to_csv(path, index=False, encoding='utf-8-sig')
 
 
# ════════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════════
def main():
    today = datetime.today().date()
    print(f"\n{'='*60}")
    print(f"🚀 농산물 데이터 수집 시작 ({today})")
    print(f"{'='*60}")
 
    # 오피넷 최근 7일치 미리 수집
    opinet_prices = get_opinet_prices()
    print(f"✅ 오피넷 경유가격 수집: {len(opinet_prices)}일치")
 
    for crop in ['양파', '배추', '고구마']:
        print(f"\n{'─'*50}")
        print(f"🌱 [{crop}] 데이터 수집 시작")
        print(f"{'─'*50}")
 
        # CSV 로드
        df = read_csv(crop)
        last_date = df['날짜'].max().date()
        print(f"   CSV 마지막 날짜: {last_date}")
 
        # 수집할 날짜 범위
        collect_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            end=today
        )
 
        if len(collect_dates) == 0:
            print(f"   ✅ 이미 최신 데이터입니다.")
            continue
 
        print(f"   수집할 날짜: {collect_dates[0].date()} ~ {collect_dates[-1].date()} ({len(collect_dates)}일)")
 
        new_rows = []
        prev_row = df.iloc[-1].copy()  # ffill용 이전 행
 
        for date in collect_dates:
            date_str = date.strftime('%Y-%m-%d')
            print(f"   📅 {date_str} 수집 중...")
 
            new_row = {'날짜': date}
 
            # 1. 가격 수집
            prices = get_kamis_price(date_str)
            new_row['가격'] = prices.get(crop, None)
 
            # 2. 기상 수집
            weather = get_weather(date_str, crop)
            new_row['기온'] = weather['기온']
            new_row['강수량']   = weather['강수량']
 
            # 3. 환율 수집
            new_row['환율'] = get_exchange_rate(date_str)
 
            # 4. 경유가격 수집
            new_row['경유가격'] = opinet_prices.get(date_str, None)
 
            # 5. 총반입량 수집
            supply = get_supply(date_str)
            new_row['총반입량'] = supply.get(crop, None)
 
            new_rows.append(new_row)
 
        if not new_rows:
            continue
 
        # 새 행 DataFrame 생성
        new_df = pd.DataFrame(new_rows)
 
        # 결측치 처리
        # 강수량 → 0
        new_df['강수량'] = new_df['강수량'].fillna(0)
 
        # 나머지 → ffill (이전 행 기준)
        for col in ['가격', '기온', '환율', '경유가격', '총반입량']:
            for i in range(len(new_df)):
                if pd.isna(new_df.loc[i, col]):
                    if i == 0:
                        new_df.loc[i, col] = prev_row[col]
                    else:
                        new_df.loc[i, col] = new_df.loc[i-1, col]
 
        # CSV에 추가
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values('날짜').reset_index(drop=True)
 
        # 저장
        save_csv(df, crop)
        print(f"   ✅ [{crop}] CSV 업데이트 완료! ({len(new_rows)}행 추가)")
 
    print(f"\n{'='*60}")
    print(f"🏆 전체 데이터 수집 완료!")
    print(f"{'='*60}")
 
 
if __name__ == '__main__':
    main()