"""
어지럼증 감별진단 예측 시스템
- 5개 질환 모델: BPPV, VN, SSNHL, Meniere, Others
- CatBoost 기반 예측 + SHAP 분석
- Streamlit Cloud 배포용 (Google Drive 연동)
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import os
import tempfile

# 한글 폰트 설정 (Streamlit Cloud 호환)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

# Google Drive API
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ========================================
# 설정
# ========================================
st.set_page_config(
    page_title="어지럼증 감별진단 예측 시스템",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS: Container 스타일 + 버튼 고정
st.markdown("""
<style>
    /* 사이드바 숨김 버튼 제거 */
    button[kind="header"] {
        display: none !important;
    }
    
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* 사이드바 배경 */
    section[data-testid="stSidebar"] > div {
        background-color: #f0f2f6;
        padding-bottom: 100px;
    }
    
    /* Container를 흰색 박스로 스타일링 */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    /* 섹션 제목 스타일 */
    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        padding-bottom: 8px;
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* 예측 버튼 하단 고정 */
    div.stButton > button[kind="primary"] {
        position: fixed !important;
        bottom: 20px !important;
        width: 310px !important;
        z-index: 999 !important;
        background-color: #FF4B4B !important;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive 파일 ID 설정
FILE_IDS = {
    'BPPV': '1BfcFZ6-RnQbg_Qo4eO8YbkkRR-9CvZdg',
    'VN': '158NQ9j_OKQoaA5JpMCjl1NgqKqc2Ezdb',
    'SSNHL': '1EiKOeCMv55m3021VW5FNTZcylZLATSTE',
    'Meniere': '1v_9jwfd6w9iJyiAIRplMOqJfqRiEqvSY',
    'Others': '1giDrrAwDntoAm9Xt8vS1eMrKo4zVlBwG',
}

DISEASE_NAMES_KR = {
    'BPPV': '양성돌발성체위현훈 (BPPV)',
    'VN': '전정신경염 (Vestibular Neuritis)',
    'SSNHL': '돌발성난청 (SSNHL)',
    'Meniere': '메니에르병 (Meniere\'s Disease)',
    'Others': '기타 원인',
}

# ========================================
# Feature 정의
# ========================================
INPUT_FEATURES = [
    'symptoms_frequency', 'symptoms_recurrence', 'symptom_recent',
    'symptom_remote_cat', 'symptom_remote_cat_is_1st_attack',
    'symptom_remote_cat_is_within_30days', 'symptom_remote_cat_is_within_1years',
    'symptom_remote_cat_is_over_1year', 'symptoms_true_vertigo',
    'symptoms_dizziness_duration_ongoing', 'symptoms_duration_minutes',
    'symptoms_duration_minutes_cat_gen', 'symptoms_duration_minutes_cat_gen_is_several_sec',
    'symptoms_duration_minutes_cat_gen_is_several_min', 'symptoms_duration_minutes_cat_gen_is_several_hours',
    'symptoms_duration_minutes_cat_gen_is_several_days', 'symptoms_duration_minutes_cat_20m',
    'symptoms_duration_minutes_cat_20m_is_several_sec', 'symptoms_duration_minutes_cat_20m_is_several_min',
    'symptoms_duration_minutes_cat_20m_is_several_hours', 'symptoms_duration_minutes_cat_20m_is_several_days',
    'symptoms_nausea', 'symptoms_vomiting', 'symptoms_headache', 'symptoms_black_out',
    'symptoms_agg_factor_position_change', 'symptoms_agg_factor_head_rotation',
    'symptoms_agg_factor_eyes_moving', 'symptoms_agg_factor_moving',
    'symptoms_agg_factor_no_moving', 'symptoms_agg_factor_position_change_combined',
    'symptoms_rel_factor_rest', 'symptoms_rel_factor_eyes_closed',
    'symptoms_hearing_impairment_combined', 'symptoms_tinnitus', 'symptoms_ear_fullness',
    'history_dm', 'history_htn', 'history_pul_tbc', 'history_asthma',
    'history_kidney', 'history_entop', 'history_trauma', 'history_ear_disease',
    'history_neckop', 'history_brain_disease', 'history_metabolic_disease',
    'history_coronary_disease', 'history_stomach', 'history_bph', 'history_gynecologic',
    'history_eye_disease', 'history_psychiatric', 'history_thyroid_disease',
    'history_pci', 'history_abdominalop', 'history_respiratory_disease',
    'history_orthopedicop', 'history_ra', 'history_autoimmune_disease',
    'etc_sn_right', 'etc_sn_left', 'etc_gaze_right', 'etc_gaze_left',
    'etc_dht_right', 'etc_dht_left', 'etc_rht_right', 'etc_rht_left',
    'etc_gn_right', 'etc_gn_left', 'etc_hit_right', 'etc_hit_left',
    'etc_hsn_right', 'etc_hsn_left', 'etc_htt_right', 'etc_htt_left',
    'etc_skew_deviation_right', 'etc_skew_deviation_left',
    'etc_weber_right', 'etc_weber_left', 'age', 'sex'
]

# ========================================
# Google Drive 연동 함수
# ========================================
def get_google_drive_service():
    """Google Drive API 서비스 생성"""
    try:
        creds_dict = dict(st.secrets["google"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Google Drive 인증 오류: {e}")
        return None

def download_file_from_drive(service, file_id, destination):
    """Google Drive에서 파일 다운로드"""
    try:
        request = service.files().get_media(fileId=file_id)
        with open(destination, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return True
    except Exception as e:
        st.error(f"파일 다운로드 오류: {e}")
        return False

# ========================================
# 모델 로드 함수
# ========================================
@st.cache_resource
def load_models():
    """Google Drive에서 모델 다운로드 및 로드 (캐싱)"""
    service = get_google_drive_service()
    if service is None:
        return None
    
    models = {}
    temp_dir = tempfile.gettempdir()
    
    for name, file_id in FILE_IDS.items():
        try:
            temp_path = os.path.join(temp_dir, f"label_{name.lower()}_model.joblib")
            
            if download_file_from_drive(service, file_id, temp_path):
                models[name] = joblib.load(temp_path)
            else:
                st.error(f"{name} 모델 다운로드 실패")
                return None
        except Exception as e:
            st.error(f"{name} 모델 로드 오류: {e}")
            return None
    
    return models

# ========================================
# 입력 UI 함수 (Container 사용)
# ========================================
def create_sidebar_inputs():
    """사이드바 입력 UI 생성 - Container로 그룹화"""
    st.sidebar.title("🩺 환자 정보 입력")
    
    inputs = {}
    
    # ========== 기본 정보 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">📋 기본 정보</div>', unsafe_allow_html=True)
        inputs['patient_name'] = st.text_input("환자 이름", value="", key="patient_name")
        inputs['age'] = st.number_input("나이", min_value=10, max_value=100, value=50, key="age")
        sex_option = st.selectbox("성별", ["여성", "남성"], key="sex")
        inputs['sex'] = 1 if sex_option == "여성" else 0
    
    # ========== 어지럼증 특성 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">🌀 어지럼증 특성</div>', unsafe_allow_html=True)
        
        inputs['symptoms_true_vertigo'] = float(st.checkbox("회전성 어지럼증 (빙글빙글 도는 느낌)", key="true_vertigo"))
        inputs['symptoms_dizziness_duration_ongoing'] = float(st.checkbox("현재 어지럼증 지속 중", key="ongoing"))
        
        inputs['symptom_recent'] = st.number_input(
            "최근 어지럼증 발생일 (며칠 전)", min_value=0, max_value=180, value=1, key="recent"
        )
        
        frequency_options = {
            "1회": 1, "2-3회": 2, "4-5회": 3, "6-10회": 4, "10회 이상": 5
        }
        freq_selected = st.selectbox("어지럼증 발생 빈도", list(frequency_options.keys()), key="frequency")
        inputs['symptoms_frequency'] = float(frequency_options[freq_selected])
        
        inputs['symptoms_recurrence'] = float(st.checkbox("재발성 어지럼증", key="recurrence"))
        
        duration_cat_options = {
            "수 초": 1, "수 분": 2, "수 시간": 3, "수 일": 4
        }
        duration_selected = st.selectbox("어지럼증 지속 시간", list(duration_cat_options.keys()), key="duration")
        inputs['symptoms_duration_minutes_cat_gen'] = float(duration_cat_options[duration_selected])
        
        inputs['symptoms_duration_minutes_cat_gen_is_several_sec'] = 1.0 if duration_selected == "수 초" else 0.0
        inputs['symptoms_duration_minutes_cat_gen_is_several_min'] = 1.0 if duration_selected == "수 분" else 0.0
        inputs['symptoms_duration_minutes_cat_gen_is_several_hours'] = 1.0 if duration_selected == "수 시간" else 0.0
        inputs['symptoms_duration_minutes_cat_gen_is_several_days'] = 1.0 if duration_selected == "수 일" else 0.0
        
        inputs['symptoms_duration_minutes_cat_20m'] = inputs['symptoms_duration_minutes_cat_gen']
        inputs['symptoms_duration_minutes_cat_20m_is_several_sec'] = inputs['symptoms_duration_minutes_cat_gen_is_several_sec']
        inputs['symptoms_duration_minutes_cat_20m_is_several_min'] = inputs['symptoms_duration_minutes_cat_gen_is_several_min']
        inputs['symptoms_duration_minutes_cat_20m_is_several_hours'] = inputs['symptoms_duration_minutes_cat_gen_is_several_hours']
        inputs['symptoms_duration_minutes_cat_20m_is_several_days'] = inputs['symptoms_duration_minutes_cat_gen_is_several_days']
        
        duration_minutes_map = {"수 초": 0.5, "수 분": 5, "수 시간": 120, "수 일": 1440}
        inputs['symptoms_duration_minutes'] = duration_minutes_map[duration_selected]
        
        remote_cat_options = {
            "첫 발작": 0, "30일 이내": 1, "1년 이내": 2, "1년 이상": 3
        }
        remote_selected = st.selectbox("과거 어지럼증 발생 시점", list(remote_cat_options.keys()), key="remote")
        inputs['symptom_remote_cat'] = float(remote_cat_options[remote_selected])
        inputs['symptom_remote_cat_is_1st_attack'] = 1.0 if remote_selected == "첫 발작" else 0.0
        inputs['symptom_remote_cat_is_within_30days'] = 1.0 if remote_selected == "30일 이내" else 0.0
        inputs['symptom_remote_cat_is_within_1years'] = 1.0 if remote_selected == "1년 이내" else 0.0
        inputs['symptom_remote_cat_is_over_1year'] = 1.0 if remote_selected == "1년 이상" else 0.0
    
    # ========== 동반 증상 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">🤢 동반 증상</div>', unsafe_allow_html=True)
        inputs['symptoms_nausea'] = float(st.checkbox("오심 (메스꺼움)", key="nausea"))
        inputs['symptoms_vomiting'] = float(st.checkbox("구토", key="vomiting"))
        inputs['symptoms_headache'] = float(st.checkbox("두통", key="headache"))
        inputs['symptoms_black_out'] = float(st.checkbox("실신/눈앞이 캄캄함", key="blackout"))
    
    # ========== 이과적 증상 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">👂 이과적 증상</div>', unsafe_allow_html=True)
        inputs['symptoms_hearing_impairment_combined'] = float(st.checkbox("청력 저하", key="hearing"))
        inputs['symptoms_tinnitus'] = float(st.checkbox("이명", key="tinnitus"))
        inputs['symptoms_ear_fullness'] = float(st.checkbox("이충만감", key="ear_fullness"))
    
    # ========== 악화/완화 요인 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">⚡ 악화/완화 요인</div>', unsafe_allow_html=True)
        
        st.markdown("**악화 요인**")
        inputs['symptoms_agg_factor_position_change'] = float(st.checkbox("체위 변화 시 악화", key="agg_position"))
        inputs['symptoms_agg_factor_head_rotation'] = float(st.checkbox("머리 회전 시 악화", key="agg_head"))
        inputs['symptoms_agg_factor_eyes_moving'] = float(st.checkbox("눈 움직일 때 악화", key="agg_eyes"))
        inputs['symptoms_agg_factor_moving'] = float(st.checkbox("움직일 때 악화", key="agg_moving"))
        inputs['symptoms_agg_factor_no_moving'] = float(st.checkbox("가만히 있을 때 악화", key="agg_no_moving"))
        inputs['symptoms_agg_factor_position_change_combined'] = inputs['symptoms_agg_factor_position_change']
        
        st.markdown("**완화 요인**")
        inputs['symptoms_rel_factor_rest'] = float(st.checkbox("휴식 시 완화", key="rel_rest"))
        inputs['symptoms_rel_factor_eyes_closed'] = float(st.checkbox("눈 감으면 완화", key="rel_eyes"))
    
    # ========== 과거력 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">📜 과거력</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            inputs['history_dm'] = float(st.checkbox("당뇨병", key="hist_dm"))
            inputs['history_htn'] = float(st.checkbox("고혈압", key="hist_htn"))
            inputs['history_ear_disease'] = float(st.checkbox("귀 질환", key="hist_ear"))
            inputs['history_brain_disease'] = float(st.checkbox("뇌질환", key="hist_brain"))
            inputs['history_thyroid_disease'] = float(st.checkbox("갑상선질환", key="hist_thyroid"))
            inputs['history_psychiatric'] = float(st.checkbox("정신과질환", key="hist_psych"))
        
        with col2:
            inputs['history_coronary_disease'] = float(st.checkbox("관상동맥질환", key="hist_coronary"))
            inputs['history_trauma'] = float(st.checkbox("외상력", key="hist_trauma"))
            inputs['history_entop'] = float(st.checkbox("이비인후과 수술력", key="hist_entop"))
            inputs['history_metabolic_disease'] = float(st.checkbox("대사질환", key="hist_metabolic"))
            inputs['history_autoimmune_disease'] = float(st.checkbox("자가면역질환", key="hist_autoimmune"))
            inputs['history_respiratory_disease'] = float(st.checkbox("호흡기질환", key="hist_respiratory"))
        
        other_history = [
            'history_pul_tbc', 'history_asthma', 'history_kidney', 'history_neckop',
            'history_stomach', 'history_bph', 'history_gynecologic', 'history_eye_disease',
            'history_pci', 'history_abdominalop', 'history_orthopedicop', 'history_ra'
        ]
        for h in other_history:
            if h not in inputs:
                inputs[h] = 0.0
    
    # ========== 신체검사 소견 ==========
    with st.sidebar.container():
        st.markdown('<div class="section-title">🔍 신체검사 소견</div>', unsafe_allow_html=True)
        
        st.markdown("**안진 검사**")
        col1, col2 = st.columns(2)
        with col1:
            inputs['etc_sn_right'] = float(st.checkbox("자발안진 (우)", key="sn_r"))
            inputs['etc_gaze_right'] = float(st.checkbox("주시안진 (우)", key="gaze_r"))
            inputs['etc_dht_right'] = float(st.checkbox("Dix-Hallpike (우)", key="dht_r"))
            inputs['etc_rht_right'] = float(st.checkbox("Roll test (우)", key="rht_r"))
        with col2:
            inputs['etc_sn_left'] = float(st.checkbox("자발안진 (좌)", key="sn_l"))
            inputs['etc_gaze_left'] = float(st.checkbox("주시안진 (좌)", key="gaze_l"))
            inputs['etc_dht_left'] = float(st.checkbox("Dix-Hallpike (좌)", key="dht_l"))
            inputs['etc_rht_left'] = float(st.checkbox("Roll test (좌)", key="rht_l"))
        
        st.markdown("**기타 검사**")
        col1, col2 = st.columns(2)
        with col1:
            inputs['etc_hit_right'] = float(st.checkbox("HIT (우)", key="hit_r"))
            inputs['etc_hsn_right'] = float(st.checkbox("HSN (우)", key="hsn_r"))
            inputs['etc_htt_right'] = float(st.checkbox("HTT (우)", key="htt_r"))
        with col2:
            inputs['etc_hit_left'] = float(st.checkbox("HIT (좌)", key="hit_l"))
            inputs['etc_hsn_left'] = float(st.checkbox("HSN (좌)", key="hsn_l"))
            inputs['etc_htt_left'] = float(st.checkbox("HTT (좌)", key="htt_l"))
        
        other_etc = [
            'etc_gn_right', 'etc_gn_left', 'etc_skew_deviation_right', 'etc_skew_deviation_left',
            'etc_weber_right', 'etc_weber_left'
        ]
        for e in other_etc:
            if e not in inputs:
                inputs[e] = 0.0
    
    return inputs

# ========================================
# 예측 함수
# ========================================
def predict_all_models(models, input_df):
    """모든 모델에서 예측 확률 계산"""
    probabilities = {}
    
    for name, model in models.items():
        try:
            prob = model.predict_proba(input_df)[0][1]
            probabilities[name] = prob
        except Exception as e:
            st.error(f"{name} 모델 예측 오류: {e}")
            probabilities[name] = 0.0
    
    return probabilities

def get_top_prediction(probabilities):
    """가장 높은 확률의 질환 반환"""
    top_disease = max(probabilities, key=probabilities.get)
    top_prob = probabilities[top_disease]
    return top_disease, top_prob

# ========================================
# SHAP 분석 함수
# ========================================
def generate_shap_plot(model, input_df, disease_name):
    """개별 환자에 대한 SHAP waterfall plot + bar plot 생성"""
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value
        
        feature_names = list(input_df.columns)
        
        sorted_idx = np.argsort(np.abs(sv))[::-1][:10]
        top_features = [feature_names[i] for i in sorted_idx]
        top_values = sv[sorted_idx]
        top_data = input_df.values[0][sorted_idx]
        
        f_x = base_value + np.sum(sv)
        
        fig_waterfall, ax = plt.subplots(figsize=(7, 5))
        
        y_pos = range(len(top_features))
        colors = ['#ff6b6b' if v > 0 else '#4dabf7' for v in top_values]
        
        bars = ax.barh(y_pos, top_values, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        
        y_labels = []
        for i in range(len(top_features)):
            val = top_data[i]
            if isinstance(val, float) and val == int(val):
                y_labels.append(f"{int(val)} = {top_features[i]}")
            elif isinstance(val, float):
                y_labels.append(f"{val:.2g} = {top_features[i]}")
            else:
                y_labels.append(f"{val} = {top_features[i]}")
        
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP value', fontsize=9)
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        for bar, val in zip(bars, top_values):
            width = bar.get_width()
            if val >= 0:
                label = f"+{val:.2f}"
            else:
                label = f"{val:.2f}"
            
            if abs(width) > 0.5:
                x_pos = width / 2
                color = 'white'
                ha = 'center'
            else:
                x_pos = width + 0.05 * (1 if width >= 0 else -1)
                color = 'black'
                ha = 'left' if width >= 0 else 'right'
            
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                   label, ha=ha, va='center', fontsize=8, color=color, fontweight='bold')
        
        ax.set_title(f'f(x) = {f_x:.3f}', fontsize=10)
        plt.tight_layout()
        
        sorted_idx_20 = np.argsort(np.abs(sv))[::-1][:20]
        top_features_20 = [feature_names[i] for i in sorted_idx_20]
        abs_values_20 = np.abs(sv[sorted_idx_20])
        
        fig_bar, ax_bar = plt.subplots(figsize=(8, 7))
        y_pos_20 = range(len(top_features_20))
        ax_bar.barh(y_pos_20, abs_values_20, color='#1E88E5', height=0.7)
        ax_bar.set_yticks(y_pos_20)
        ax_bar.set_yticklabels(top_features_20, fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=9)
        ax_bar.tick_params(axis='x', labelsize=8)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        plt.tight_layout()
        
        return fig_waterfall, fig_bar
    except Exception as e:
        st.error(f"SHAP 분석 오류: {e}")
        return None, None

# ========================================
# 메인 앱
# ========================================
def main():
    st.title("🩺 어지럼증 감별진단 예측 시스템")
    st.markdown("---")
    
    models = load_models()
    
    if models is None:
        st.error("모델을 로드할 수 없습니다. Google Drive 연결 및 Secrets 설정을 확인하세요.")
        st.stop()
    
    inputs = create_sidebar_inputs()
    
    input_data = {feat: inputs.get(feat, np.nan) for feat in INPUT_FEATURES}
    input_df = pd.DataFrame([input_data])
    
    predict_button = st.sidebar.button("🔮 예측 실행", type="primary", use_container_width=True, key="predict_btn")
    
    if predict_button:
        
        with st.spinner("예측 중..."):
            probabilities = predict_all_models(models, input_df)
            top_disease, top_prob = get_top_prediction(probabilities)
        
        st.header("📊 예측 결과")

        if inputs['patient_name']:
            st.markdown(f"**환자명: {inputs['patient_name']}**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"### 🎯 가장 가능성 높은 진단: **{DISEASE_NAMES_KR[top_disease]}**")
            st.metric("예측 확률", f"{top_prob*100:.1f}%")
        
        with col2:
            st.markdown("### 각 질환별 확률")
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for disease, prob in sorted_probs:
                bar_color = "🟢" if disease == top_disease else "⚪"
                st.write(f"{bar_color} **{disease}**: {prob*100:.1f}%")
        
        st.markdown("---")
        
        st.header("🔬 변수 기여도 분석 (SHAP)")
        st.markdown(f"**{DISEASE_NAMES_KR[top_disease]}** 예측에 각 변수가 어떻게 기여했는지 보여줍니다.")
        
        with st.spinner("SHAP 분석 중..."):
            fig_waterfall, fig_bar = generate_shap_plot(models[top_disease], input_df, top_disease)
            
            if fig_waterfall and fig_bar:
                with st.expander("📊 개별 예측 기여도 (Top 10)", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        buf1 = io.BytesIO()
                        fig_waterfall.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
                        buf1.seek(0)
                        plt.close(fig_waterfall)
                        st.image(buf1, use_container_width=True)
                    with col2:
                        st.markdown("""
                        **📖 해석 가이드**
                        
                        - 🔴 **빨간색**: 해당 질환 예측 확률 ↑
                        - 🔵 **파란색**: 해당 질환 예측 확률 ↓
                        - **막대 길이**: 변수의 영향력 크기
                        - **f(x)**: 최종 예측 점수
                        """)
                
                with st.expander("📈 변수 중요도 (Top 20)", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        buf2 = io.BytesIO()
                        fig_bar.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
                        buf2.seek(0)
                        plt.close(fig_bar)
                        st.image(buf2, use_container_width=True)
                    with col2:
                        st.markdown("""
                        **📖 해석 가이드**
                        
                        - **막대 길이**: 변수의 평균 영향력
                        - 상위 변수일수록 예측에 중요
                        """)
        
        with st.expander("📋 입력된 환자 정보 요약"):
            summary_data = {
                "환자 이름": inputs['patient_name'] if inputs['patient_name'] else "(미입력)",
                "나이": inputs['age'],
                "성별": "여성" if inputs['sex'] == 1 else "남성",
                "회전성 어지럼증": "예" if inputs['symptoms_true_vertigo'] else "아니오",
                "청력 저하": "예" if inputs['symptoms_hearing_impairment_combined'] else "아니오",
                "이명": "예" if inputs['symptoms_tinnitus'] else "아니오",
            }
            st.json(summary_data)
    
    else:
        st.info("👈 왼쪽 사이드바에서 환자 정보를 입력하고 '예측 실행' 버튼을 눌러주세요.")
        
        with st.expander("ℹ️ 사용 안내"):
            st.markdown("""
            ### 시스템 설명
            이 시스템은 어지럼증 환자의 증상, 과거력, 신체검사 소견을 바탕으로 
            5가지 주요 원인 질환을 감별진단하는 AI 예측 모델입니다.
            
            ### 예측 가능한 질환
            - **BPPV** (양성돌발성체위현훈)
            - **VN** (전정신경염)
            - **SSNHL** (돌발성난청)
            - **Meniere** (메니에르병)
            - **Others** (기타 원인)
            
            ### 주의사항
            - 이 시스템은 임상 의사결정 보조 도구입니다.
            - 최종 진단은 반드시 전문의의 판단에 따라야 합니다.
            """)

if __name__ == "__main__":
    main()
