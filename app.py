import streamlit as st
import pandas as pd
import sys
import os
import io

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.analysis_pipeline import AnalysisPipeline

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="🔍 AI 데이터 분석 대시보드",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 AI 데이터 분석 대시보드")
    st.markdown("---")
    
    # 파이프라인 초기화
    pipeline = AnalysisPipeline()
    
    # 사이드바 - 진행 상황 표시
    with st.sidebar:
        st.header("📋 분석 진행 단계")
        
        current_step = st.session_state.get('current_step', 1)
        
        # 단계별 표시
        steps = [
            "1️⃣ 기획서 생성",
            "2️⃣ 기획서 검토",
            "3️⃣ 분석 실행",
            "4️⃣ 결과 확인"
        ]
        
        for i, step in enumerate(steps, 1):
            if i == current_step:
                st.success(f"**{step}** ← 현재 단계")
            elif i < current_step:
                st.info(f"~~{step}~~ ✅")
            else:
                st.write(step)
        
        st.markdown("---")
        
        # 세션 초기화 버튼
        if st.button("🔄 처음부터 다시 시작", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # 메인 콘텐츠
    
    # 데이터 업로드 섹션 (항상 표시)
    st.header("📂 데이터 업로드")
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "📊 Excel 파일을 업로드하세요",
        type=['xlsx', 'xls'],
        help="분석할 데이터가 포함된 Excel 파일을 선택해주세요."
    )
    
    if uploaded_file is not None:
        try:
            # 데이터 로드 (인덱스 컬럼 제거)
            data = pd.read_excel(uploaded_file, index_col=None)
            
            # Unnamed 컬럼 제거
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            
            # 데이터 타입 정리 (Arrow 호환성)
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        # 혼재된 타입을 문자열로 통일
                        data[col] = data[col].astype(str)
                        # NaN 값을 빈 문자열로 변경
                        data[col] = data[col].replace('nan', '')
                    except:
                        continue
            
            st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
            
            # 데이터 기본 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("행 수", f"{len(data):,}")
            with col2:
                st.metric("컬럼 수", len(data.columns))
            with col3:
                st.metric("데이터 크기", f"{data.shape[0] * data.shape[1]:,} cells")
            
            # 데이터 설명 입력
            description = st.text_area(
                "📝 데이터에 대한 설명을 입력해주세요",
                placeholder="예: 2024년 직원 만족도 설문조사 결과입니다. 직원들의 업무 환경, 복지, 커뮤니케이션에 대한 의견이 포함되어 있습니다.",
                height=100,
                help="AI가 데이터를 이해하고 적절한 분석 계획을 세우는 데 도움이 됩니다."
            )
            
            if description.strip():
                st.markdown("---")
                
                # 단계별 실행
                current_step = st.session_state.get('current_step', 1)
                
                if current_step == 1:
                    # 1단계: 분석 기획서 생성
                    pipeline.step1_generate_plan(data, description)
                
                elif current_step == 2:
                    # 2단계: 기획서 검토
                    pipeline.step2_review_plan()
                
                elif current_step == 3:
                    # 3단계: 분석 실행
                    pipeline.step3_execute_analysis(data)
                
                elif current_step == 4:
                    # 4단계: 결과 표시
                    pipeline.step4_display_results()
            
            else:
                st.info("👆 데이터에 대한 설명을 입력하면 AI 분석을 시작할 수 있습니다.")
        
        except Exception as e:
            st.error(f"파일 로드 중 오류가 발생했습니다: {str(e)}")
            st.info("Excel 파일 형식을 확인해주세요.")
    
    else:
        # 업로드 안내
        st.info("📁 Excel 파일을 업로드하여 AI 기반 데이터 분석을 시작하세요!")
        
        # 샘플 데이터 다운로드 제공
        st.write("### 📋 사용 방법")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**1단계: 데이터 업로드**")
            st.write("- Excel 파일(.xlsx, .xls) 업로드")
            st.write("- 텍스트 데이터가 포함된 컬럼 필요")
            
            st.write("**2단계: 데이터 설명**")
            st.write("- 데이터의 목적과 내용 설명")
            st.write("- AI가 분석 계획 수립에 활용")
        
        with col2:
            st.write("**3단계: AI 분석 계획**")
            st.write("- AI가 자동으로 분석 기획서 생성")
            st.write("- 사용자가 검토 및 수정 가능")
            
            st.write("**4단계: 자동 분석**")
            st.write("- 감정 분석, 키워드 추출, 요약 등")
            st.write("- 결과를 시각화로 제공")
        
        # 샘플 데이터 예시
        st.write("### 📄 샘플 데이터 형식")
        
        sample_data = pd.DataFrame({
            '이름': ['홍길동', '김철수', '이영희'],
            '부서': ['개발팀', '마케팅팀', '인사팀'],
            '만족도': [4, 3, 5],
            '의견': [
                '업무 환경이 좋습니다. 동료들과의 협업도 원활해요.',
                '워라밸이 개선되었으면 좋겠습니다.',
                '복지 제도가 만족스럽고 회사 분위기가 좋아요.'
            ]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        # 샘플 파일 다운로드 버튼
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                sample_data.to_excel(writer, index=False, sheet_name='Sample Data')

            excel_data = output.getvalue()

            st.download_button(
                label = "📥 샘플 데이터 다운로드",
                data = excel_data,
                file_name = "sample_survey_data.xlsx",
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"샘플 데이터 다운로드 중 오류 발생: {e}")

    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 14px;'>
            🤖 AI 기반 데이터 분석 대시보드 | 
            Built with Streamlit & OpenAI GPT-4
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
