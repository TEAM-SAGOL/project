import streamlit as st
import pandas as pd
import sys
import os
import io
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from modules.categorize import generate_wordcloud_from_freq
from langchain_community.chat_models import ChatOpenAI
from modules.analysis_pipeline import AnalysisPipeline

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 기본 설정
st.set_page_config(page_title="HR 응답 분석", layout="wide")
# client = openai.OpenAI(api_key=st.secrets["your_section"]["api_key"])

# GPT 모델 정의
llm = ChatOpenAI(
    model="gpt-4o-mini",            
    temperature=0,
    openai_api_key=st.secrets["your_section"]["api_key"]  
)


# ✅ 사이드바 - 진행 상황 표시 및 페이지 선택
with st.sidebar:
    # 🔧 페이지 이동
    menu = st.selectbox("📌 페이지 선택", ["🏠 홈", "📊 분석", "⚙️ 설정"])
    st.header("📋 분석 진행 단계")

    # 진행 단계 정의
    steps = [
        "1️⃣ 기획서 생성",
        "2️⃣ 기획서 검토",
        "3️⃣ 분석 실행",
        "4️⃣ 결과 확인"
    ]

    # 현재 단계 세션에서 가져오기 (기본: 1)
    current_step = st.session_state.get('current_step', 1)

    # 단계별 상태 표시
    for i, step in enumerate(steps, 1):
        if i == current_step:
            st.success(f"**{step}** ← 현재 단계")
        elif i < current_step:
            st.info(f"~~{step}~~ ✅")
        else:
            st.write(step)

    st.markdown("---")

    # 🔁 세션 초기화 버튼
    if st.button("🔄 처음부터 다시 시작", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# 홈 페이지
if menu == "🏠 홈":
    st.title("🔍 AI 데이터 분석 대시보드")
    st.markdown("---")
    
    #파이프라인 초기화
    pipeline = AnalysisPipeline(llm=llm)
    
    # 파일 업로드
    uploaded = st.file_uploader(
        "## 📊 Excel 파일을 업로드하세요",
        type=['xlsx', 'xls'],
        help="분석할 데이터가 포함된 Excel 파일을 선택해주세요."
    )

    if uploaded:
        df = pd.read_excel(uploaded, index_col=None)
        st.success("업로드 완료!")
        # Unnamed 컬럼 제거
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # 데이터 타입 정리 (Arrow 호환성)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # 혼재된 타입을 문자열로 통일
                    df[col] = df[col].astype(str)
                    # NaN 값을 빈 문자열로 변경
                    df[col] = df[col].replace('nan', '')
                except:
                    continue
        st.dataframe(df)
        st.markdown('---')
        
        # 데이터 기본 정보 표시
        with st.container():
            st.markdown("### 📊 데이터 기본 정보")
            col1, col2, col3 = st.columns(3)

            with col1:
                with st.container():
                    st.metric("행 수", f"{len(df):,}")
            with col2:
                with st.container():
                    st.metric("컬럼 수", len(df.columns))
            with col3:
                with st.container():
                    st.metric("데이터 크기", f"{df.shape[0] * df.shape[1]:,} cells")
    
        # 데이터 설명 입력
            description = st.text_area(
                "📝 데이터에 대한 설명을 입력해주세요",
                placeholder="예: 2024년 직원 만족도 설문조사 결과입니다. 직원들의 업무 환경, 복지, 커뮤니케이션에 대한 의견이 포함되어 있습니다.", #버튼 없어두 되나
                height=100,
                help="AI가 데이터를 이해하고 적절한 분석 계획을 세우는 데 도움이 됩니다."
            )
            if description.strip():
                st.markdown("---")
                
                # 단계별 실행
                current_step = st.session_state.get('current_step', 1)
                
                if current_step == 1:
                    # 1단계: 분석 기획서 생성
                    pipeline.step1_generate_plan(df, description)
                
                elif current_step == 2:
                    # 2단계: 기획서 검토
                    pipeline.step2_review_plan()
                
                elif current_step == 3:
                    # 3단계: 분석 실행
                    pipeline.step3_execute_analysis(df)
                
                elif current_step == 4:
                    # 4단계: 결과 표시
                    pipeline.step4_display_results()

                results = st.session_state.get("analysis_results", {})

                if results:
                    # 1️⃣ 키워드 분석 결과 시각화
                    if "keyword_analysis" in results:
                        keyword_result = results["keyword_analysis"]
                        df_kw = keyword_result.get("freq_df")       # 수정: freq_df를 바로 가져옴
                        keywords = keyword_result.get("keywords")   # 필요시 keyword 리스트로도 저장 가능

                        if isinstance(df_kw, pd.DataFrame) and {'keyword', 'category', 'count'}.issubset(df_kw.columns):
                            st.subheader("☁️ GPT 키워드 기반 워드클라우드")

                            # 워드클라우드 
                            wc = generate_wordcloud_from_freq(df_kw)
                            if wc:
                                fig, ax = plt.subplots()
                                ax.imshow(wc, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            else:
                                st.warning("워드클라우드를 생성할 수 없습니다.")
                        else:
                            st.warning("키워드 데이터에 'keyword' 또는 'category' 컬럼이 없습니다.")

                            # 키워드 빈도 막대 그래프
                            st.subheader("📊 GPT 키워드 빈도 상위 20개")

                            if freq_df is not None and {'keyword', 'category', 'count'}.issubset(freq_df.columns):
                                top20 = freq_df.sort_values(by="count", ascending=False).head(20)

                                fig2, ax2 = plt.subplots()
                                sns.barplot(data=top20, y='keyword', x='count', hue='category', dodge=False, ax=ax2)
                                ax2.set_ylabel("키워드")
                                ax2.set_xlabel("count")
                                st.pyplot(fig2)
                            else:
                                st.warning("키워드 데이터에 필수 컬럼이 없습니다.")
                    else:
                        st.warning("키워드 리스트가 비어 있습니다.")

                    # 2️⃣ 요약 분석 결과 시각화
                    if "summary_analysis" in results:
                        summary_text = results["summary_analysis"].get("summary", "")
                        if summary_text:
                            st.subheader("🧠 GPT 요약 결과")
                            st.success("요약 완료!")
                            st.write(summary_text)
                        else:
                            st.warning("요약된 내용이 없습니다.")

                    # 3️⃣ 감정 분석 시각화
                    if "sentiment_analysis" in results: #예외 처리 발생함..
                        sentiment_result = results["sentiment_analysis"]
                        summary_df = sentiment_result.get("summary_df")

                        if summary_df:
                            summary_df = pd.DataFrame(summary_df)

                            if not summary_df.empty and 'sentiment' in summary_df.columns:
                                st.subheader("❤️ 감정 분석 결과")
                                overall_sentiment = summary_df.groupby('sentiment')['percentage'].sum().reset_index()

                                fig = px.pie(
                                    overall_sentiment,
                                    names='sentiment',
                                    values='percentage',
                                    title='전체 감정 분포',
                                    color='sentiment',
                                    color_discrete_map={'긍정': '#63b2ee', '부정': '#ff9999', '중립': '#ffcc66'}
                                )
                                fig.update_traces(textinfo='percent+label')
                                st.plotly_chart(fig)
                                
                            else:
                                st.warning("감정 분석 결과가 비어있거나 형식이 맞지 않습니다.")
                        else:
                            st.warning("감정 분석 요약 데이터가 없습니다.")
                else:
                    st.info("👆 데이터에 대한 설명을 입력하면 AI 분석을 시작할 수 있습니다.")
                    
    else:
        # 업로드 안내
        st.info("📁 Excel 파일을 업로드하여 AI 기반 데이터 분석을 시작하세요!")
        st.markdown("---")
        
        with st.container():
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
            
            st.markdown("---")
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
        
                
# ✅ 추후 확장용 페이지
elif menu == "📊 분석":
    st.title("📊 분석 기능")
    st.write("추후 분석 기능이 추가될 예정입니다.")

elif menu == "⚙️ 설정":
    st.title("⚙️ 설정")
    st.write("API 키 등 설정 가능")
    
    