import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import openai
import sys
import os
import io

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.analysis.categorize import run_keyword_analysis, generate_wordcloud_from_freq
from modules.analysis.summary_module import generate_summary_with_gpt
from modules.analysis.sentiment_module import (
    analyze_sentiment_with_finbert,
    refine_neutral_keywords_with_gpt,
    merge_sentiment_results,
    summarize_sentiment_by_category
)
from modules.analysis_pipeline import AnalysisPipeline
from langchain.chat_models import ChatOpenAI

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

# 페이지 선택
menu = st.sidebar.selectbox("페이지 선택", ["🏠 홈", "📊 분석", "⚙️ 설정"])

# 홈 페이지
if menu == "🏠 홈":
    st.title("💼 대상자 & 관계 기반 HR 응답 분석 대시보드")
    uploaded = st.file_uploader("📂 엑셀 파일 업로드", type=["xlsx", "xls"])

    if uploaded:
        df = pd.read_excel(uploaded)
        st.success("업로드 완료!")
        st.dataframe(df)

        # AI로 컬럼/로우 데이터 분석 
        columns = df.columns.tolist()
        none_option = "❌ 선택 안함"

        name_col = st.selectbox("🧾 대상자 컬럼을 선택하세요(없으면 '선택 안함')", [none_option] + columns)
        relation_col = st.selectbox("🔗 관계 컬럼을 선택하세요(없으면 '선택 안함')", [none_option] + columns)
        text_col = st.selectbox("📝 텍스트 컬럼을 선택하세요", columns)

        if text_col:
            filtered_df = df.copy()

            # 대상자 선택
            if name_col != none_option:
                name_values = df[name_col].dropna().unique()
                selected_name = st.selectbox("👤 대상자 선택", name_values)
                filtered_df = filtered_df[filtered_df[name_col] == selected_name]
                st.spinner("분석 중...")
                
                
            # 텍스트 리스트 추출 (dropna 후 시리즈 형태 유지)
            text_series = filtered_df[text_col].dropna().astype(str)
            texts = text_series.tolist()

            if texts:
                freq_df, categorized_df = run_keyword_analysis(texts, llm)
                # extracted_keywords = freq_df["keyword"].unique().tolist() # 리스트 변환
                # freq_dict = pd.Series(extracted_keywords).value_counts().to_dict()
                
                col1, col2 = st.columns(2)
                
                # ☁️ GPT 키워드 기반 워드클라우드
                with col1:
                    st.subheader("☁️ GPT 키워드 기반 워드클라우드")
                    wc = generate_wordcloud_from_freq(freq_df)
                    if wc:
                        st.success("워드클라우드 생성!")
                        fig, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.warning("워드클라우드를 생성할 수 없습니다.")

                # 📊 GPT 키워드 기반 빈도 막대그래프
                with col2:
                    st.subheader("📊 GPT 키워드 빈도 상위 20개")
                    st.success("키워드 빈도 분석 완료!")
                    freq_df["count"] = freq_df["count"].astype(int)
                    freq_plot_df = freq_df.sort_values(by="count", ascending=False).head(20)
                    fig2, ax2 = plt.subplots()
                    sns.barplot(data=freq_plot_df, y='keyword', x='count', hue='category', dodge=False, ax=ax2)
                    # xlabel 정수
                    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
                    
                    ax2.set_ylabel("키워드")
                    ax2.set_xlabel("count")
                    st.pyplot(fig2)
                    
                
                # 감정 분석
                st.subheader("❤️ 감정 분석 결과")
                with st.spinner("감정 분석 중..."):
                    
                    # 1. 감정 분석 처리
                    sentiment_df, _, _ = analyze_sentiment_with_finbert(texts, llm)
                    refined_df = refine_neutral_keywords_with_gpt(sentiment_df, llm)
                    updated_df = merge_sentiment_results(sentiment_df, refined_df)
                    summary = summarize_sentiment_by_category(freq_df, updated_df)
                    
                    # # 2. 감정 분포 시각화
                    # sentiment_counts = updated_df['sentiment'].value_counts()
                    # fig1, ax1 = plt.subplots(figsize=(5, 5))
                    # sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax1)
                    # ax1.set_title("감정 분석 결과 분포")
                    # ax1.set_xlabel("감정 유형")
                    # ax1.set_ylabel("응답 수")
                    # st.pyplot(fig1)

                    # 3. 키워드별 감정 비율 시각화(Plotly PieChart 시각화)
                    overall_sentiment = summary.groupby('sentiment')['percentage'].sum().reset_index()

                    fig = px.pie(
                        overall_sentiment,
                        names='sentiment',
                        values='percentage',
                        title='전체 감정 분포 (모든 키워드 기준)',
                        color='sentiment',
                        color_discrete_map={'긍정': '#63b2ee', '부정': '#ff9999', '중립': '#ffcc66'}
                    )

                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig)
                                        
                # GPT 요약
                st.subheader("🧠 GPT 요약 결과")
                with st.spinner("요약 중..."):
                    summary = generate_summary_with_gpt(texts)
                    st.success("요약 완료!")
                    st.write(summary)
                    
            else:
                st.warning("분석할 텍스트가 없습니다.")
        else:
            st.warning("텍스트 컬럼을 찾을 수 없습니다.")

# ✅ 추후 확장용 페이지
elif menu == "📊 분석":
    st.title("📊 분석 기능")
    st.write("추후 분석 기능이 추가될 예정입니다.")

elif menu == "⚙️ 설정":
    st.title("⚙️ 설정")
    st.write("API 키 등 설정 가능")
