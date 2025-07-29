import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from langchain.chat_models import ChatOpenAI
from modules.categorize import run_keyword_analysis, generate_wordcloud_from_freq
from modules.summary_module import generate_summary_with_gpt
from modules.sentiment_module import analyze_sentiment, plot_sentiment_distribution


# ✅ 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 기본 설정
st.set_page_config(page_title="HR 응답 분석", layout="wide")
client = openai.OpenAI(api_key=st.secrets["your_section"]["api_key"])
llm = ChatOpenAI(api_key=st.secrets["your_section"]["api_key"], model="gpt-4o-mini", temperature=0)


# ✅ 페이지 선택
menu = st.sidebar.selectbox("페이지 선택", ["🏠 홈", "📊 분석", "⚙️ 설정"])

# ✅ 홈 페이지
if menu == "🏠 홈":
    st.title("💼 대상자 & 관계 기반 HR 응답 분석 대시보드")
    uploaded = st.file_uploader("📂 엑셀 파일 업로드", type=["xlsx"])

    if uploaded:
        df = pd.read_excel(uploaded)
        st.success("업로드 완료!")
        st.dataframe(df)

        # ⬇️ 사용자에게 컬럼 선택 UI 제공
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
                
                
            # 텍스트 리스트 추출 (dropna 후 시리즈 형태 유지)
            text_series = filtered_df[text_col].dropna().astype(str)
            texts = text_series.tolist()

            if texts:
                keyword_results = run_keyword_analysis(texts, llm)  
                extracted_keywords = keyword_results["keywords"]
                freq_dict = pd.Series(extracted_keywords).value_counts().to_dict()
                
                col1, col2 = st.columns(2)
                
                # 키워드 추출   
                with col1:
                    st.subheader("☁️ 워드클라우드")
                    words = [w for t in texts for w in t.split()]
                    freq_dict = pd.Series(words).value_counts().to_dict()
                    wc = generate_wordcloud_from_freq(freq_dict)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.subheader("📊 단어 빈도 막대그래프")
                    words = [w for t in texts for w in t.split()]
                    freq = pd.Series(words).value_counts().reset_index()
                    freq.columns = ['keyword', 'count']
                    fig2, ax2 = plt.subplots()
                    sns.barplot(data=freq.head(20), y='keyword', x='count', ax=ax2)
                    st.pyplot(fig2)

                # 감정 분석
                st.subheader("❤️ 감정 분석 결과")
                with st.spinner("감정 분석 중..."):
                    sentiments = [analyze_sentiment(text) for text in text_series]
                    filtered_df.loc[text_series.index, "감정"] = sentiments
                    plot_sentiment_distribution(sentiments)

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
