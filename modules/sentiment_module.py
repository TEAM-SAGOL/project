# sentiment_module.py

import pandas as pd
import matplotlib.pyplot as plt
import openai
import streamlit as st

# ✅ 감정 분석 함수
def analyze_sentiment(text):
    client = openai.OpenAI(api_key=st.secrets["your_section"]["api_key"])

    prompt = f"""다음 문장의 감정은 긍정, 부정, 중립 중 무엇인가요? 감정만 한 단어로 답해주세요:\n\n{text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "오류"

# ✅ 감정 분포 시각화
def plot_sentiment_distribution(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig, ax = plt.subplots(figsize=(5,5))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
    ax.set_title("감정 분석 결과 분포")
    ax.set_xlabel("감정 유형")
    ax.set_ylabel("응답 수")
    st.pyplot(fig)
