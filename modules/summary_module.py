# summary_module.py

import openai
import streamlit as st

# ✅ GPT 요약 함수
def generate_summary_with_gpt(texts):
    client = openai.OpenAI(api_key=st.secrets["your_section"]["api_key"])

    prompt = f"""다음은 구성원에 대한 응답입니다. 자주 등장하는 키워드와 전반적인 분위기를 바탕으로  
    💡1. 긍정적인 피드백  
    🛠️2. 개선점  
    👥3. 구성원에 대한 팀 내 인식  
    을 요약해 주세요.  
    응답 샘플 (최대 50개):
    {texts[:50]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 조직 심리 분석 전문가야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ GPT 요약 실패: {e}"

