# import json

# if __name__ == '__main__':
#     dict = [{'categorize':1}, {'categorize':1}, {'categorize':1}]
#     val = json.dumps(dict)
#     print(type(val))
#     print(val)
    
    
# import streamlit as st
# import pandas as pd
# from langchain.chat_models import ChatOpenAI
# from modules.categorize import run_keyword_analysis
# from modules.sentiment_module import (
#     analyze_sentiment_with_finbert,
#     refine_neutral_keywords_with_gpt,
#     merge_sentiment_results
# )

# # ✅ 임시 데이터
# texts = [
#     "리더십이 뛰어나고 소통을 잘함",
#     "협업 태도가 부족하고 개선이 필요함",
#     "팀워크가 좋고 책임감이 강함",
#     "소극적인 태도가 아쉬움"
# ]

# # ✅ GPT 설정 (secrets.toml 필요)
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     openai_api_key=st.secrets["your_section"]["api_key"]
# )

# # ✅ 분석 실행
# freq_df, categorized_df = run_keyword_analysis(texts, llm)
# sentiment_df, _, _ = analyze_sentiment_with_finbert(freq_df, categorized)
# refined_df = refine_neutral_keywords_with_gpt(sentiment_df, llm)
# updated_df = merge_sentiment_results(sentiment_df, refined_df)

# # ✅ 컬럼 확인 출력
# st.write("✅ freq_df 컬럼:", freq_df.columns.tolist())
# st.write("✅ updated_df 컬럼:", updated_df.columns.tolist())


