import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline
from langchain.schema import HumanMessage
from modules.categorize import run_keyword_analysis 

# 0단계: 모델 로딩 (FinBERT)
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

sentiment_map = {
    'positive': '긍정',
    'negative': '부정',
    'neutral': '중립'
}

# 1단계: 키워드 감정 분석 (FinBERT + 룰 기반 후처리)
def analyze_sentiment_with_finbert(texts, llm):
    freq_df, categorized_df = run_keyword_analysis(texts, llm)
    unique_keywords = freq_df["keyword"].tolist()

    results = classifier(unique_keywords)

    def map_sentiment_label(result_label, text):
        positive_keywords = ['긍정적', '적극', '모범적', '솔선수범', '개선']
        if any(keyword in text for keyword in positive_keywords):
            return '긍정'
        if '부족' in text:
            return '부정'
        return sentiment_map.get(result_label, '알 수 없음')

    sentiment_df = pd.DataFrame({
        "keyword": unique_keywords,
        "sentiment": [map_sentiment_label(res['label'], kw) for res, kw in zip(results, unique_keywords)],
        "confidence": [round(res['score'], 3) for res in results],
        "category": [categorized_df.get(kw, "기타") for kw in unique_keywords]
    })

    return sentiment_df, freq_df, categorized_df

# 2단계: 중립 키워드만 GPT로 재분석
def refine_neutral_keywords_with_gpt(sentiment_df, llm):
    neutral_keywords = sentiment_df[sentiment_df['sentiment'] == '중립']['keyword'].tolist()
    keyword_sentiments = []

    for keyword in neutral_keywords:
        prompt = f"""
        다음 키워드는 사용자 응답에서 추출된 핵심 키워드입니다.
        이 키워드가 담고 있는 감정을 긍정/부정/중립 중 하나로 판단해 주세요.

        키워드: "{keyword}"

        - 긍정이면 1, 부정이면 0, 중립이면 2만 출력해 주세요.
        """
        resp = llm([HumanMessage(content=prompt)])
        try:
            sentiment = int(resp.content.strip())
            sentiment = {1: '긍정', 0: '부정', 2: '중립'}.get(sentiment, '중립')
        except:
            sentiment = '중립'

        keyword_sentiments.append({
            'keyword': keyword,
            'sentiment': sentiment
        })

    refined_df = pd.DataFrame(keyword_sentiments)
    return refined_df

# 3단계: 기존 + 재분류된 중립 감정 통합
def merge_sentiment_results(original_df, refined_df):
    # 중립 키워드의 기존 category 가져오기
    neutral_original = original_df[original_df['sentiment'] == '중립'][['keyword', 'category']]

    # refined_df와 category 병합
    refined_df = pd.merge(neutral_original, refined_df, on='keyword', how='left')

    # 재분류 안 된 중립 키워드
    remaining_neutral = neutral_original[~neutral_original['keyword'].isin(refined_df['keyword'])]
    remaining_neutral['sentiment'] = '중립'

    # 최종 통합
    updated_df = pd.concat([
        original_df[original_df['sentiment'] != '중립'][['keyword', 'sentiment']],
        refined_df[['keyword','sentiment']],
        remaining_neutral
    ], ignore_index=True)

    return updated_df

# 4단계: 카테고리별 감정 비율 요약
def summarize_sentiment_by_category(freq_df, sentiment_df):
    sentiment_df = sentiment_df.drop(columns=['confidence'], errors='ignore')
    # 1. freq_df와 sentiment_df 병합
    merged_df = pd.merge(freq_df, sentiment_df, on='keyword', how='left')
    # 2. 카테고리별 감정별 count 집계
    summary = (
        merged_df
        .groupby(['keyword', 'sentiment'])['count']
        .sum()
        .reset_index()
    )
    # 3. 카테고리별 전체 count 계산 후 merge
    total = summary.groupby('keyword')['count'].transform('sum')
    # 4. 비율 계산
    summary['percentage'] = (summary['count'] / total * 100).round(2)
    
    return summary

