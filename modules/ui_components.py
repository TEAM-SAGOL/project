import streamlit as st
import pandas as pd
from typing import Dict, Any

def display_analysis_plan(plan: Dict[str, Any]):
    """분석 기획서를 스트림릿에 표시"""
    
    st.subheader("📋 분석 기획서")
    
    # 1. 데이터 주제
    st.write("### 🎯 데이터 주제")
    st.write(plan["data_subject"]) # 내용 짤림 
    
    # 2. 컬럼 분석
    st.write("### 📊 컬럼 분석")
    col_df = pd.DataFrame.from_dict(plan["column_analysis"], orient='index')
    if not col_df.empty:
        st.dataframe(col_df)
    else: 
        st.info("컬럼 분석 정보가 없습니다.")
    
    # 3. 목표 인사이트
    st.write("### 💡 목표 인사이트")
    for i, insight in enumerate(plan["target_insights"], 1):
        st.write(f"{i}. {insight}")
    
    # 4. 추천 모듈
    st.write("### 🔧 추천 분석 모듈")
    modules = plan["recommended_modules"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**감정 분석**")
        if modules["sentiment_analysis"]["use"]:
            st.success("✅ 사용")
            st.write(f"대상: {', '.join(modules['sentiment_analysis']['target_columns'])}")
            st.write(f"이유: {modules['sentiment_analysis']['reason']}")
        else:
            st.info("❌ 미사용")
    
    with col2:
        st.write("**키워드 분석**")
        if modules["keyword_analysis"]["use"]:
            st.success("✅ 사용")
            st.write(f"대상: {', '.join(modules['keyword_analysis']['target_columns'])}")
            st.write(f"이유: {modules['keyword_analysis']['reason']}")
        else:
            st.info("❌ 미사용")
    
    with col3:
        st.write("**요약 분석**")
        if modules["summary_analysis"]["use"]:
            st.success("✅ 사용")
            st.write(f"대상: {', '.join(modules['summary_analysis']['target_columns'])}")
            st.write(f"이유: {modules['summary_analysis']['reason']}")
        else:
            st.info("❌ 미사용")
    
    # 5. 분석 워크플로우
    if "analysis_workflow" in plan:
        st.write("### 🔄 분석 워크플로우")
        for step in plan["analysis_workflow"]:
            st.write(f"- {step}")

def display_plan_review_interface(plan: Dict[str, Any]) -> bool:
    """기획서 검토 인터페이스"""
    
    # 검증
    from .profiling_module import validate_plan
    issues = validate_plan(plan)
    
    if issues:
        st.error("다음 문제를 해결해주세요:")
        for issue in issues:
            st.write(f"- {issue}")
        return False
    else:
        st.success("✅ 기획서 검증 완료!")
        return True

def feedback_input_interface() -> str:
    """피드백 입력 인터페이스"""
    
    feedback = st.text_area(
        "📝 수정 요청사항 (자연어로 입력)",
        placeholder="예: 감정 분석을 추가하고, 키워드 분석에서 '의견' 컬럼도 포함해주세요",
        height=100,
        help="자연어로 원하는 수정사항을 입력하면 AI가 기획서를 수정합니다."
    )
    
    return feedback