import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import json
from datetime import datetime
import random
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from modules.profiling_module import AnalysisPlan, validate_plan
from modules.ui_components import display_analysis_plan, display_plan_review_interface, feedback_input_interface
from modules.categorize import generate_wordcloud_from_freq, categorize_keywords_batch
from modules.summary_module import generate_summary_with_gpt 
from modules.sentiment_module import merge_sentiment_results, refine_neutral_keywords_with_gpt, analyze_sentiment_with_finbert, summarize_sentiment_by_category

class AnalysisPipeline:
    """분석 파이프라인 관리 클래스"""
    
    def __init__(self):
        self.planner = AnalysisPlan()
        self.reset_session_state()
    
    def reset_session_state(self):
        """세션 상태 초기화"""
        if 'analysis_plan' not in st.session_state:
            st.session_state.analysis_plan = None
        if 'plan_approved' not in st.session_state:
            st.session_state.plan_approved = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
            
    def _display_summary_results(self, summary_result: dict) -> str:
        return summary_result.get("summary", "")

    def _display_sentiment_results(self, sentiment_result: dict) -> pd.DataFrame:
        return sentiment_result.get("summary_df")

    def _display_keyword_results(self, keyword_result: dict) -> pd.DataFrame:
        freq_df = keyword_result.get("freq_df")  # 'freq_df' 키로 저장된 데이터를 가져옴
        if isinstance(freq_df, pd.DataFrame):
            return freq_df
        return pd.DataFrame()
    
    def step1_generate_plan(self, data: pd.DataFrame, description: str):
        """1단계: 분석 기획서 생성"""
        st.header("1️⃣ 스마트 AI 분석 기획서 생성")  # 제목 변경
        
        # col1, col2 = st.columns([2, 1])
        
        # with col1:
        #     st.write("**데이터 미리보기**")
        #     st.dataframe(data.head())
        
        # with col2:
        #     st.write("**데이터 정보**")
        #     st.write(f"- 행 수: {len(data):,}")
        #     st.write(f"- 컬럼 수: {len(data_subjectdata.columns)}")
        #     st.write(f"- 컬럼: {', '.join(data.columns[:3])}{'...' if len(data.columns) > 3 else ''}")
        
        # st.write("**데이터 설명**")
        # st.write(description)
        
        # 기획서 생성 버튼
        if st.button("🤖 AI 분석 계획 생성 (강화버전)", type="primary", use_container_width=True):  # 버튼명 변경
            with st.spinner("🧠 AI가 데이터를 심층 분석하여 맞춤형 계획을 수립 중..."):  # 메시지 변경
                plan = self.planner.generate_initial_plan(data, description)
                st.session_state.analysis_plan = plan
                st.session_state.plan_approved = False
                st.session_state.current_step = 2
                st.rerun()
        
        # 기획서가 있으면 표시
        if st.session_state.analysis_plan:
            st.write("---")
            display_analysis_plan(st.session_state.analysis_plan)
            return True
        return False
    
    def step2_review_plan(self):
        """2단계: 기획서 검토 및 수정"""
        if not st.session_state.analysis_plan:
            st.warning("먼저 분석 기획서를 생성해주세요.")
            return False
            
        st.header("2️⃣ 기획서 검토 및 수정")
        
        # 현재 기획서 표시
        display_analysis_plan(st.session_state.analysis_plan)
        
        st.write("---")
        
        # 검증 결과 표시
        validation_passed = display_plan_review_interface(st.session_state.analysis_plan)
        
        # 피드백 입력 섹션
        st.write("### 📝 기획서 수정")
        feedback = feedback_input_interface()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✏️ 기획서 수정", use_container_width=True) and feedback:
                with st.spinner("피드백을 반영하여 기획서를 수정중입니다..."):
                    revised_plan = self.planner.revise_plan_with_feedback(
                        st.session_state.analysis_plan, feedback
                    )
                    st.session_state.analysis_plan = revised_plan
                    st.rerun()
        
        with col2:
            if st.button("🔄 기획서 재생성", use_container_width=True):
                st.session_state.analysis_plan = None
                st.session_state.current_step = 1
                st.rerun()
        
        with col3:
            approved = st.button("✅ 기획서 승인", type="primary", use_container_width=True)
                
        # ✅ 한 줄 아래에 승인 메시지 출력
        if approved and validation_passed:
            st.session_state.plan_approved = True
            st.session_state.current_step = 3
            st.success("🎉 기획서가 승인되었습니다! 분석을 진행할 수 있습니다.")
            st.rerun()
            
        return st.session_state.plan_approved
    
    def step3_execute_analysis(self, data: pd.DataFrame):
        """3단계: 분석 실행"""
        if not st.session_state.plan_approved:
            st.warning("먼저 기획서를 승인해주세요.")
            return False
            
        st.header("3️⃣ 분석 실행")
        
        # 분석 실행 정보 표시
        plan = st.session_state.analysis_plan
        modules = plan["recommended_modules"]
        
        st.write("**실행 예정 분석**")
        execution_list = []
        if modules["sentiment_analysis"]["use"]:
            execution_list.append("😊 감정 분석")
        if modules["keyword_analysis"]["use"]:
            execution_list.append("🔤 키워드 분석")
        if modules["summary_analysis"]["use"]:
            execution_list.append("📝 요약 분석")
        
        for item in execution_list:
            st.write(f"- {item}")
        
        if not execution_list:
            st.error("실행할 분석이 없습니다. 기획서를 다시 검토해주세요.")
            return False
        
        # 분석 실행 버튼
        if st.button("🔍 분석 시작", type="primary", use_container_width=True):
            results = self._run_analysis(data, plan)
            
            if results:
                st.session_state.analysis_results = results
                st.session_state.current_step = 4
                st.success("✅ 분석이 완료되었습니다!")
                st.rerun()
            else:
                st.error("분석 중 오류가 발생했습니다.")
                return False
        
        return False
    
    def step4_display_results(self):
        """4단계: 결과 표시"""
        if not st.session_state.analysis_results:
            st.warning("분석 결과가 없습니다.")
            return
            
        st.header("4️⃣ 분석 결과")
        results = st.session_state.analysis_results
        
        # 결과 개요
        st.write("### 📊 분석 개요")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("분석 완료 시간", results.get("timestamp", "Unknown"))
        with col2:
            st.metric("소요 시간", results.get("elapsed_time", "Unknown"))
        with col3:
            st.metric("분석된 모듈 수", len([k for k, v in results.items() if k.endswith("_analysis")]))
        with col4:
            if "keyword_analysis" in results:
                keyword_count = len(results["keyword_analysis"].get("keywords", []))
                st.metric("추출된 키워드 수", keyword_count)
                            
        # 각 분석 결과 표시
        if "sentiment_analysis" in results:
            self._display_sentiment_results(results["sentiment_analysis"])
        
        if "keyword_analysis" in results:
            self._display_keyword_results(results["keyword_analysis"])
        
        if "summary_analysis" in results:
            self._display_summary_results(results["summary_analysis"])
        
    
    def _run_analysis(self, data: pd.DataFrame, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """🚀 병렬 분석 실행 (전체 데이터)"""
        results = {}
        
        # 분석 대상 데이터 준비
        target_data = self._prepare_analysis_data(data, plan)
        
        if target_data is None or target_data.empty:
            st.error("분석할 데이터가 없습니다.")
            return None
        
        modules = plan["recommended_modules"]
        total_steps = sum(1 for module in modules.values() if module["use"])
        
        if total_steps == 0:
            st.error("실행할 분석 모듈이 없습니다.")
            return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            start_time = time.time()
            
            # 🚀 병렬 처리로 모든 분석 동시 실행
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                # 각 분석을 별도 스레드에서 실행
                if modules["keyword_analysis"]["use"]:
                    futures["keyword"] = executor.submit(
                        self.keyword_analysis,
                        target_data, modules["keyword_analysis"]["target_columns"]
                    )
                
                if modules["sentiment_analysis"]["use"]:
                    futures["sentiment"] = executor.submit(
                        self._run_sentiment_analysis_fast,
                        target_data, modules["sentiment_analysis"]["target_columns"]
                    )
                
                if modules["summary_analysis"]["use"]:
                    futures["summary"] = executor.submit(
                        self._run_summary_analysis_fast,
                        target_data, modules["summary_analysis"]["target_columns"]
                    )
                
                # 결과 수집
                completed_count = 0
                for analysis_type, future in futures.items():
                    status_text.text(f"⚡ {analysis_type} 분석 완료 대기 중...")
                    
                    try:
                        result = future.result(timeout=60)  # 60초 타임아웃
                        results[f"{analysis_type}_analysis"] = result
                        completed_count += 1
                        progress_bar.progress(completed_count / len(futures))
                        st.success(f"✅ {analysis_type} 분석 완료!")
                    except Exception as e:
                        st.error(f"❌ {analysis_type} 분석 실패: {e}")
                        results[f"{analysis_type}_analysis"] = {"error": str(e)}
                        completed_count += 1
                        progress_bar.progress(completed_count / len(futures))
            
            progress_bar.progress(1.0)
            elapsed_time = time.time() - start_time
            status_text.text(f"✅ 모든 분석 완료! (소요 시간: {elapsed_time:.1f}초)")
            
            # 결과에 메타 정보 추가
            results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results["plan"] = plan
            results["elapsed_time"] = f"{elapsed_time:.1f}초"
            
            return results
            
        except Exception as e:
            st.error(f"분석 중 오류 발생: {str(e)}")
            return None
    
    def _prepare_analysis_data(self, data: pd.DataFrame, plan: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """🚀 병렬 분석을 위한 데이터 준비 (전체 데이터)"""
        
        # 분석 대상 컬럼들 수집
        target_columns = set()
        modules = plan["recommended_modules"]
        
        st.write("**분석 모듈별 대상 컬럼:**")
        for module_name, module_info in modules.items():
            if module_info["use"]:
                cols = module_info["target_columns"]
                target_columns.update(cols)
                st.write(f"- {module_name}: {cols}")
        
        if not target_columns:
            st.error("❌ 분석 대상 컬럼이 지정되지 않았습니다.")
            return None
        
        # 대상 컬럼이 데이터에 있는지 확인
        available_columns = [col for col in target_columns if col in data.columns]
        missing_columns = [col for col in target_columns if col not in data.columns]
        
        st.write(f"**사용 가능한 컬럼:** {available_columns}")
        if missing_columns:
            st.warning(f"⚠️ 데이터에 없는 컬럼: {missing_columns}")
        
        if not available_columns:
            st.error("❌ 분석 가능한 컬럼이 없습니다.")
            return None
        
        # 데이터 필터링 (샘플링 제거)
        filtered_data = data[available_columns].copy()
        
        # 모든 컬럼이 빈 값인 행만 제거
        filtered_data = filtered_data.dropna(how='all')
        
        if filtered_data.empty:
            st.error("❌ 필터링 후 분석할 데이터가 없습니다.")
            return None
        
        st.success(f"✅ 전체 데이터 분석 준비 완료: {len(filtered_data)}행, {len(available_columns)}개 컬럼")
        st.info("🚀 병렬 처리로 전체 데이터를 분석합니다.")
        return filtered_data
    
    def _run_keyword_analysis_fast(self, data: pd.DataFrame, target_columns: list) -> Dict[str, Any]:
        """🚀 병렬 키워드 분석 (개선된 오류 처리)"""
        
        # 🔍 상세한 진행 상황 표시
        st.write("🔑 **키워드 분석 시작**")
        progress_container = st.empty()
        
        try:
            # 1단계: LLM 초기화
            progress_container.write("🤖 1/5 단계: LLM 초기화 중...")
            
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=st.secrets["your_section"]["api_key"],
                    temperature=0
                )
                progress_container.write("   ✅ LLM 초기화 완료")
                
            except Exception as e:
                error_msg = f"❌ LLM 초기화 실패:\n{str(e)}\nAPI 키와 인터넷 연결을 확인해주세요."
                st.error(error_msg)
                return {"error": error_msg, "texts_analyzed": 0}
            
            # 2단계: 텍스트 데이터 수집
            progress_container.write("📊 2/5 단계: 텍스트 데이터 수집 중...")
            
            texts = []
            for col in target_columns:
                if col in data.columns:
                    valid_texts = data[col].dropna().astype(str)
                    valid_texts = valid_texts[valid_texts.str.strip() != '']
                    texts.extend(valid_texts.tolist())
                    progress_container.write(f'   - [{col}] 컬럼에서 텍스트 {len(valid_texts):,}개 텍스트 수집')
            
            if not texts:
                error_msg = f"❌ 분석할 텍스트가 없습니다.\n대상 컬럼: {target_columns}\n각 컬럼의 데이터 상태를 확인해주세요."
                st.error(error_msg)
                return {"error": error_msg, "texts_analyzed": 0}
            
            # 3단계: 데이터 크기 및 품질 검증
            progress_container.write(f"📏 3/5 단계: 데이터 품질 검증 ({len(texts):,}개 텍스트)")
            
            # 텍스트 평균 길이 계산
            avg_length = sum(len(str(text)) for text in texts) / len(texts)
            progress_container.write(f"   평균 텍스트 길이: {avg_length:.1f}자")
            
            if avg_length < 10:
                st.warning("⚠️ 텍스트가 너무 짧습니다. 키워드 추출 품질이 낮을 수 있습니다.")
            
            if len(texts) > 500:
                st.warning(f"⚠️ 텍스트가 많습니다 ({len(texts):,}개). 처리 시간이 오래 걸릴 수 있습니다.")
            
            # 4단계: 키워드 분석 모듈 가져오기
            progress_container.write("📦 4/5 단계: 키워드 분석 모듈 로드 중...")
            
            try:
                from modules.categorize import extract_keywords_parallel
                progress_container.write("   ✅ 키워드 분석 모듈 로드 완료")
            except ImportError as e:
                error_msg = f"❌ 키워드 분석 모듈을 찾을 수 없습니다:\n{str(e)}\n'modules/categorize.py' 파일이 존재하는지 확인해주세요."
                st.error(error_msg)
                return {"error": error_msg, "texts_analyzed": len(texts)}
            
            # 5단계: 키워드 추출 실행
            progress_container.write("🔄 5/5 단계: 키워드 추출 실행 중...")
            
            try:
                # 🔧 안전한 키워드 추출 (청크 크기와 워커 수 조정)
                chunk_size = min(20, max(5, len(texts) // 10))  # 동적 청크 크기
                max_workers = min(3, max(1, len(texts) // 100))  # 동적 워커 수
                
                progress_container.write(f"   설정: 청크 크기={chunk_size}, 워커 수={max_workers}")
                
                keywords = extract_keywords_parallel(texts, llm, chunk_size=chunk_size, max_workers=max_workers)
                
                if not keywords:
                    error_msg = "❌ 키워드 추출 결과가 없습니다.\n가능한 원인:\n- API 호출 실패\n- 텍스트 품질 문제\n- 네트워크 연결 문제"
                    st.error(error_msg)
                    return {"error": error_msg, "texts_analyzed": len(texts)}
                
                # 성공 메시지
                progress_container.write(f"✅ 키워드 분석 완료! 추출된 키워드: {len(keywords):,}개")
                
                return {
                    "keywords": keywords,
                    "texts_analyzed": len(texts),
                    "avg_text_length": avg_length,
                    "chunk_size": chunk_size,
                    "max_workers": max_workers,
                    "status": "completed",
                    "method": "parallel_extraction"
                }
                
            except Exception as e:
                error_msg = f"❌ 키워드 추출 중 오류 발생:\n{str(e)}\n\n상세 정보:\n- 텍스트 수: {len(texts):,}\n- 평균 길이: {avg_length:.1f}자\n- 오류 타입: {type(e).__name__}"
                st.error(error_msg)
                return {"error": error_msg, "texts_analyzed": len(texts)}
                
        except Exception as e:
            error_msg = f"❌ 키워드 분석 중 예상치 못한 오류:\n{str(e)}\n\n전체 스택:\n{type(e).__name__}: {str(e)}"
            st.error(error_msg)
            progress_container.empty()
            return {"error": error_msg, "texts_analyzed": 0}

    
    def _run_sentiment_analysis_fast(self, data: pd.DataFrame, target_columns: list) -> Dict[str, Any]:
        """🚀 감정 분석 (요약 기반)"""
        try:
            # LLM 초기화
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=st.secrets["your_section"]["api_key"],
                    temperature=0
                )
            except Exception as e:
                return {"error": f"LLM 초기화 실패: {str(e)}", "texts_analyzed": 0}

            # 텍스트 수집
            texts = []
            for col in target_columns:
                if col in data.columns:
                    valid_texts = data[col].dropna().astype(str)
                    valid_texts = valid_texts[valid_texts.str.strip() != '']
                    texts.extend(valid_texts.tolist())

            if not texts:
                return {"error": "분석할 텍스트가 없습니다.", "texts_analyzed": 0}
            
           

            # 키워드 기반 분류 (필수)
            categorized_df = categorize_keywords_batch(texts, llm)


            # 감정 분석 전체 파이프라인
            sentiment_df, _, categorized_df = analyze_sentiment_with_finbert(texts, llm)  # llm 없이 진행
            refined_df = refine_neutral_keywords_with_gpt(sentiment_df, None)
            updated_df = merge_sentiment_results(sentiment_df, refined_df)
            summary = summarize_sentiment_by_category(categorized_df, updated_df)

            return {
                "summary_df": summary,  # PieChart용 감정 분포
                "updated_sentiment_df": updated_df,
                "texts_analyzed": len(texts),
                "status": "completed"
            }

        except Exception as e:
            return {"error": str(e), "texts_analyzed": 0}

    def _run_keyword_analysis_fast(self, data: pd.DataFrame, target_columns: list) -> Dict[str, Any]:
        """🚀 병렬 키워드 분석 (스레드 안전 버전)"""
        
        try:
            # LLM 초기화
            try:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=st.secrets["your_section"]["api_key"],
                    temperature=0
                )
            except Exception as e:
                return {"error": f"LLM 초기화 실패: {str(e)}", "texts_analyzed": 0}
            
            # 텍스트 데이터 수집
            texts = []
            for col in target_columns:
                if col in data.columns:
                    valid_texts = data[col].dropna().astype(str)
                    valid_texts = valid_texts[valid_texts.str.strip() != '']
                    texts.extend(valid_texts.tolist())
            
            if not texts:
                return {"error": "분석할 텍스트가 없습니다.", "texts_analyzed": 0}
            
            # 텍스트 품질 검증
            avg_length = sum(len(str(text)) for text in texts) / len(texts)
            
            # 키워드 분석 모듈 로드
            try:
                from modules.categorize import extract_keywords_parallel
            except ImportError as e:
                return {"error": f"키워드 분석 모듈 로드 실패: {str(e)}", "texts_analyzed": len(texts)}
            
            # 키워드 추출 실행
            try:
                chunk_size = min(20, max(5, len(texts) // 10))
                max_workers = min(2, max(1, len(texts) // 100))  # 워커 수 줄임
                
                keywords = extract_keywords_parallel(texts, llm, chunk_size=chunk_size, max_workers=max_workers)
                
                if not keywords:
                    return {"error": "키워드 추출 결과가 없습니다.", "texts_analyzed": len(texts)}
                
                return {
                    "keywords": keywords,
                    "texts_analyzed": len(texts),
                    "avg_text_length": avg_length,
                    "chunk_size": chunk_size,
                    "max_workers": max_workers,
                    "status": "completed",
                    "method": "thread_safe_extraction"
                }
                
            except Exception as e:
                return {"error": f"키워드 추출 중 오류: {str(e)}", "texts_analyzed": len(texts)}
            
        except Exception as e:
            return {"error": str(e), "texts_analyzed": 0}

    def _run_summary_analysis(self, data: pd.DataFrame, target_columns: list) -> Dict[str, Any]:
        """📝 요약 분석 (스레드 안전 버전)"""
        
        try:
            # 텍스트 데이터 수집
            texts = []
            for col in target_columns:
                if col in data.columns:
                    valid_texts = data[col].dropna().astype(str)
                    valid_texts = valid_texts[valid_texts.str.strip() != '']
                    texts.extend(valid_texts.tolist())
            
            if not texts:
                return {"error": "요약할 텍스트가 없습니다.", "texts_analyzed": 0}
            
            # 데이터 품질 검증
            total_length = sum(len(str(text)) for text in texts)
            avg_length = total_length / len(texts)
            
            # 요약 모듈 로드
            try:
                from modules.summary_module import generate_summary_with_gpt
            except ImportError as e:
                return {"error": f"요약 모듈 로드 실패: {str(e)}", "texts_analyzed": len(texts)}
            
            # 요약 실행
            try:
                summary = generate_summary_with_gpt(texts)
                
                if not summary or len(str(summary).strip()) < 10:
                    return {"error": "요약 생성에 실패했습니다.", "texts_analyzed": len(texts)}
                
                summary_length = len(str(summary))
                compression_ratio = (total_length / summary_length) if summary_length > 0 else 0
                
                return {
                    "summary": summary,
                    "texts_analyzed": len(texts),
                    "total_length": total_length,
                    "summary_length": summary_length,
                    "compression_ratio": compression_ratio,
                    "status": "completed"
                }
                
            except Exception as e:
                return {"error": f"요약 생성 중 오류: {str(e)}", "texts_analyzed": len(texts)}
            
        except Exception as e:
            return {"error": str(e), "texts_analyzed": 0}

    def step3_execute_analysis(self, data: pd.DataFrame):
        """3단계: 분석 실행 (스레드 안전 UI)"""
        
        plan = st.session_state.get('analysis_plan', {})
        if not plan:
            st.error("❌ 분석 계획이 없습니다. 2단계로 돌아가서 계획을 확인해주세요.")
            return
        
        st.header("3️⃣ AI 분석 실행")
        
        # 분석 준비 정보 표시
        target_columns = []
        modules = plan.get("recommended_modules", {})
        for module_name, module_info in modules.items():
            if module_info.get("use", False):
                target_columns.extend(module_info.get("target_columns", []))
        
        target_columns = list(set(target_columns))
        
        if not target_columns:
            st.error("❌ 분석할 컬럼이 지정되지 않았습니다.")
            return
        
        st.write("### 📊 분석 준비 정보")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 데이터", f"{len(data):,}행")
        with col2:
            st.metric("분석 컬럼", f"{len(target_columns)}개")
        with col3:
            total_texts = sum(len(data[col].dropna()) for col in target_columns if col in data.columns)
            st.metric("분석 텍스트", f"{total_texts:,}개")
        
        # 분석 실행
        if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
            
            st.write("### 🔄 분석 진행 상황")
            
            # 🔧 UI는 메인 스레드에서만 업데이트
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # 병렬 실행
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                # 🔧 스레드 안전한 분석 함수들 실행
                if modules.get("sentiment_analysis", {}).get("use", False):
                    futures["sentiment"] = executor.submit(self._run_sentiment_analysis_fast, data, target_columns)
                    status_placeholder.write("😊 감정 분석 시작됨...")
                
                if modules.get("keyword_analysis", {}).get("use", False):
                    futures["keyword"] = executor.submit(self.keyword_analysis, data, target_columns)
                    status_placeholder.write("🔑 키워드 분석 시작됨...")
                
                if modules.get("summary_analysis", {}).get("use", False):
                    futures["summary"] = executor.submit(self._run_summary_analysis, data, target_columns)
                    status_placeholder.write("📝 요약 분석 시작됨...")
                
                # 🔧 결과 수집 (메인 스레드에서 UI 업데이트)
                results = {}
                completed_count = 0
                total_analyses = len(futures)
                
                for analysis_type, future in futures.items():
                    try:
                        # 진행률 업데이트
                        progress_placeholder.progress(completed_count / total_analyses)
                        status_placeholder.write(f"⚡ {analysis_type} 분석 완료 대기 중...")
                        
                        result = future.result(timeout=300)  # 5분 타임아웃
                        
                        if result and not result.get("error"):
                            results[f"{analysis_type}_analysis"] = result
                            st.success(f"✅ {analysis_type} 분석 완료!")
                        else:
                            st.error(f"❌ {analysis_type} 분석 실패: {result.get('error', '알 수 없는 오류')}")
                        
                        completed_count += 1
                        progress_placeholder.progress(completed_count / total_analyses)
                        
                    except Exception as e:
                        st.error(f"❌ {analysis_type} 분석 중 예외: {str(e)}")
                        completed_count += 1
                        progress_placeholder.progress(completed_count / total_analyses)
        
            # 최종 처리
            progress_placeholder.empty()
            status_placeholder.empty()
            
            if results:
                st.session_state.analysis_results = results
                st.success("🎉 분석이 완료되었습니다!")
                st.session_state.current_step = 4
                st.rerun()
            else:
                st.error("❌ 모든 분석이 실패했습니다. 오류 메시지를 확인해주세요.")
                
                
            # 결과 다운로드 옵션
            st.write("---")
            st.write("### 💾 결과 저장")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 JSON으로 다운로드"):
                    self._download_results_as_json(results)
            
            with col2:
                if st.button("🔄 새 분석 시작"):
                    # 세션 초기화
                    for key in ['analysis_plan', 'plan_approved', 'analysis_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.current_step = 1
                    st.rerun()