import openai
import streamlit as st
import json
from typing import Dict, List, Any
import pandas as pd

class AnalysisPlan:
    '''분석 계획을 정의하는 클래스'''
    def __init__(self):
        self.client = openai.OpenAI(api_key = st.secrets["your_section"]["api_key"])
    
    def generate_initial_plan(self, data: pd.DataFrame, description: str):
        """분석 계획 생성 (실제 플로우 반영)"""
        
        data_info = self._extract_data_info(data)
        
        # 🔧 실제 시스템 플로우에 맞춘 프롬프트
        enhanced_prompt = f"""
        당신은 데이터 분석 전문가입니다. 
        주어진 데이터를 분석하여 최적의 텍스트 분석 계획을 수립해주세요.

        ## 📊 데이터 정보
        **사용자 설명**: {description}
        **컬럼 목록**: {list(data.columns)}
        **데이터 구조**: {data_info}
        **실제 데이터 샘플**:
        {data.head(3).to_string()}

        ## 🎯 분석 계획 요청

        위 데이터를 종합적으로 분석하여 다음을 포함한 분석 계획을 JSON으로 생성해주세요:

        1. **데이터 특성 파악**: 이 데이터의 목적과 성격 분석
        2. **컬럼별 역할 분류**: 각 컬럼이 분석에서 담당할 역할
        3. **최적 분석 방법**: 감정분석, 키워드분석, 요약분석 적용 여부
        4. **예상 인사이트**: 분석을 통해 얻을 수 있는 구체적 인사이트

        ## 📋 JSON 응답 형식 예시

        {{
            "data_subject": "데이터 주제 (구체적으로 출력해주세요.)",
            "column_analysis": {{
                "컬럼명": {{
                    "include": true/false,
                    "reason": "포함/제외 이유 (구체적으로 출력해주세요.)",
                    "expected_insight": "이 컬럼에서 얻을 수 있는 인사이트"
                }}
            }},
            "target_insights": [
                "구체적인 분석 목표 1 (예: 직원 만족도의 주요 불만 요소 파악)",
                "구체적인 분석 목표 2 (예: 부서별 만족도 차이 분석)",
                "구체적인 분석 목표 3 (예: 개선 우선순위 도출)"
            ],
            "recommended_modules": {{
                "sentiment_analysis": {{
                    "use": true/false,
                    "reason": "감정분석이 필요한 구체적 이유",
                    "target_columns": ["실제존재하는컬럼명"]
                }},
                "keyword_analysis": {{
                    "use": true/false, 
                    "reason": "키워드분석의 기대 효과",
                    "target_columns": ["실제존재하는컬럼명"]
                }},
                "summary_analysis": {{
                    "use": true/false,
                    "reason": "요약분석의 비즈니스 가치",
                    "target_columns": ["실제존재하는컬럼명"]
                }}
            }},
            "analysis_workflow": [
                "1단계: AI 분석 기획서 생성 및 검토",
                "2단계: 기획서 승인 및 분석 준비",
                "3단계: 감정분석, 키워드분석, 요약분석 병렬 실행",
                "4단계: 분석 결과 시각화 및 인사이트 도출"
            ]
        }}

        ## 🧠 분석 가이드라인

        - 컬럼명과 실제 데이터 내용을 모두 고려하세요
        - 텍스트 길이와 내용의 질을 평가하세요
        - 비즈니스 실무에 도움되는 인사이트에 집중하세요
        - target_columns에는 반드시 실제 존재하는 컬럼명만 입력하세요
        - 현실적으로 의미있는 분석이 가능한지 판단하세요

        **중요**: 
        - target_columns 필드를 반드시 포함하세요
        - 가능한 컬럼: {list(data.columns)}
        - 텍스트 분석이 가능한 컬럼만 target_columns에 포함하세요

        실제 시스템 플로우에 맞는 분석 계획을 수립해주세요.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 텍스트 데이터 분석 전문가입니다. target_columns를 반드시 포함한 정확한 JSON을 생성하고, 실제 시스템 동작에 맞는 워크플로우를 제시하세요."
                    },
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,
                max_tokens=2500  # 토큰 수 증가
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # JSON 추출 및 파싱
            if "```json" in plan_text:
                json_start = plan_text.find("```json") + 7
                json_end = plan_text.find("```", json_start)
                plan_text = plan_text[json_start:json_end]
            elif "{" in plan_text:
                json_start = plan_text.find("{")
                json_end = plan_text.rfind("}") + 1
                plan_text = plan_text[json_start:json_end]
            
            plan = json.loads(plan_text)
            
            # 🔧 target_columns 검증 및 자동 수정
            plan = self._fix_target_columns(plan, data)
            
            # 🔧 워크플로우도 실제 시스템에 맞게 수정
            plan = self._fix_workflow(plan)
            
            if validate_plan(plan):
                return plan
            else:
                return self._create_fallback_plan(data, description)
                
        except Exception as e:
            st.error(f"분석 계획 생성 중 오류: {e}")
            return self._create_fallback_plan(data, description)
        
    def revise_plan_with_feedback(self, original_plan: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        '''피드백을 반영하여 기획서 수정'''

        prompt = f"""
        다음은 기존 분석 기획서입니다:
        {json.dumps(original_plan, ensure_ascii=False, indent=2)}

        **검토자 피드백**: {feedback}

        피드백을 반영하여 수정된 분석 기획서를 동일한 JSON 형태로 제공해주세요.
        변경된 부분에 대해서만 수정하고, 나머지는 그대로 유지해주세요.
        
        중요: 반드시 다음 필드들을 모두 포함해야 합니다:
        - data_subject
        - column_analysis
        - target_insights
        - recommended_modules
        - analysis_workflow
        
        유효한 JSON 형식으로만 응답하세요.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 분석 계획 수정 전문가입니다. 피드백을 반영하되 반드시 완전한 JSON을 반환해야 합니다."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1  # 더 일관된 결과를 위해 낮춤
            )

            plan_text = response.choices[0].message.content.strip()
            if plan_text.startswith("```json"):
                plan_text = plan_text[7:-3]
            elif plan_text.startswith("```"):
                plan_text = plan_text[3:-3]

            revised_plan = json.loads(plan_text)
            
            # 필수 필드 검증
            required_fields = ["data_subject", "column_analysis", "target_insights", "recommended_modules"]
            for field in required_fields:
                if field not in revised_plan:
                    st.warning(f"⚠️ AI가 '{field}' 필드를 누락했습니다. 원본 계획을 유지합니다.")
                    return original_plan
            
            st.success("✅ 피드백이 반영된 기획서가 생성되었습니다!")
            return revised_plan
        
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON 파싱 오류: {e}")
            st.info("원본 기획서를 유지합니다.")
            return original_plan
        except Exception as e:
            st.error(f"❌ 기획서 수정 중 오류 발생: {e}")
            st.info("원본 기획서를 유지합니다.")
            return original_plan
        
    def _extract_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        '''데이터의 기본 정보를 추출'''
        
        # 샘플 데이터 안전하게 추출
        sample_data = data.head(3).copy()
        
        # Arrow 호환성을 위해 데이터 정리
        for col in sample_data.columns:
            if sample_data[col].dtype == 'object':
                # 모든 값을 문자열로 변환하고 NaN 처리
                sample_data[col] = sample_data[col].astype(str).replace('nan', '빈 값')
                # 너무 긴 텍스트는 잘라내기
                sample_data[col] = sample_data[col].str[:50] + '...'
        
        return {
            "rows": len(data),
            "columns": len(data.columns),
            "column_list": list(data.columns),
            "sample_data": sample_data.to_dict('records')
        }
    
    def _get_default_plan(self) -> Dict[str, Any]:
        '''기본 분석 기획서'''
        return {
            "data_subject": "데이터 분석이 필요합니다.",
            "column_analysis": {},
            "target_insights": ["기본 데이터 분석"],
            "recommended_modules": {
                "sentiment_analysis": {"use": False, "target_columns": [], "reason": ""},
                "keyword_analysis": {"use": False, "target_columns": [], "reason": ""},
                "summary_analysis": {"use": False, "target_columns": [], "reason": ""}
            },
            "analysis_workflow": ["데이터 확인", "분석 진행"]
        }

    def _fix_target_columns(self, plan: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """target_columns 자동 수정"""
        
        # 텍스트 컬럼 자동 감지
        text_columns = []
        for col in data.columns:
            if data[col].dtype == 'object':
                sample_text = str(data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else "")
                if len(sample_text) > 10:
                    text_columns.append(col)
        
        # recommended_modules의 target_columns 수정
        modules = plan.get("recommended_modules", {})
        
        for module_name, module_info in modules.items():
            if isinstance(module_info, dict):
                if module_info.get("use", False):
                    if "target_columns" not in module_info or not module_info["target_columns"]:
                        module_info["target_columns"] = text_columns
                    else:
                        valid_columns = [col for col in module_info["target_columns"] 
                                       if col in data.columns]
                        module_info["target_columns"] = valid_columns if valid_columns else text_columns
                else:
                    module_info["target_columns"] = []
        
        return plan

    def _fix_workflow(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """실제 시스템 플로우에 맞게 워크플로우 수정"""
        
        # 실제 시스템의 정확한 플로우로 고정
        plan["analysis_workflow"] = [
            "1단계: AI 분석 기획서 생성 및 검토",
            "2단계: 기획서 승인 및 분석 대상 확정", 
            "3단계: 감정분석, 키워드분석, 요약분석 병렬 실행",
            "4단계: 분석 결과 시각화 및 종합 인사이트 도출"
        ]
        
        return plan

    def _create_fallback_plan(self, data: pd.DataFrame, description: str) -> Dict[str, Any]:
        """안전한 기본 계획 생성"""
        
        text_columns = []
        for col in data.columns:
            if data[col].dtype == 'object':
                sample_text = str(data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else "")
                if len(sample_text) > 5:
                    text_columns.append(col)
        
        return {
            "data_subject": f"텍스트 데이터 분석 - {description[:50]}",
            "column_analysis": {
                col: {
                    "include": col in text_columns,
                    "reason": "텍스트 분석 가능" if col in text_columns else "텍스트 분석 불가",
                    "expected_insight": "텍스트 패턴 및 주요 키워드 파악" if col in text_columns else "분석 제외"
                } for col in data.columns
            },
            "target_insights": [
                "텍스트 데이터의 주요 패턴 및 트렌드 분석",
                "핵심 키워드 및 감정 분포 파악",
                "비즈니스 인사이트 및 개선점 도출"
            ],
            "recommended_modules": {
                "sentiment_analysis": {
                    "use": len(text_columns) > 0,
                    "reason": "텍스트 컬럼이 있어 감정 분석 가능",
                    "target_columns": text_columns
                },
                "keyword_analysis": {
                    "use": len(text_columns) > 0,
                    "reason": "텍스트 컬럼이 있어 키워드 분석 가능", 
                    "target_columns": text_columns
                },
                "summary_analysis": {
                    "use": len(text_columns) > 0,
                    "reason": "텍스트 컬럼이 있어 요약 분석 가능",
                    "target_columns": text_columns
                }
            },
            "analysis_workflow": [
                "1단계: AI 분석 기획서 생성 및 검토",
                "2단계: 기획서 승인 및 분석 대상 확정",
                "3단계: 감정분석, 키워드분석, 요약분석 병렬 실행", 
                "4단계: 분석 결과 시각화 및 종합 인사이트 도출"
            ]
        }
def validate_plan(plan: Dict[str, Any]) -> List[str]:
    '''분석 기획서 검증'''
    issues = []

    #필수 필드 확인
    required_fields = ["data_subject", "column_analysis", "target_insights", "recommended_modules"]
    for field in required_fields:
        if field not in plan or not plan[field]:
            issues.append(f"필수 필드 '{field}'이 누락되었습니다.")
    #분석 대상 컬럼이 있는지 확인
    has_analysis_target = False
    modules = plan.get("recommended_modules", {})
    for module_name, module_info in modules.items():
        if module_info.get("use", False) and module_info.get("target_columns"):
            has_analysis_target = True
            break
    
    if not has_analysis_target:
        issues.append("분석 대상 컬럼이 지정되지 않았습니다.")

    return issues