from collections.abc import Callable
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

import json
import re
from services.coach_engine import (build_coach_recommendation)

class TrainingAgentState(TypedDict,total=False,):
    activities_df: pd.DataFrame
    wellness_df: pd.DataFrame
    wellness_sport_info_df: pd.DataFrame
    events_df: pd.DataFrame
    
    preferred_sport: str
    available_minutes: int
    coach_config: dict

    user_request: str

    activities_analysis: str
    wellness_analysis: str
    sport_info_analysis: str
    events_analysis: str
    analysis_data: str
    
    coach_recommendation: dict
    coach_data: str
    
    prompt: str
    answer: str

    current_step: str
    progress: int
    error: str


def create_training_graph(
    ollama_client,
    activities_builder,
    wellness_builder,
    sport_info_builder,
    events_builder,
):
    """
    トレーニング分析用LangGraphを構築する。

    Args:
        ollama_client:
            既存のOllamaClientインスタンス。

        analysis_builder:
            app.pyのbuild_analysis_data関数。

    Returns:
        CompiledStateGraph:
            実行可能なLangGraph。
    """

    def validate_data(state: TrainingAgentState) -> dict:
        activities_df = state.get("activities_df")

        wellness_df = state.get("wellness_df",pd.DataFrame())

        events_df = state.get("events_df",pd.DataFrame())

        if activities_df is None:
            raise ValueError(
                "Activities DataFrameが"
                "渡されていません。"
            )

        if activities_df.empty:
            raise ValueError(
                "Activitiesデータがありません。"
            )

        message = (
            f"Activities: {len(activities_df):,}件, "
            f"Wellness: {len(wellness_df):,}件, "
            f"Events: {len(events_df):,}件"
        )

        return {
            "validation_message": message,
            "current_step": "3種類のデータ検証完了",
            "progress": 20,
        }

    def prepare_analysis_data(state: TrainingAgentState) -> dict:
        """
        Activities、Wellness、Eventsを
        それぞれ分析用テキストへ変換する。
        """

        activities_df = state.get("activities_df",pd.DataFrame())
        wellness_df = state.get("wellness_df",pd.DataFrame())
        sport_info_df = state.get("wellness_sport_info_df",pd.DataFrame())
        events_df = state.get("events_df",pd.DataFrame())
        activities_analysis = (activities_builder(activities_df))
        wellness_analysis = (wellness_builder(wellness_df))
        events_analysis = (events_builder(events_df))
        sport_info_analysis = (sport_info_builder(sport_info_df))
        analysis_data = "\n\n".join([activities_analysis,wellness_analysis,sport_info_analysis,events_analysis])

        return {
            "activities_analysis": activities_analysis,
            "wellness_analysis": wellness_analysis,
            "events_analysis": events_analysis,
            "analysis_data": analysis_data,
            "current_step": (
                "Activities・Wellness・Events"
                "分析データ作成完了"
            ),
            "progress": 45,
        }

    def prepare_coach_data(state: TrainingAgentState) -> dict:
        """
        Pythonの決定的ルールで、
        次回メニュー候補を作成する。
        """

        recommendation = (
            build_coach_recommendation(
                wellness_df=state.get(
                    "wellness_df",
                    pd.DataFrame(),
                ),
                sport_info_df=state.get(
                    "wellness_sport_info_df",
                    pd.DataFrame(),
                ),
                events_df=state.get(
                    "events_df",
                    pd.DataFrame(),
                ),
                available_minutes=state.get(
                    "available_minutes",
                    60,
                ),
                preferred_sport=state.get(
                    "preferred_sport",
                    "Ride",
                ),
                config=state.get(
                    "coach_config",
                    {},
                ),
            )
        )

        coach_data = json.dumps(
            recommendation,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        return {
            "coach_recommendation": (
                recommendation
            ),
            "coach_data": coach_data,
            "current_step": (
                "次回メニュー候補作成完了"
            ),
            "progress": 58,
        }

    def build_prompt(state: TrainingAgentState) -> dict:
        user_request = state.get("user_request","").strip()
        analysis_data = state["analysis_data"]
        coach_data = state["coach_data"]

        prompt = f"""
    あなたは、日本語のみを使用する
    持久系スポーツのトレーニング支援アシスタントです。

    以下には、トレーニング実績、Wellness、
    種目別eFTP、今後の予定があります。

    【分析データ】
    {analysis_data}

    【Python側で作成した次回メニュー候補】
    {coach_data}

    【利用者の依頼】
    {user_request}

    重要なルール:

    - Python側のメニュー候補を基本案として使用してください。
    - 数値、時間、強度、TSSを勝手に変更しないでください。
    - データにない値を作らないでください。
    - HRV、睡眠、安静時心拍から健康状態を断定しないでください。
    - 医学的診断を行わないでください。
    - 痛みや強い不調がある場合は実施を控える旨を記載してください。
    - 回答はすべて自然な日本語にしてください。
    - 韓国語、中国語などを混在させないでください。
    - 日付は「YYYY-MM-DD時点」と記述してください。

    次の形式で回答してください。

    ## 1. 現在の状態

    - 分析基準日
    - CTL
    - ATL
    - TSB
    - Wellnessの個人内基準との比較
    - 判断に使えなかったデータ

    ## 2. 今後の予定

    - 最も近いイベント
    - イベントまでの日数
    - 次回メニューへの影響

    ## 3. 次回トレーニングメニュー

    - 目的
    - 種目
    - 合計時間
    - 目標強度
    - 想定TSS
    - ウォームアップ
    - メインセット
    - クールダウン

    ## 4. このメニューを選んだ理由

    Python側のreason_codesを、
    利用者に分かる日本語で説明してください。

    ## 5. 実施時の調整条件

    - 短縮する条件
    - 中止する条件
    - データ不足による注意点

    ## 6. 次回確認するデータ

    トレーニング後に確認する項目を示してください。
    """.strip()

        return {
            "prompt": prompt,
            "current_step": (
                "コーチ用プロンプト作成完了"
            ),
            "progress": 70,
        }

    def call_ollama(state: TrainingAgentState) -> dict:
        """
        Ollamaを呼び出して分析結果を取得する。
        """

        if not ollama_client.check_connection():
            raise ConnectionError(
                "Ollamaに接続できません。"
            )

        if not ollama_client.is_model_installed():
            raise RuntimeError(
                "config.iniで指定されたOllamaモデルが"
                "インストールされていません。"
            )

        answer = ollama_client.analyze(
            state["prompt"]
        )

        return {
            "answer": answer,
            "current_step": "Ollama分析完了",
            "progress": 100,
        }

    def validate_answer(state: TrainingAgentState) -> dict:
        answer = state.get("answer","")

        if not answer.strip():
            raise ValueError(
                "Ollamaの回答が空です。"
            )

        # ハングル範囲
        hangul_pattern = re.compile(
            r"[\u1100-\u11ff"
            r"\u3130-\u318f"
            r"\uac00-\ud7af]"
        )

        if hangul_pattern.search(answer):
            raise ValueError(
                "回答にハングル文字が"
                "含まれています。"
            )

        required_sections = [
            "現在の状態",
            "今後の予定",
            "次回トレーニングメニュー",
            "このメニューを選んだ理由",
            "実施時の調整条件",
        ]

        missing_sections = [
            section
            for section in required_sections
            if section not in answer
        ]

        if missing_sections:
            raise ValueError(
                "回答に必要な章がありません: "
                + ", ".join(missing_sections)
            )

        return {
            "answer": answer,
            "current_step": "回答検証完了",
            "progress": 100,
        }

    graph_builder = StateGraph(TrainingAgentState)

    graph_builder.add_node("validate_data",validate_data)
    graph_builder.add_node("prepare_analysis_data",prepare_analysis_data)
    graph_builder.add_node("prepare_coach_data",prepare_coach_data)
    graph_builder.add_node("build_prompt",build_prompt)
    graph_builder.add_node("call_ollama",call_ollama)
    graph_builder.add_node("validate_answer",validate_answer)

    graph_builder.add_edge(START,"validate_data")
    graph_builder.add_edge("validate_data","prepare_analysis_data")
    graph_builder.add_edge("prepare_analysis_data","prepare_coach_data")
    graph_builder.add_edge("prepare_coach_data","build_prompt")
    graph_builder.add_edge("build_prompt","call_ollama")
    graph_builder.add_edge("call_ollama","validate_answer")
    graph_builder.add_edge("validate_answer",END)

    return graph_builder.compile()