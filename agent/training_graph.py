from collections.abc import Callable
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph


class TrainingAgentState(TypedDict,total=False,):
    activities_df: pd.DataFrame
    wellness_df: pd.DataFrame
    events_df: pd.DataFrame

    user_request: str

    activities_analysis: str
    wellness_analysis: str
    events_analysis: str

    analysis_data: str
    prompt: str
    answer: str

    current_step: str
    progress: int
    error: str


def create_training_graph(
    ollama_client,
    activities_builder,
    wellness_builder,
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

    def validate_data(state: TrainingAgentState,) -> dict:
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

    def prepare_analysis_data(state: TrainingAgentState,) -> dict:
        """
        Activities、Wellness、Eventsを
        それぞれ分析用テキストへ変換する。
        """

        activities_df = state.get("activities_df",pd.DataFrame())
        wellness_df = state.get("wellness_df",pd.DataFrame())
        events_df = state.get("events_df",pd.DataFrame())
        activities_analysis = (activities_builder(activities_df))
        wellness_analysis = (wellness_builder(wellness_df))
        events_analysis = (events_builder(events_df))

        analysis_data = "\n\n".join([activities_analysis,wellness_analysis,events_analysis])

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

    def build_prompt(state: TrainingAgentState,) -> dict:
        """
        Ollamaへ送信するプロンプトを作成する。
        """

        user_request = state.get(
            "user_request",
            "",
        ).strip()

        if not user_request:
            user_request = (
                "トレーニング傾向を分析し、"
                "次回に向けた一般的な提案をしてください。"
            )

        analysis_data = state["analysis_data"]

        prompt = f"""
あなたは、持久系スポーツのトレーニングデータを
客観的に整理する日本語アシスタントです。

以下のデータを分析してください。

{analysis_data}

利用者の追加依頼:

{user_request}

データの意味:

- Activitiesは、完了済みトレーニングです。
- Wellnessは、日別の身体状態と
  フィットネス指標です。
- Eventsは、カレンダー上の予定です。
- 負荷はTSS相当です。
- フィットネスはCTLです。
- ファティーグはATLです。
- フォームはTSBです。
- TSBはCTLからATLを引いた値です。
- 強度はIFです。
- 正規化パワーはNPです。
- eFTPは推定FTPです。

次の構成で回答してください。

## 1. 結論

重要な特徴を3点以内で示してください。

## 2. 完了済みトレーニング

直近7日、前の7日、直近28日の
距離、時間、負荷、IF、NP、eFTPを説明してください。

## 3. 現在の状態

CTL、ATL、TSBと、
取得できているWellness指標を説明してください。

## 4. 今後の予定

Eventsに予定がある場合、
今後のトレーニングやイベントを整理してください。

## 5. データ上の確認点

欠損値、取得できない指標、
判断できない項目を明記してください。

## 6. 次回への一般的な提案

完了済みトレーニング、Wellness、予定を
総合して、選択肢を最大3つ示してください。

回答ルール:

- 提供されていない数値を作らないでください。
- 医学的な診断は行わないでください。
- HRVや安静時心拍だけから体調を断定しないでください。
- Wellnessがない場合は、その旨を明記してください。
- Eventsがない場合は、予定なしと明記してください。
- 数値には基準日または対象期間を付けてください。
- 回答は日本語のMarkdown形式にしてください。
""".strip()

        return {
            "prompt": prompt,
            "current_step": "プロンプト作成完了",
            "progress": 60,
        }

    def call_ollama(state: TrainingAgentState,) -> dict:
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

    graph_builder = StateGraph(
        TrainingAgentState
    )

    graph_builder.add_node(
        "validate_data",
        validate_data,
    )

    graph_builder.add_node(
        "prepare_analysis_data",
        prepare_analysis_data,
    )

    graph_builder.add_node(
        "build_prompt",
        build_prompt,
    )

    graph_builder.add_node(
        "call_ollama",
        call_ollama,
    )

    graph_builder.add_edge(
        START,
        "validate_data",
    )

    graph_builder.add_edge(
        "validate_data",
        "prepare_analysis_data",
    )

    graph_builder.add_edge(
        "prepare_analysis_data",
        "build_prompt",
    )

    graph_builder.add_edge(
        "build_prompt",
        "call_ollama",
    )

    graph_builder.add_edge(
        "call_ollama",
        END,
    )

    return graph_builder.compile()