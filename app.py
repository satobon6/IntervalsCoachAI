from pathlib import Path
import configparser
import time
import traceback

import pandas as pd
import streamlit as st

from services.database import (
    initialize_database,
    load_activities,
    load_events,
    load_wellness,
    load_wellness_sport_info,
    save_activities,
    save_events,
    save_wellness,
)
from services.intervals_client import IntervalsClient
from services.ollama_client import OllamaClient
from agent.training_graph import create_training_graph
from datetime import date, timedelta

# Streamlit画面設定。最初のStreamlit命令にする。
st.set_page_config(
    page_title="Intervals.icu AI分析",
    page_icon="🚴",
    layout="wide",
)

st.title("Intervals.icu トレーニング分析")
st.caption("Intervals.icu取得 → SQLite保存 → Streamlit表示 → Ollama分析")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.ini"

def create_sync_date_range(sync_days: int,) -> dict:
    """
    activities、wellness、eventsの取得期間を作成する。
    """
    today = date.today()
    oldest = today - timedelta(days=sync_days - 1)
    event_newest = today + timedelta(days=30)

    return {
        "oldest": oldest.isoformat(),
        "newest": today.isoformat(),
        "event_newest": event_newest.isoformat(),
    }

def set_status(container, message: str, state: str = "running") -> None:
    """処理中の状態を画面上の同じ場所へ表示する。"""
    icons = {"running": "⏳", "success": "✅", "error": "❌", "info": "ℹ️"}
    container.info(f"{icons.get(state, 'ℹ️')} {message}")

def load_config() -> configparser.ConfigParser:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {CONFIG_FILE}")

    config = configparser.ConfigParser()
    loaded = config.read(CONFIG_FILE, encoding="utf-8")
    if not loaded:
        raise RuntimeError(f"設定ファイルを読み込めません: {CONFIG_FILE}")

    for section in ("INTERVALS", "OLLAMA", "DATABASE", "SYNC"):
        if section not in config:
            raise KeyError(f"config.iniに[{section}]セクションがありません。")
    return config

def build_wellness_analysis_data(wellness_df: pd.DataFrame,) -> str:
    """
    Wellnessから最新のコンディション情報を作成する。
    """

    if wellness_df.empty:
        return "Wellnessデータはありません。"
    working_df = wellness_df.copy()
    if "id" not in working_df.columns:
        return ("Wellnessに日付フィールドidがありません。")
    working_df["_wellness_date"] = pd.to_datetime(working_df["id"],errors="coerce")
    working_df = working_df.dropna(subset=["_wellness_date"])
    if working_df.empty:
        return ("Wellnessに有効な日付データがありません。")

    working_df = working_df.sort_values("_wellness_date",ascending=True)

    latest_row = working_df.iloc[-1]
    latest_date = latest_row["_wellness_date"]

    lines = ["【最新Wellness】",f"日付: {latest_date:%Y-%m-%d}"]

    ctl = pd.to_numeric(
        pd.Series([latest_row.get("ctl")]),
        errors="coerce",
    ).iloc[0]

    atl = pd.to_numeric(pd.Series([latest_row.get("atl")]),errors="coerce").iloc[0]

    if pd.notna(ctl):
        lines.append(f"CTL（フィットネス）: {float(ctl):.1f}")

    if pd.notna(atl):
        lines.append(f"ATL（ファティーグ）: {float(atl):.1f}")

    if pd.notna(ctl) and pd.notna(atl):
        tsb = float(ctl) - float(atl)

        lines.append(
            f"TSB（フォーム = CTL - ATL）: "
            f"{tsb:+.1f}"
        )

    field_mapping = {
        "weight": ("体重", "kg", 1),
        "restingHR": ("安静時心拍", "bpm", 0),
        "hrv": ("HRV", "ms", 1),
        "sleepScore": ("睡眠スコア", "", 0),
        "readiness": ("レディネス", "", 0),
        "steps": ("歩数", "歩", 0),
    }

    for field_name, field_info in field_mapping.items():
        if field_name not in latest_row.index:
            continue

        label, unit, decimals = field_info

        value = pd.to_numeric(pd.Series([latest_row[field_name]]),errors="coerce",).iloc[0]

        if pd.isna(value):
            continue

        value_text = f"{float(value):.{decimals}f}"

        lines.append(f"{label}: {value_text} {unit}".rstrip())

    if "sleepSecs" in latest_row.index:
        sleep_seconds = pd.to_numeric(pd.Series([latest_row["sleepSecs"]]),errors="coerce").iloc[0]

        if pd.notna(sleep_seconds):
            sleep_hours = float(sleep_seconds) / 3600
            lines.append(f"睡眠時間: {sleep_hours:.1f} 時間")

    return "\n".join(lines)

def build_sport_info_analysis_data(sport_info_df: pd.DataFrame,) -> str:
    """
    種目別eFTPを分析用テキストへ変換する。
    """

    if sport_info_df.empty:
        return "種目別eFTPデータはありません。"

    required_columns = {"wellness_id","sport_type","eftp"}

    if not required_columns.issubset(
        sport_info_df.columns
    ):
        return (
            "種目別eFTPに必要な列が"
            "不足しています。"
        )

    working_df = sport_info_df.copy()

    working_df["_date"] = pd.to_datetime(
        working_df["wellness_id"],
        errors="coerce",
    )

    working_df["eftp"] = pd.to_numeric(
        working_df["eftp"],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[
            "_date",
            "sport_type",
            "eftp",
        ]
    )

    if working_df.empty:
        return (
            "有効な種目別eFTPデータはありません。"
        )

    working_df = working_df.sort_values(
        "_date",
        ascending=True,
    )

    latest_by_sport = (
        working_df
        .groupby(
            "sport_type",
            as_index=False,
        )
        .tail(1)
        .sort_values(
            "sport_type"
        )
    )

    lines = [
        "【最新の種目別eFTP】",
    ]

    for _, row in latest_by_sport.iterrows():
        date_text = row["_date"].strftime(
            "%Y-%m-%d"
        )

        lines.append(
            f"{row['sport_type']}: "
            f"{row['eftp']:.0f} W "
            f"（{date_text}）"
        )

    return "\n".join(lines)

def build_events_analysis_data(events_df: pd.DataFrame,) -> str:
    """
    Eventsから今後の予定を分析用テキストへ変換する。
    """

    if events_df.empty:
        return "イベントデータはありません。"

    working_df = events_df.copy()

    date_column = find_column(working_df,["start_date_local","start_date","date"])

    if date_column is None:
        return ("イベントの日付列が見つかりません。")

    working_df["_event_date"] = pd.to_datetime(working_df[date_column],errors="coerce")

    working_df = working_df.dropna(subset=["_event_date"])

    if working_df.empty:
        return ("有効なイベント日付がありません。")

    today_timestamp = pd.Timestamp.today().normalize()

    future_df = working_df[
        working_df["_event_date"].dt.normalize()
        >= today_timestamp
    ].sort_values(
        "_event_date",
        ascending=True,
    ).head(10)

    if future_df.empty:
        return "今後のイベントはありません。"

    lines = ["【今後の予定】"]

    for _, row in future_df.iterrows():
        event_date = row["_event_date"]
        event_name = str(row.get("name","名称なし"))

        category = str(row.get("category","分類なし"))

        event_type = str(row.get("type",""))

        details = [event_date.strftime("%Y-%m-%d"),category,event_type,event_name]

        lines.append(
            " / ".join(
                detail
                for detail in details
                if detail
                and detail != "nan"
            )
        )

    return "\n".join(lines)

# 起動状況を常に表示
startup_status = st.status("アプリを初期化しています", expanded=True)
try:
    startup_status.write("1/4 config.iniを確認しています")
    config = load_config()
    startup_status.write("✅ config.iniの読込み完了")

    ATHLETE_ID = config["INTERVALS"].get("ATHLETE_ID", "0").strip()
    INTERVALS_API_KEY = config["INTERVALS"].get("API_KEY", "").strip()
    OLLAMA_BASE_URL = config["OLLAMA"].get("BASE_URL", "http://localhost:11434").strip()
    OLLAMA_MODEL = config["OLLAMA"].get("MODEL", "llama3.1:latest").strip()
    DATABASE_RELATIVE_PATH = config["DATABASE"].get("FILE", "data/intervals.db").strip()
    SYNC_DAYS = config["SYNC"].getint("DAYS",90)
    DB_PATH = BASE_DIR / DATABASE_RELATIVE_PATH
    
    sync_dates = create_sync_date_range(SYNC_DAYS)
    OLDEST_DATE = sync_dates["oldest"]
    NEWEST_DATE = sync_dates["newest"]
    EVENT_NEWEST_DATE = sync_dates["event_newest"]

    startup_status.write("2/4 SQLiteを初期化しています")
    initialize_database(DB_PATH)
    startup_status.write(f"✅ SQLite初期化完了: {DB_PATH}")

    startup_status.write("3/4 APIクライアントを初期化しています")
    intervals_client = IntervalsClient(ATHLETE_ID, INTERVALS_API_KEY)
    ollama_client = OllamaClient(OLLAMA_BASE_URL,OLLAMA_MODEL)
    startup_status.write("✅ クライアント初期化完了")

    startup_status.write("4/4 SQLiteから保存済みデータを読み込んでいます")
    activities_df = load_activities(DB_PATH)
    wellness_df = load_wellness(DB_PATH)
    wellness_sport_info_df = (load_wellness_sport_info(DB_PATH))
    events_df = load_events(DB_PATH)
    startup_status.write(
    "✅ SQLite読込み完了: "
    f"activities={len(activities_df):,}件, "
    f"wellness={len(wellness_df):,}件, "
    f"events={len(events_df):,}件"
    )
    startup_status.update(label="アプリの初期化が完了しました", state="complete", expanded=False)
except Exception as error:
    startup_status.write(f"❌ {type(error).__name__}: {error}")
    startup_status.code(traceback.format_exc())
    startup_status.update(label="初期化に失敗しました", state="error", expanded=True)
    st.stop()


def find_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    columns = {str(c).strip().lower(): str(c) for c in dataframe.columns}
    for candidate in candidates:
        found = columns.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def create_summary(dataframe: pd.DataFrame) -> dict:
    summary = {
        "activity_count": len(dataframe),
        "start_date": None,
        "end_date": None,
        "distance_km": None,
        "duration_hours": None,
        "training_load": None,
    }
    if dataframe.empty:
        return summary

    date_col = find_column(dataframe, ["start_date_local", "start_date", "date"])
    distance_col = find_column(dataframe, ["distance", "distance_km"])
    duration_col = find_column(dataframe, ["moving_time", "elapsed_time", "duration", "icu_recording_time"])
    load_col = find_column(dataframe, ["icu_training_load", "training_load", "training load"])

    if date_col:
        dates = pd.to_datetime(dataframe[date_col], errors="coerce").dropna()
        if not dates.empty:
            summary["start_date"] = dates.min()
            summary["end_date"] = dates.max()

    if distance_col:
        value = float(pd.to_numeric(dataframe[distance_col], errors="coerce").fillna(0).sum())
        summary["distance_km"] = value / 1000 if value > 10000 else value

    if duration_col:
        value = float(pd.to_numeric(dataframe[duration_col], errors="coerce").fillna(0).sum())
        summary["duration_hours"] = value / 3600 if value > 1000 else value

    if load_col:
        summary["training_load"] = float(pd.to_numeric(dataframe[load_col], errors="coerce").fillna(0).sum())

    return summary


def summary_to_text(summary: dict) -> str:
    lines = [f"アクティビティ件数: {summary['activity_count']}件"]
    if summary["start_date"] is not None:
        lines.append(f"対象期間: {summary['start_date']:%Y-%m-%d} ～ {summary['end_date']:%Y-%m-%d}")
    if summary["distance_km"] is not None:
        lines.append(f"合計距離: {summary['distance_km']:,.1f} km")
    if summary["duration_hours"] is not None:
        lines.append(f"合計運動時間: {summary['duration_hours']:,.1f} 時間")
    if summary["training_load"] is not None:
        lines.append(f"合計トレーニング負荷: {summary['training_load']:,.1f}")
    return "\n".join(lines)

def build_analysis_data(
    dataframe: pd.DataFrame,
) -> str:
    """
    Ollamaへ渡す詳細な分析データを作成する。
    直近7日、前7日、直近28日を集計する。
    """

    if dataframe.empty:
        return "分析対象データはありません。"

    date_column = find_column(
        dataframe,
        [
            "start_date_local",
            "start_date",
            "date",
        ],
    )

    if date_column is None:
        return (
            "日付列が見つからないため、"
            "期間別分析を実行できません。"
        )

    analysis_df = dataframe.copy()

    analysis_df["_analysis_date"] = pd.to_datetime(
        analysis_df[date_column],
        errors="coerce",
    )

    analysis_df = analysis_df.dropna(
        subset=["_analysis_date"]
    )

    if analysis_df.empty:
        return "有効な日付データがありません。"

    # 将来日付の影響を避けるため、
    # データ内の最終日を分析基準日にする
    reference_date = (
        analysis_df["_analysis_date"]
        .max()
        .normalize()
    )

    distance_column = find_column(
        analysis_df,
        [
            "distance",
            "distance_km",
        ],
    )

    duration_column = find_column(
        analysis_df,
        [
            "moving_time",
            "elapsed_time",
            "duration",
            "icu_recording_time",
        ],
    )

    load_column = find_column(
        analysis_df,
        [
            "icu_training_load",
            "training_load",
            "training load",
        ],
    )

    fitness_column = find_column(
        analysis_df,
        [
            "icu_fitness",
            "fitness",
            "ctl",
        ],
    )

    fatigue_column = find_column(
        analysis_df,
        [
            "icu_fatigue",
            "fatigue",
            "atl",
        ],
    )

    intensity_column = find_column(
        analysis_df,
        [
            "icu_intensity",
            "intensity",
            "if",
        ],
    )

    normalized_power_column = find_column(
        analysis_df,
        [
            "icu_normalized_watts",
            "normalized_watts",
            "normalized_power",
            "np",
        ],
    )

    eftp_column = find_column(
        analysis_df,
        [
            "icu_eftp",
            "eftp",
            "mftp",
        ],
    )

    type_column = find_column(
        analysis_df,
        [
            "type",
            "sport",
            "activity_type",
        ],
    )

    name_column = find_column(
        analysis_df,
        [
            "name",
            "activity_name",
        ],
    )

    def aggregate_period(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> dict:
        """
        指定期間を集計する。
        start_date以上、end_date以下を対象にする。
        """

        period_df = analysis_df[
            (
                analysis_df["_analysis_date"].dt.normalize()
                >= start_date
            )
            & (
                analysis_df["_analysis_date"].dt.normalize()
                <= end_date
            )
        ].copy()

        result = {
            "count": len(period_df),
            "distance_km": 0.0,
            "duration_hours": 0.0,
            "training_load": 0.0,
            "ctl": None,
            "atl": None,
            "tsb": None,
            "average_if": None,
            "average_np": None,
            "latest_eftp": None,
        }

        if distance_column is not None:
            distance_values = pd.to_numeric(
                period_df[distance_column],
                errors="coerce",
            ).fillna(0)

            # Intervals.icuのdistanceをメートルとして処理
            result["distance_km"] = (
                float(distance_values.sum()) / 1000
            )

        if duration_column is not None:
            duration_values = pd.to_numeric(
                period_df[duration_column],
                errors="coerce",
            ).fillna(0)

            # 時間値を秒として処理
            result["duration_hours"] = (
                float(duration_values.sum()) / 3600
            )

        if load_column is not None:
            load_values = pd.to_numeric(
                period_df[load_column],
                errors="coerce",
            ).fillna(0)

            result["training_load"] = float(
                load_values.sum()
            )
        
        # 日付順に並べて、対象期間の最新値を取得する
        sorted_period_df = period_df.sort_values(
            "_analysis_date",
            ascending=True,
        )

        # CTL（フィットネス）の最新値
        if fitness_column is not None:
            ctl_values = pd.to_numeric(
                sorted_period_df[fitness_column],
                errors="coerce",
            ).dropna()

            if not ctl_values.empty:
                result["ctl"] = float(
                    ctl_values.iloc[-1]
                )

        # ATL（ファティーグ）の最新値
        if fatigue_column is not None:
            atl_values = pd.to_numeric(
                sorted_period_df[fatigue_column],
                errors="coerce",
            ).dropna()

            if not atl_values.empty:
                result["atl"] = float(
                    atl_values.iloc[-1]
                )

        # TSB（フォーム）をCTL - ATLで計算
        if (
            result["ctl"] is not None
            and result["atl"] is not None
        ):
            result["tsb"] = (
                result["ctl"] - result["atl"]
            )

        # IF（強度）の期間平均
        if intensity_column is not None:
            intensity_values = pd.to_numeric(
                period_df[intensity_column],
                errors="coerce",
            ).dropna()

            if not intensity_values.empty:
                # Intervals.icuで85のような百分率なら0.85へ変換
                intensity_values = intensity_values.apply(
                    lambda value: value / 100
                    if value > 2
                    else value
                )

                result["average_if"] = float(
                    intensity_values.mean()
                )

        # NP（正規化パワー）の期間平均
        if normalized_power_column is not None:
            np_values = pd.to_numeric(
                period_df[normalized_power_column],
                errors="coerce",
            ).dropna()

            if not np_values.empty:
                result["average_np"] = float(
                    np_values.mean()
                )

        # eFTPの最新値
        if eftp_column is not None:
            eftp_values = pd.to_numeric(
                sorted_period_df[eftp_column],
                errors="coerce",
            ).dropna()

            if not eftp_values.empty:
                result["latest_eftp"] = float(
                    eftp_values.iloc[-1]
                )

        return result

    current_7_start = reference_date - pd.Timedelta(days=6)
    current_7_end = reference_date

    previous_7_start = reference_date - pd.Timedelta(days=13)
    previous_7_end = reference_date - pd.Timedelta(days=7)

    current_28_start = reference_date - pd.Timedelta(days=27)
    current_28_end = reference_date

    current_7 = aggregate_period(
        current_7_start,
        current_7_end,
    )

    previous_7 = aggregate_period(
        previous_7_start,
        previous_7_end,
    )

    current_28 = aggregate_period(
        current_28_start,
        current_28_end,
    )

    lines = [
        "【分析基準】",
        f"基準日: {reference_date:%Y-%m-%d}",
        "",
        "【直近7日】",
        (
            f"対象期間: "
            f"{current_7_start:%Y-%m-%d} ～ "
            f"{current_7_end:%Y-%m-%d}"
        ),
        f"アクティビティ件数: {current_7['count']}件",
        f"合計距離: {current_7['distance_km']:.1f} km",
        f"合計時間: {current_7['duration_hours']:.1f} 時間",
        f"合計負荷: {current_7['training_load']:.1f}",

        (
            f"CTL（フィットネス）: {current_7['ctl']:.1f}"
            if current_7["ctl"] is not None
            else "CTL（フィットネス）: データなし"
        ),
        (
            f"ATL（ファティーグ）: {current_7['atl']:.1f}"
            if current_7["atl"] is not None
            else "ATL（ファティーグ）: データなし"
        ),
        (
            f"TSB（フォーム = CTL - ATL）: "
            f"{current_7['tsb']:+.1f}"
            if current_7["tsb"] is not None
            else "TSB（フォーム）: 計算不可"
        ),
        (
            f"平均IF（強度）: {current_7['average_if']:.2f}"
            if current_7["average_if"] is not None
            else "平均IF（強度）: データなし"
        ),
        (
            f"平均NP（正規化パワー）: "
            f"{current_7['average_np']:.0f} W"
            if current_7["average_np"] is not None
            else "平均NP（正規化パワー）: データなし"
        ),
        (
            f"最新eFTP: {current_7['latest_eftp']:.0f} W"
            if current_7["latest_eftp"] is not None
            else "最新eFTP: データなし"
        ),
        "",
        "【前の7日】",
        (
            f"対象期間: "
            f"{previous_7_start:%Y-%m-%d} ～ "
            f"{previous_7_end:%Y-%m-%d}"
        ),
        f"アクティビティ件数: {previous_7['count']}件",
        f"合計距離: {previous_7['distance_km']:.1f} km",
        f"合計時間: {previous_7['duration_hours']:.1f} 時間",
        f"合計負荷: {previous_7['training_load']:.1f}",
        
        (
            f"CTL（フィットネス）: {previous_7['ctl']:.1f}"
            if previous_7["ctl"] is not None
            else "CTL（フィットネス）: データなし"
        ),
        (
            f"ATL（ファティーグ）: {previous_7['atl']:.1f}"
            if previous_7["atl"] is not None
            else "ATL（ファティーグ）: データなし"
        ),
        (
            f"TSB（フォーム = CTL - ATL）: "
            f"{previous_7['tsb']:+.1f}"
            if previous_7["tsb"] is not None
            else "TSB（フォーム）: 計算不可"
        ),
        (
            f"平均IF（強度）: {previous_7['average_if']:.2f}"
            if previous_7["average_if"] is not None
            else "平均IF（強度）: データなし"
        ),
        (
            f"平均NP（正規化パワー）: "
            f"{previous_7['average_np']:.0f} W"
            if previous_7["average_np"] is not None
            else "平均NP（正規化パワー）: データなし"
        ),
        (
            f"最新eFTP: {previous_7['latest_eftp']:.0f} W"
            if previous_7["latest_eftp"] is not None
            else "最新eFTP: データなし"
        ),
        "",
        "【直近28日】",
        (
            f"対象期間: "
            f"{current_28_start:%Y-%m-%d} ～ "
            f"{current_28_end:%Y-%m-%d}"
        ),
        f"アクティビティ件数: {current_28['count']}件",
        f"合計距離: {current_28['distance_km']:.1f} km",
        f"合計時間: {current_28['duration_hours']:.1f} 時間",
        f"合計負荷: {current_28['training_load']:.1f}",

        (
            f"CTL（フィットネス）: {current_28['ctl']:.1f}"
            if current_28["ctl"] is not None
            else "CTL（フィットネス）: データなし"
        ),
        (
            f"ATL（ファティーグ）: {current_28['atl']:.1f}"
            if current_28["atl"] is not None
            else "ATL（ファティーグ）: データなし"
        ),
        (
            f"TSB（フォーム = CTL - ATL）: "
            f"{current_28['tsb']:+.1f}"
            if current_28["tsb"] is not None
            else "TSB（フォーム）: 計算不可"
        ),
        (
            f"平均IF（強度）: {current_28['average_if']:.2f}"
            if current_28["average_if"] is not None
            else "平均IF（強度）: データなし"
        ),
        (
            f"平均NP（正規化パワー）: "
            f"{current_28['average_np']:.0f} W"
            if current_28["average_np"] is not None
            else "平均NP（正規化パワー）: データなし"
        ),
        (
            f"最新eFTP: {current_28['latest_eftp']:.0f} W"
            if current_28["latest_eftp"] is not None
            else "最新eFTP: データなし"
        ),
    ]

    # 直近7日と前7日の変化率を追加
    comparisons = [
        (
            "距離",
            current_7["distance_km"],
            previous_7["distance_km"],
        ),
        (
            "時間",
            current_7["duration_hours"],
            previous_7["duration_hours"],
        ),
        (
            "負荷",
            current_7["training_load"],
            previous_7["training_load"],
        ),
    ]

    lines.extend([
        "",
        "【前の7日との比較】",
    ])

    for label, current_value, previous_value in comparisons:
        if previous_value > 0:
            change_rate = (
                (current_value - previous_value)
                / previous_value
                * 100
            )

            lines.append(
                f"{label}: {change_rate:+.1f}%"
            )
        else:
            lines.append(
                f"{label}: 比較不可"
                "（前の7日の値が0）"
            )

    # 種目別集計
    recent_28_df = analysis_df[
        (
            analysis_df["_analysis_date"].dt.normalize()
            >= current_28_start
        )
        & (
            analysis_df["_analysis_date"].dt.normalize()
            <= current_28_end
        )
    ].copy()

    if type_column is not None and not recent_28_df.empty:
        lines.extend([
            "",
            "【直近28日の種目別件数】",
        ])

        type_counts = (
            recent_28_df[type_column]
            .fillna("不明")
            .astype(str)
            .value_counts()
        )

        for activity_type, count in type_counts.items():
            lines.append(
                f"{activity_type}: {count}件"
            )

    # 最近の活動を最大10件追加
    recent_df = analysis_df.sort_values(
        "_analysis_date",
        ascending=False,
    ).head(10)

    lines.extend([
        "",
        "【最近のアクティビティ】",
    ])

    for _, row in recent_df.iterrows():
        date_text = row["_analysis_date"].strftime(
            "%Y-%m-%d"
        )

        activity_name = (
            str(row[name_column])
            if name_column is not None
            else "名称不明"
        )

        activity_type = (
            str(row[type_column])
            if type_column is not None
            else "種目不明"
        )

        details = [
            date_text,
            activity_type,
            activity_name,
        ]

        if distance_column is not None:
            distance_value = pd.to_numeric(
                pd.Series([row[distance_column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(distance_value):
                details.append(
                    f"{float(distance_value) / 1000:.1f} km"
                )

        if load_column is not None:
            load_value = pd.to_numeric(
                pd.Series([row[load_column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(load_value):
                details.append(
                    f"負荷 {float(load_value):.1f}"
                )

        lines.append(" / ".join(details))

    return "\n".join(lines)

training_graph = create_training_graph(
    ollama_client=ollama_client,
    activities_builder=build_analysis_data,
    wellness_builder=build_wellness_analysis_data,
    events_builder=build_events_analysis_data,
)

with st.sidebar:
    st.header("接続・処理")
    st.write(f"Athlete ID: `{ATHLETE_ID}`")
    st.write(f"Ollamaモデル: `{OLLAMA_MODEL}`")
    st.write(f"DB: `{DB_PATH.name}`")

    if INTERVALS_API_KEY:
        st.success("Intervals.icu APIキー: 設定済み")
    else:
        st.error("Intervals.icu APIキー: 未設定")

    if st.button("接続状態を確認", width="stretch"):
        check_status = st.status("接続確認中", expanded=True)
        try:
            check_status.write("1/2 Ollamaサーバーへ接続しています")
            if not ollama_client.check_connection():
                raise ConnectionError(f"Ollamaに接続できません: {OLLAMA_BASE_URL}")
            check_status.write("✅ Ollama接続成功")

            check_status.write("2/2 指定モデルを確認しています")
            models = ollama_client.get_installed_models()
            check_status.write("インストール済み: " + ", ".join(models))
            if not ollama_client.is_model_installed():
                raise RuntimeError(f"指定モデルが見つかりません: {OLLAMA_MODEL}")
            check_status.write(f"✅ モデル確認成功: {OLLAMA_MODEL}")
            check_status.update(label="接続確認が完了しました", state="complete")
        except Exception as error:
            check_status.write(f"❌ {error}")
            check_status.update(label="接続確認に失敗しました", state="error", expanded=True)

    fetch_button = st.button("Intervals.icuから取得", type="primary", width="stretch")

if fetch_button:
    progress_box = st.status("Intervals.icuデータを更新しています",expanded=True,)

    progress = st.progress(0,text="開始しています")

    try:
        if not INTERVALS_API_KEY:raise ValueError("config.iniにAPI_KEYを設定してください。")

        progress_box.write("1/6 取得期間を確認しています")
        progress.progress(5,text="取得期間を確認中")
        progress_box.write(
            f"activity / wellness: "
            f"{OLDEST_DATE} ～ {NEWEST_DATE}"
        )
        progress_box.write(
            f"events: "
            f"{OLDEST_DATE} ～ {EVENT_NEWEST_DATE}"
        )

        # ----------------------------------------
        # Activities
        # ----------------------------------------

        progress_box.write("2/6 アクティビティを取得しています")
        progress.progress(15,text="activities取得中")
        activities_api_df = (intervals_client.get_activities(oldest=OLDEST_DATE,newest=NEWEST_DATE))
        saved_activities = save_activities(DB_PATH,activities_api_df)
        progress_box.write(
            f"✅ activities: "
            f"{saved_activities:,}件保存"
        )
        progress.progress(40,text="activities保存完了")

        # ----------------------------------------
        # Wellness
        # ----------------------------------------

        progress_box.write("3/6 Wellnessを取得しています")
        wellness_api_df = (intervals_client.get_wellness(oldest=OLDEST_DATE,newest=NEWEST_DATE))
        # print(wellness_api_df.dtypes)

        # print()
        # print("===== LIST COLUMN CHECK =====")

        # for column in wellness_api_df.columns:
        #     sample = wellness_api_df[column].dropna()

        #     if len(sample) == 0:
        #         continue

        #     first_value = sample.iloc[0]

        #     if isinstance(first_value, list):
        #         print(
        #             f"{column}: LIST"
        #         )

        #     elif isinstance(first_value, dict):
        #         print(
        #             f"{column}: DICT"
        #         )
        wellness_save_result = save_wellness(DB_PATH,wellness_api_df)

        saved_wellness = (wellness_save_result["wellness"])

        saved_sport_info = (wellness_save_result["sport_info"])

        progress_box.write(
            f"✅ wellness: "
            f"{saved_wellness:,}件保存"
        )

        progress_box.write(
            f"✅ wellness_sport_info: "
            f"{saved_sport_info:,}件保存"
        )
        progress.progress(65,text="wellness保存完了")

        # ----------------------------------------
        # Events
        # ----------------------------------------

        progress_box.write("4/6 イベントを取得しています")
        events_api_df = (intervals_client.get_events(oldest=OLDEST_DATE,newest=EVENT_NEWEST_DATE,category=None))
        saved_events = save_events(DB_PATH,events_api_df)
        progress_box.write(
            f"✅ events: "
            f"{saved_events:,}件保存"
        )
        progress.progress(85,text="events保存完了")

        # ----------------------------------------
        # 完了
        # ----------------------------------------

        progress_box.write("5/6 保存結果を確認しています")
        progress_box.write(f"activities: {saved_activities:,}件")
        progress_box.write(f"wellness: {saved_wellness:,}件")
        progress_box.write(
            "wellness_sport_info: "
            f"{saved_sport_info:,}件"
        )
        progress_box.write(f"events: {saved_events:,}件")
        progress.progress(95,text="保存結果を確認中",)
        progress_box.write("6/6 画面を更新します")
        progress.progress(100,text="すべて完了しました")
        progress_box.update(label="3種類のデータ更新が完了しました",state="complete",expanded=True)
        st.rerun()

    except Exception as error:
        progress.progress(
            100,
            text="エラーで停止しました",
        )

        progress_box.write(
            f"❌ {type(error).__name__}: {error}"
        )

        progress_box.code(
            traceback.format_exc()
        )

        progress_box.update(
            label="データ更新に失敗しました",
            state="error",
            expanded=True,
        )

if activities_df.empty:
    
    st.info("保存済みデータがありません。サイドバーの「Intervals.icuから取得」を押してください。")
else:
    summary = create_summary(activities_df)
    summary_text = summary_to_text(summary)

    st.subheader("データ概要")
    cols = st.columns(4)
    cols[0].metric("件数", f"{summary['activity_count']:,}件")
    cols[1].metric("合計距離", "列なし" if summary["distance_km"] is None else f"{summary['distance_km']:,.1f} km")
    cols[2].metric("合計時間", "列なし" if summary["duration_hours"] is None else f"{summary['duration_hours']:,.1f}時間")
    cols[3].metric("負荷", "列なし" if summary["training_load"] is None else f"{summary['training_load']:,.1f}")

    st.subheader("SQLite保存データ")

    activity_tab, wellness_tab, sport_info_tab, events_tab = st.tabs(
        [
            "Activities",
            "Wellness",
            "Wellness Sport Info",
            "Events",
        ]
    )
    with activity_tab:
        st.write(f"{len(activities_df):,}件")
        if activities_df.empty:st.info("Activitiesデータがありません。")
        else:
            st.dataframe(
                activities_df,
                width="stretch",
                hide_index=True,
            )
    with wellness_tab:
        st.write(f"{len(wellness_df):,}件")
        if wellness_df.empty:st.info("Wellnessデータがありません。")
        else:
            st.dataframe(
                wellness_df,
                width="stretch",
                hide_index=True,
            )
    with sport_info_tab:
        st.write(
            f"{len(wellness_sport_info_df):,}件"
        )

        if wellness_sport_info_df.empty:
            st.info(
                "Wellness Sport Infoデータがありません。"
            )

        else:
            st.dataframe(
                wellness_sport_info_df,
                width="stretch",
                hide_index=True,
            )
    with events_tab:
        st.write(f"{len(events_df):,}件")
        if events_df.empty:
            st.info(
                "Eventsデータがありません。"
            )
        else:
            st.dataframe(
                events_df,
                width="stretch",
                hide_index=True,
            )
    
    st.subheader("LangGraph + Ollama分析")

    request = st.text_area(
        "AIへの依頼",
        value=(
            "トレーニング傾向を分析し、"
            "次回に向けた一般的な提案をしてください。"
        ),
        height=100,
    )

    if st.button(
        "LangGraphで分析",
        type="primary",
        width="stretch",
    ):
        agent_status = st.status(
            "LangGraph分析を開始しています",
            expanded=True,
        )

        agent_progress = st.progress(
            0,
            text="準備中",
        )

        try:
            # LangGraphへ最初に渡すデータ
            initial_state = {
                "activities_df": activities_df,
                "wellness_df": wellness_df,
                "events_df": events_df,
                "user_request": request,
                "current_step": "分析開始",
                "progress": 0,
            }

            # 各ノードの結果を保存する辞書
            accumulated_state = {}

            # LangGraphを実行する
            for event in training_graph.stream(
                initial_state,
                stream_mode="updates",
            ):
                # eventの例:
                # {
                #     "validate_data": {
                #         "current_step": "データ検証完了",
                #         "progress": 20
                #     }
                # }

                for node_name, update in event.items():
                    # updateが辞書か確認
                    if not isinstance(update, dict):
                        continue

                    # 今までの結果へ今回の更新を追加
                    accumulated_state.update(update)

                    current_step = update.get(
                        "current_step",
                        node_name,
                    )

                    progress_value = update.get(
                        "progress",
                        0,
                    )

                    # 進行状況を画面へ表示
                    agent_status.write(
                        f"✅ {node_name}: {current_step}"
                    )

                    agent_progress.progress(
                        progress_value,
                        text=current_step,
                    )

            # LangGraphが返した最終回答を取得
            answer = accumulated_state.get(
                "answer"
            )

            if not answer:
                raise RuntimeError(
                    "LangGraphの実行結果に"
                    "answerが含まれていません。"
                )

            agent_progress.progress(
                100,
                text="分析完了",
            )

            agent_status.update(
                label="LangGraph分析が完了しました",
                state="complete",
                expanded=False,
            )

            st.markdown("### 分析結果")
            st.markdown(answer)

        except ConnectionError as error:
            agent_progress.progress(
                100,
                text="接続エラーで停止しました",
            )

            agent_status.write(
                f"❌ 接続エラー: {error}"
            )

            agent_status.update(
                label="Ollamaへの接続に失敗しました",
                state="error",
                expanded=True,
            )

        except TimeoutError as error:
            agent_progress.progress(
                100,
                text="タイムアウトで停止しました",
            )

            agent_status.write(
                f"❌ タイムアウト: {error}"
            )

            agent_status.update(
                label="分析がタイムアウトしました",
                state="error",
                expanded=True,
            )

        except Exception as error:
            agent_progress.progress(
                100,
                text="エラーで停止しました",
            )

            agent_status.write(
                f"❌ {type(error).__name__}: {error}"
            )

            agent_status.code(
                traceback.format_exc()
            )

            agent_status.update(
                label="LangGraph分析に失敗しました",
                state="error",
                expanded=True,
            )