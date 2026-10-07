from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def to_number(
    value: Any,
) -> float | None:
    """
    値をfloatへ変換する。
    変換できない場合はNoneを返す。
    """

    result = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(result):
        return None

    return float(result)


def prepare_wellness_dataframe(
    wellness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wellnessを日付順に並べ、
    分析用日付列を追加する。
    """

    if wellness_df.empty:
        return pd.DataFrame()

    if "id" not in wellness_df.columns:
        return pd.DataFrame()

    working_df = wellness_df.copy()

    working_df["_date"] = pd.to_datetime(
        working_df["id"],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=["_date"]
    )

    return working_df.sort_values(
        "_date",
        ascending=True,
    )


def latest_numeric_value(
    dataframe: pd.DataFrame,
    column: str,
) -> tuple[float | None, pd.Timestamp | None]:
    """
    指定列の最新有効値と、その日付を返す。
    """

    if dataframe.empty:
        return None, None

    if column not in dataframe.columns:
        return None, None

    valid_df = dataframe[
        ["_date", column]
    ].copy()

    valid_df[column] = pd.to_numeric(
        valid_df[column],
        errors="coerce",
    )

    valid_df = valid_df.dropna(
        subset=[column]
    )

    if valid_df.empty:
        return None, None

    latest_row = valid_df.iloc[-1]

    return (
        float(latest_row[column]),
        latest_row["_date"],
    )


def period_average(
    dataframe: pd.DataFrame,
    column: str,
    days: int,
    reference_date: pd.Timestamp,
) -> float | None:
    """
    指定期間の数値平均を返す。
    """

    if dataframe.empty:
        return None

    if column not in dataframe.columns:
        return None

    start_date = (
        reference_date.normalize()
        - pd.Timedelta(days=days - 1)
    )

    period_df = dataframe[
        (
            dataframe["_date"].dt.normalize()
            >= start_date
        )
        & (
            dataframe["_date"].dt.normalize()
            <= reference_date.normalize()
        )
    ].copy()

    values = pd.to_numeric(
        period_df[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def get_latest_eftp(
    sport_info_df: pd.DataFrame,
    preferred_sport: str,
) -> tuple[float | None, str | None]:
    """
    指定種目の最新eFTPを取得する。
    """

    required_columns = {
        "wellness_id",
        "sport_type",
        "eftp",
    }

    if sport_info_df.empty:
        return None, None

    if not required_columns.issubset(
        sport_info_df.columns
    ):
        return None, None

    working_df = sport_info_df.copy()

    working_df["_date"] = pd.to_datetime(
        working_df["wellness_id"],
        errors="coerce",
    )

    working_df["eftp"] = pd.to_numeric(
        working_df["eftp"],
        errors="coerce",
    )

    working_df = working_df[
        working_df["sport_type"]
        .astype(str)
        .str.casefold()
        == preferred_sport.casefold()
    ]

    working_df = working_df.dropna(
        subset=["_date", "eftp"]
    ).sort_values("_date")

    if working_df.empty:
        return None, None

    latest_row = working_df.iloc[-1]

    return (
        float(latest_row["eftp"]),
        latest_row["_date"].strftime(
            "%Y-%m-%d"
        ),
    )


def get_next_event(
    events_df: pd.DataFrame,
) -> dict | None:
    """
    今日以降で最も近いイベントを返す。
    """

    if events_df.empty:
        return None

    date_column = None

    for candidate in (
        "start_date_local",
        "start_date",
        "date",
    ):
        if candidate in events_df.columns:
            date_column = candidate
            break

    if date_column is None:
        return None

    working_df = events_df.copy()

    working_df["_event_date"] = pd.to_datetime(
        working_df[date_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=["_event_date"]
    )

    today = pd.Timestamp(
        date.today()
    ).normalize()

    future_df = working_df[
        working_df["_event_date"].dt.normalize()
        >= today
    ].sort_values("_event_date")

    if future_df.empty:
        return None

    event_row = future_df.iloc[0]
    event_date = event_row["_event_date"]

    return {
        "date": event_date.strftime(
            "%Y-%m-%d"
        ),
        "days_until": int(
            (
                event_date.normalize()
                - today
            ).days
        ),
        "name": str(
            event_row.get(
                "name",
                "名称なし",
            )
        ),
        "category": str(
            event_row.get(
                "category",
                "分類なし",
            )
        ),
        "type": str(
            event_row.get(
                "type",
                "",
            )
        ),
    }


def classify_coach_status(
    wellness_df: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Wellnessデータを個人内基準と比較し、
    メニュー選択用の状態を作成する。

    医学的診断は行わない。
    """

    working_df = prepare_wellness_dataframe(
        wellness_df
    )

    if working_df.empty:
        return {
            "status": "DATA_LIMITED",
            "reason_codes": [
                "WELLNESS_DATA_MISSING"
            ],
            "metrics": {},
        }

    reference_date = (
        working_df["_date"]
        .max()
        .normalize()
    )

    metrics = {}
    reason_codes = []

    for column in (
        "ctl",
        "atl",
        "hrv",
        "restingHR",
        "sleepSecs",
        "sleepScore",
    ):
        latest_value, latest_date = (
            latest_numeric_value(
                working_df,
                column,
            )
        )

        average_28 = period_average(
            working_df,
            column,
            28,
            reference_date,
        )

        metrics[column] = {
            "latest": latest_value,
            "latest_date": (
                latest_date.strftime(
                    "%Y-%m-%d"
                )
                if latest_date is not None
                else None
            ),
            "average_28": average_28,
        }

    ctl = metrics["ctl"]["latest"]
    atl = metrics["atl"]["latest"]

    tsb = None

    if ctl is not None and atl is not None:
        tsb = ctl - atl

    metrics["tsb"] = tsb

    sleep_seconds = (
        metrics["sleepSecs"]["latest"]
    )

    sleep_hours = (
        sleep_seconds / 3600
        if sleep_seconds is not None
        else None
    )

    metrics["sleep_hours"] = sleep_hours

    if (
        sleep_hours is not None
        and sleep_hours
        < config["min_sleep_hours"]
    ):
        reason_codes.append(
            "SLEEP_BELOW_USER_LIMIT"
        )

    hrv_latest = metrics["hrv"]["latest"]
    hrv_average = metrics["hrv"]["average_28"]

    if (
        hrv_latest is not None
        and hrv_average is not None
        and hrv_average > 0
        and hrv_latest
        < hrv_average
        * (
            1
            - config["hrv_drop_ratio"]
        )
    ):
        reason_codes.append(
            "HRV_BELOW_PERSONAL_BASELINE"
        )

    resting_latest = (
        metrics["restingHR"]["latest"]
    )

    resting_average = (
        metrics["restingHR"]["average_28"]
    )

    if (
        resting_latest is not None
        and resting_average is not None
        and resting_latest
        > resting_average
        + config["resting_hr_rise"]
    ):
        reason_codes.append(
            "RESTING_HR_ABOVE_PERSONAL_BASELINE"
        )

    if (
        tsb is not None
        and tsb < config["tsb_low_limit"]
    ):
        reason_codes.append(
            "TSB_BELOW_USER_LIMIT"
        )

    # 複数の注意信号が重なった場合は
    # 高強度を提示しない。
    if len(reason_codes) >= 2:
        status = "RECOVERY"

    elif len(reason_codes) == 1:
        status = "EASY"

    else:
        status = "NORMAL"

    return {
        "status": status,
        "reason_codes": reason_codes,
        "metrics": metrics,
        "reference_date": (
            reference_date.strftime(
                "%Y-%m-%d"
            )
        ),
    }


def create_workout_candidate(
    status: str,
    available_minutes: int,
    eftp: float | None,
    config: dict,
) -> dict:
    """
    判定状態からメニュー候補を1つ作成する。

    最終判断ではなく、利用者が確認する候補。
    """

    if status == "RECOVERY":
        duration = min(
            available_minutes,
            config["recovery_duration"],
        )

        workout = {
            "workout_type": "RECOVERY",
            "title": "低強度の回復目的ライド",
            "duration_minutes": duration,
            "target_if": 0.55,
            "estimated_tss": round(
                (duration / 60)
                * (0.55 ** 2)
                * 100
            ),
            "steps": [
                {
                    "name": "ウォームアップ",
                    "minutes": 10,
                    "intensity": "非常に軽い強度",
                },
                {
                    "name": "メイン",
                    "minutes": max(
                        duration - 20,
                        10,
                    ),
                    "intensity": "Z1～低めのZ2",
                },
                {
                    "name": "クールダウン",
                    "minutes": 10,
                    "intensity": "非常に軽い強度",
                },
            ],
        }

    elif status == "EASY":
        duration = min(
            available_minutes,
            config["endurance_duration"],
        )

        workout = {
            "workout_type": "ENDURANCE",
            "title": "低～中強度の持久走",
            "duration_minutes": duration,
            "target_if": 0.65,
            "estimated_tss": round(
                (duration / 60)
                * (0.65 ** 2)
                * 100
            ),
            "steps": [
                {
                    "name": "ウォームアップ",
                    "minutes": 10,
                    "intensity": "Z1",
                },
                {
                    "name": "メイン",
                    "minutes": max(
                        duration - 20,
                        20,
                    ),
                    "intensity": "Z2",
                },
                {
                    "name": "クールダウン",
                    "minutes": 10,
                    "intensity": "Z1",
                },
            ],
        }

    elif status == "NORMAL":
        duration = min(
            available_minutes,
            config["tempo_duration"],
        )

        workout = {
            "workout_type": "TEMPO",
            "title": "テンポ～SST候補",
            "duration_minutes": duration,
            "target_if": 0.80,
            "estimated_tss": round(
                (duration / 60)
                * (0.80 ** 2)
                * 100
            ),
            "steps": [
                {
                    "name": "ウォームアップ",
                    "minutes": 15,
                    "intensity": "Z1～Z2",
                },
                {
                    "name": "メイン1",
                    "minutes": 10,
                    "intensity": "FTPの88～92%",
                },
                {
                    "name": "回復",
                    "minutes": 5,
                    "intensity": "軽い強度",
                },
                {
                    "name": "メイン2",
                    "minutes": 10,
                    "intensity": "FTPの88～92%",
                },
                {
                    "name": "クールダウン",
                    "minutes": 10,
                    "intensity": "Z1",
                },
            ],
        }

    else:
        duration = min(
            available_minutes,
            config["recovery_duration"],
        )

        workout = {
            "workout_type": "DATA_LIMITED",
            "title": "データ不足時の軽いライド候補",
            "duration_minutes": duration,
            "target_if": None,
            "estimated_tss": None,
            "steps": [
                {
                    "name": "全体",
                    "minutes": duration,
                    "intensity": (
                        "会話可能な軽い強度。"
                        "状態を確認しながら実施"
                    ),
                }
            ],
        }

    # eFTPがある場合だけ目標Wを追加する。
    if eftp is not None:
        workout["eftp"] = round(eftp)

        for step in workout["steps"]:
            intensity = step["intensity"]

            if "88～92%" in intensity:
                step["target_watts"] = {
                    "min": round(
                        eftp * 0.88
                    ),
                    "max": round(
                        eftp * 0.92
                    ),
                }

            elif intensity == "Z2":
                step["target_watts"] = {
                    "min": round(
                        eftp * 0.60
                    ),
                    "max": round(
                        eftp * 0.70
                    ),
                }

    return workout


def build_coach_recommendation(
    wellness_df: pd.DataFrame,
    sport_info_df: pd.DataFrame,
    events_df: pd.DataFrame,
    available_minutes: int,
    preferred_sport: str,
    config: dict,
) -> dict:
    """
    コーチAIへ渡す構造化された提案データを作成する。
    """

    status_result = classify_coach_status(
        wellness_df,
        config,
    )

    eftp, eftp_date = get_latest_eftp(
        sport_info_df,
        preferred_sport,
    )

    next_event = get_next_event(
        events_df
    )

    status = status_result["status"]

    # 直近イベントがある場合は、強度を上げない。
    # 日数は利用者設定に移行してもよい。
    if (
        next_event is not None
        and next_event["days_until"] <= 2
        and status == "NORMAL"
    ):
        status = "EASY"

        status_result[
            "reason_codes"
        ].append(
            "EVENT_WITHIN_TWO_DAYS"
        )

    workout = create_workout_candidate(
        status=status,
        available_minutes=available_minutes,
        eftp=eftp,
        config=config,
    )

    return {
        "coach_status": status,
        "reason_codes": (
            status_result["reason_codes"]
        ),
        "reference_date": (
            status_result.get(
                "reference_date"
            )
        ),
        "metrics": status_result["metrics"],
        "preferred_sport": preferred_sport,
        "eftp": eftp,
        "eftp_date": eftp_date,
        "next_event": next_event,
        "workout": workout,
        "disclaimer": (
            "この内容はトレーニング候補です。"
            "健康状態の診断ではありません。"
            "痛み、強い不調、普段と異なる症状が"
            "ある場合は実施せず、適切な専門家へ"
            "相談してください。"
        ),
    }