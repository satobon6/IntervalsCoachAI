from pathlib import Path
import json
import sqlite3

import pandas as pd

ACTIVITIES_TABLE = "activities"
WELLNESS_TABLE = "wellness"
WELLNESS_SPORT_INFO_TABLE = "wellness_sport_info"
EVENTS_TABLE = "events"

def initialize_database(
    db_path: Path,
) -> None:
    """
    SQLiteファイルと管理テーブルを初期化する。
    """

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_info (
                name TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS wellness_sport_info (
                wellness_id TEXT NOT NULL,
                sport_type TEXT NOT NULL,
                eftp REAL,
                data_json TEXT,
                PRIMARY KEY (
                    wellness_id,
                    sport_type
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_wellness_sport_info_date
            ON wellness_sport_info (
                wellness_id
            )
            """
        )

        connection.commit()

def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    SQLiteへ保存できる形式にDataFrameを正規化する。

    処理内容:
    - 列名をSQLite向けに正規化
    - list、dict、tupleをJSON文字列へ変換
    - pandasの欠損値をNoneへ変換
    """

    normalized_df = dataframe.copy()

    # ----------------------------------------
    # 列名を正規化
    # ----------------------------------------

    normalized_df.columns = [
        str(column)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        for column in normalized_df.columns
    ]

    # ----------------------------------------
    # SQLite非対応型を変換
    # ----------------------------------------

    def convert_value(value):
        """
        SQLiteへ保存可能な値へ変換する。
        """

        if isinstance(
            value,
            (list, dict, tuple),
        ):
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        # pandas、NumPyの欠損値をNoneへ変換
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            # list等にpd.isnaを使った場合への保険
            pass

        return value

    for column in normalized_df.columns:
        normalized_df[column] = (
            normalized_df[column]
            .map(convert_value)
        )

    return normalized_df

def split_wellness_sport_info(
    wellness_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wellness DataFrameを次の2つへ分離する。

    1. wellness本体
    2. wellness_sport_info

    Returns:
        tuple:
            wellness本体のDataFrame
            sportInfoのDataFrame
    """

    if wellness_df.empty:
        return (
            wellness_df.copy(),
            pd.DataFrame(),
        )

    main_df = wellness_df.copy()

    sport_info_records = []

    if "sportInfo" not in main_df.columns:
        return (
            main_df,
            pd.DataFrame(),
        )

    for _, wellness_row in main_df.iterrows():
        wellness_id = wellness_row.get("id")
        sport_info_value = wellness_row.get(
            "sportInfo"
        )

        if pd.isna(wellness_id):
            continue

        if sport_info_value is None:
            continue

        # SQLiteから読み戻したJSON文字列にも対応
        if isinstance(sport_info_value, str):
            try:
                sport_info_value = json.loads(
                    sport_info_value
                )
            except json.JSONDecodeError:
                continue

        if not isinstance(
            sport_info_value,
            list,
        ):
            continue

        for sport_item in sport_info_value:
            if not isinstance(
                sport_item,
                dict,
            ):
                continue

            sport_type = (
                sport_item.get("type")
                or sport_item.get("sport")
                or "Unknown"
            )

            eftp = sport_item.get("eftp")

            sport_info_records.append(
                {
                    "wellness_id": str(
                        wellness_id
                    ),
                    "sport_type": str(
                        sport_type
                    ),
                    "eftp": eftp,
                    "data_json": json.dumps(
                        sport_item,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

    # 親テーブルからリスト列を削除
    main_df = main_df.drop(
        columns=["sportInfo"],
        errors="ignore",
    )

    sport_info_df = pd.DataFrame(
        sport_info_records
    )

    return (
        main_df,
        sport_info_df,
    )

def save_dataframe(
    db_path: Path,
    table_name: str,
    dataframe: pd.DataFrame,
) -> int:
    """
    DataFrameを指定テーブルへ保存する。
    """

    if dataframe.empty:
        return 0

    save_df = normalize_columns(
        dataframe
    )

    # SQLite非対応型が残っていないか検証
    for column in save_df.columns:
        complex_values = save_df[column].map(
            lambda value: isinstance(
                value,
                (list, dict, tuple),
            )
        )

        if complex_values.any():
            raise TypeError(
                f"SQLiteへ保存できない値が残っています。"
                f"列名: {column}"
            )

    with sqlite3.connect(db_path) as connection:
        save_df.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )

    return len(save_df)

def load_dataframe(
    db_path: Path,
    table_name: str,
) -> pd.DataFrame:
    """
    指定テーブルをDataFrameとして読み込む。
    """

    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (table_name,),
        ).fetchone()

        table_exists = result[0] > 0

        if not table_exists:
            return pd.DataFrame()

        dataframe = pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            connection,
        )

    return dataframe

def save_activities(
    db_path: Path,
    dataframe: pd.DataFrame,
) -> int:
    return save_dataframe(
        db_path,
        ACTIVITIES_TABLE,
        dataframe,
    )

def save_wellness(
    db_path: Path,
    dataframe: pd.DataFrame,
) -> dict:
    """
    WellnessとsportInfoをトランザクション内で保存する。
    """

    if dataframe.empty:
        return {
            "wellness": 0,
            "sport_info": 0,
        }

    wellness_main_df, sport_info_df = (
        split_wellness_sport_info(
            dataframe
        )
    )

    wellness_main_df = normalize_columns(
        wellness_main_df
    )

    if not sport_info_df.empty:
        sport_info_df = normalize_columns(
            sport_info_df
        )

        sport_info_df["eftp"] = pd.to_numeric(
            sport_info_df["eftp"],
            errors="coerce",
        )

    with sqlite3.connect(db_path) as connection:
        # 親テーブルは現段階では全置換
        wellness_main_df.to_sql(
            WELLNESS_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )

        # 子テーブルは明示した構造を維持
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            wellness_sport_info (
                wellness_id TEXT NOT NULL,
                sport_type TEXT NOT NULL,
                eftp REAL,
                data_json TEXT,
                PRIMARY KEY (
                    wellness_id,
                    sport_type
                )
            )
            """
        )

        connection.execute(
            """
            DELETE FROM wellness_sport_info
            """
        )

        if not sport_info_df.empty:
            sport_info_df.to_sql(
                WELLNESS_SPORT_INFO_TABLE,
                connection,
                if_exists="append",
                index=False,
            )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_wellness_sport_info_date
            ON wellness_sport_info (
                wellness_id
            )
            """
        )

        connection.commit()

    return {
        "wellness": len(
            wellness_main_df
        ),
        "sport_info": len(
            sport_info_df
        ),
    }

def save_events(
    db_path: Path,
    dataframe: pd.DataFrame,
) -> int:
    return save_dataframe(
        db_path,
        EVENTS_TABLE,
        dataframe,
    )

def load_activities(
    db_path: Path,
) -> pd.DataFrame:
    return load_dataframe(
        db_path,
        ACTIVITIES_TABLE,
    )

def load_wellness(
    db_path: Path,
) -> pd.DataFrame:
    return load_dataframe(
        db_path,
        WELLNESS_TABLE,
    )

def load_wellness_sport_info(
    db_path: Path,
) -> pd.DataFrame:
    """
    wellness_sport_infoテーブルを読み込む。
    """

    return load_dataframe(
        db_path,
        WELLNESS_SPORT_INFO_TABLE,
    )

def load_events(
    db_path: Path,
) -> pd.DataFrame:
    return load_dataframe(
        db_path,
        EVENTS_TABLE,
    )