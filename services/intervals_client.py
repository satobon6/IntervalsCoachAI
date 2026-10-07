from io import StringIO

import pandas as pd
import requests


class IntervalsClient:
    """
    Intervals.icu APIクライアント。

    activities、wellness、eventsを取得する。
    """

    BASE_URL = "https://intervals.icu/api/v1"

    def __init__(
        self,
        athlete_id: str,
        api_key: str,
    ):
        self.athlete_id = athlete_id
        self.api_key = api_key

    def _get(
        self,
        endpoint: str,
        params: dict | None = None,
        timeout: int = 60,
    ) -> requests.Response:
        """
        Intervals.icuへ共通GETリクエストを送信する。
        """

        url = f"{self.BASE_URL}{endpoint}"

        response = requests.get(
            url,
            auth=("API_KEY", self.api_key),
            params=params,
            timeout=timeout,
        )

        response.raise_for_status()

        return response

    def get_activities(
        self,
        oldest: str | None = None,
        newest: str | None = None,
    ) -> pd.DataFrame:
        """
        完了済みアクティビティをCSV形式で取得する。
        """

        endpoint = (
            f"/athlete/{self.athlete_id}/activities.csv"
        )

        params = {}

        if oldest:
            params["oldest"] = oldest

        if newest:
            params["newest"] = newest

        response = self._get(
            endpoint=endpoint,
            params=params,
            timeout=120,
        )

        content_type = response.headers.get(
            "Content-Type",
            "",
        )

        if "html" in content_type.lower():
            raise ValueError(
                "activities APIからCSVではなく"
                "HTMLが返されました。"
            )

        csv_text = response.text.strip()

        if not csv_text:
            return pd.DataFrame()

        return pd.read_csv(
            StringIO(csv_text)
        )

    def get_wellness(
        self,
        oldest: str,
        newest: str,
    ) -> pd.DataFrame:
        """
        日別Wellnessデータを取得する。
        """

        endpoint = (
            f"/athlete/{self.athlete_id}/wellness"
        )

        params = {
            "oldest": oldest,
            "newest": newest,
        }

        response = self._get(
            endpoint=endpoint,
            params=params,
            timeout=60,
        )

        result = response.json()

        if result is None:
            return pd.DataFrame()

        if isinstance(result, dict):
            result = [result]

        if not isinstance(result, list):
            raise ValueError(
                "wellness APIから想定外の形式が"
                "返されました。"
            )

        return pd.json_normalize(
            result,
            sep="_",
        )

    def get_events(
        self,
        oldest: str,
        newest: str,
        category: str | None = None,
    ) -> pd.DataFrame:
        """
        カレンダーイベントを取得する。

        categoryをWORKOUTにすると予定ワークアウトだけを取得。
        Noneの場合は期間内の全イベントを取得する。
        """

        endpoint = (
            f"/athlete/{self.athlete_id}/events"
        )

        params = {
            "oldest": oldest,
            "newest": newest,
        }

        if category:
            params["category"] = category

        response = self._get(
            endpoint=endpoint,
            params=params,
            timeout=60,
        )

        result = response.json()

        if result is None:
            return pd.DataFrame()

        if isinstance(result, dict):
            result = [result]

        if not isinstance(result, list):
            raise ValueError(
                "events APIから想定外の形式が"
                "返されました。"
            )

        return pd.json_normalize(
            result,
            sep="_",
        )