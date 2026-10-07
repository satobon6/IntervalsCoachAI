import requests


class OllamaClient:
    """
    OllamaのローカルAPIを呼び出すためのクライアント。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def check_connection(self) -> bool:
        """
        Ollamaサーバーに接続できるか確認する。

        Returns:
            bool:
                接続成功時はTrue。
                接続失敗時はFalse。
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )

            response.raise_for_status()
            return True

        except requests.RequestException:
            return False

    def get_installed_models(self) -> list:
        """
        Ollamaにインストールされているモデル名を取得する。

        Returns:
            listインストール済みモデル名のリスト。
        """
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        return [
            model["name"]
            for model in result.get("models", [])
            if "name" in model
        ]

    def is_model_installed(self) -> bool:
        """
        config.iniで指定されたモデルが
        インストール済みか確認する。

        Returns:
            bool:
                指定モデルが存在する場合はTrue。
        """
        try:
            installed_models = self.get_installed_models()

            configured_name = self.model.lower()

            for installed_model in installed_models:
                installed_name = installed_model.lower()

                if installed_name == configured_name:
                    return True

                # llama3.1 と llama3.1:latest の差を吸収
                if installed_name.split(":")[0] == configured_name:
                    return True

            return False

        except requests.RequestException:
            return False

    def analyze(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Ollamaへプロンプトを送信して、
        生成された文章を返す。

        Args:
            prompt:
                Ollamaへ送信するユーザープロンプト。

            system_prompt:
                モデルの役割や回答ルール。
                指定しない場合は既定値を使用する。

        Returns:
            str:
                Ollamaが生成した回答。
        """
        if not prompt or not prompt.strip():
            raise ValueError(
                "Ollamaへ送信するプロンプトが空です。"
            )

        if system_prompt is None:
            system_prompt = (
                "あなたはトレーニングデータを分析する"
                "日本語のアシスタントです。"
                "提示されていない数値は推測せず、"
                "客観的で分かりやすく回答してください。"
                "医学的な診断は行わないでください。"
            )

        request_data = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=300,
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError as error:
            raise ConnectionError(
                "Ollamaに接続できません。"
                "Ollamaが起動していることを確認してください。"
            ) from error

        except requests.exceptions.Timeout as error:
            raise TimeoutError(
                "Ollamaからの応答がタイムアウトしました。"
                "軽量モデルへの変更を検討してください。"
            ) from error

        except requests.exceptions.HTTPError as error:
            status_code = response.status_code
            response_text = response.text[:500]

            raise RuntimeError(
                "Ollama APIでHTTPエラーが発生しました。\n"
                f"ステータスコード: {status_code}\n"
                f"応答: {response_text}"
            ) from error

        except requests.RequestException as error:
            raise RuntimeError(
                f"Ollamaとの通信に失敗しました: {error}"
            ) from error

        try:
            result = response.json()

        except requests.exceptions.JSONDecodeError as error:
            raise ValueError(
                "OllamaからJSON形式ではない応答が返されました。"
            ) from error

        generated_text = result.get("response")

        if generated_text is None:
            raise ValueError(
                "Ollamaの応答に'response'がありません。\n"
                f"取得した応答: {result}"
            )

        return generated_text.strip()