# Intervals.icu AIコーチシステム

Intervals.icuからトレーニング実績、日別のWellness情報、カレンダーイベントを取得し、SQLiteへ保存したうえで、Streamlit、LangGraph、Ollamaを組み合わせて分析するローカルAIアプリケーションです。

現在は「トレーニング分析AI」として動作しており、今後は現在の状態、今後の予定、目標を踏まえて具体的な次回メニューを提案する「コーチAI」への拡張を予定しています。

---

## 1. システム概要

```text
Intervals.icu API
    │
    ├─ Activities
    ├─ Wellness
    └─ Events
    │
    ▼
SQLite
    ├─ activities
    ├─ wellness
    ├─ wellness_sport_info
    └─ events
    │
    ▼
Streamlit
    │
    ▼
LangGraph
    ├─ validate_data
    ├─ prepare_analysis_data
    ├─ build_prompt
    └─ call_ollama
    │
    ▼
Ollama
    │
    ▼
日本語Markdown形式の分析結果
```

### 各コンポーネントの役割

| コンポーネント | 役割 |
|---|---|
| Intervals.icu | アクティビティ、Wellness、Eventsの提供 |
| Python / pandas | API応答の読込み、変換、期間集計 |
| SQLite | 取得データのローカル保存 |
| Streamlit | データ取得操作、一覧表示、進捗表示、分析結果表示 |
| LangGraph | 分析処理の状態管理と実行順序の制御 |
| Ollama | ローカルLLMによる日本語の分析文章生成 |

---

## 2. 現在の主な機能

- Intervals.icu APIからActivitiesをCSV形式で取得
- WellnessをJSON形式で取得
- EventsをJSON形式で取得
- 取得期間を`config.ini`で設定
- SQLiteへ4テーブルで保存
- Activities、Wellness、Sport Info、Eventsをタブ表示
- 直近7日、前の7日、直近28日を集計
- CTL、ATL、TSB、IF、NP、eFTPを分析
- Wellnessの最新値を分析データへ追加
- 今後のEventsを最大10件まで分析データへ追加
- LangGraphのノード単位で進行状況を表示
- Ollamaによる日本語Markdown回答を表示

---

## 3. プロジェクト構成

```text
intervals_ai_app/
├─ app.py
├─ config.ini
├─ requirements.txt
│
├─ data/
│  └─ intervals.db
│
├─ services/
│  ├─ __init__.py
│  ├─ database.py
│  ├─ intervals_client.py
│  └─ ollama_client.py
│
└─ agent/
   ├─ __init__.py
   └─ training_graph.py
```

### ファイルの役割

#### `app.py`

- Streamlit画面の構築
- `config.ini`の読込み
- APIクライアントとSQLiteの初期化
- Activities、Wellness、Eventsの取得処理
- 各データの期間集計
- LangGraphへの初期Stateの受渡し
- 分析結果の画面表示

#### `services/intervals_client.py`

- Intervals.icuへのBasic認証
- Activities CSVの取得
- Wellness JSONの取得
- Events JSONの取得
- HTTPエラーと応答形式の検証

#### `services/database.py`

- SQLiteファイルの初期化
- DataFrameの列名と値の正規化
- `list`、`dict`、`tuple`のJSON文字列化
- Wellness本体と`sportInfo`の分離
- 各テーブルの保存・読込み

#### `services/ollama_client.py`

- Ollamaサーバーへの接続確認
- インストール済みモデルの取得
- 指定モデルの存在確認
- `/api/generate`を使った文章生成
- 接続、タイムアウト、HTTP、JSONエラーの処理

#### `agent/training_graph.py`

- LangGraphのState定義
- データ検証
- Activities、Wellness、Eventsの分析データ作成
- Ollama用プロンプトの作成
- Ollama呼出し
- ノード間の実行順序管理

---

## 4. 設定ファイル

`config.ini`の例です。

```ini
[INTERVALS]
ATHLETE_ID = 0
API_KEY = YOUR_INTERVALS_ICU_API_KEY

[OLLAMA]
BASE_URL = http://localhost:11434
MODEL = llama3.1:latest

[DATABASE]
FILE = data/intervals.db

[SYNC]
DAYS = 90
```

### 設定項目

| セクション | キー | 内容 |
|---|---|---|
| INTERVALS | ATHLETE_ID | Intervals.icuのAthlete ID。個人APIキーでは`0`も使用可能 |
| INTERVALS | API_KEY | Intervals.icuのAPIキー |
| OLLAMA | BASE_URL | Ollama APIのURL |
| OLLAMA | MODEL | 使用するOllamaモデル名 |
| DATABASE | FILE | SQLiteファイルの相対パス |
| SYNC | DAYS | ActivitiesとWellnessの取得日数 |

> `config.ini`にはAPIキーが含まれるため、Gitへ登録しないでください。

---

## 5. データ取得範囲

現在は実行日を起点に、次の範囲を作成します。

```text
Activities : 今日からSYNC.DAYS分さかのぼった期間 ～ 今日
Wellness   : 今日からSYNC.DAYS分さかのぼった期間 ～ 今日
Events     : 今日からSYNC.DAYS分さかのぼった期間 ～ 30日後
```

Eventsだけ未来30日を含めることで、今後の大会や予定ワークアウトを分析対象にします。

---

## 6. SQLite構成

### `activities`

実施済みアクティビティを保存します。

主な分析対象は次のとおりです。

```text
start_date_local
name
type
distance
moving_time
icu_training_load
icu_fitness
icu_fatigue
icu_intensity
icu_normalized_watts
icu_eftp
```

### `wellness`

日別の身体状態とフィットネス情報を保存します。

```text
id
ctl
atl
rampRate
weight
restingHR
hrv
sleepSecs
sleepScore
sleepQuality
steps
vo2max
```

`sportInfo`は親テーブルから取り除き、`wellness_sport_info`へ保存します。

### `wellness_sport_info`

Wellnessの`sportInfo`を正規化した子テーブルです。

```text
wellness_id
sport_type
eftp
data_json
```

複合主キーは次の組合せです。

```text
wellness_id + sport_type
```

元データの例です。

```json
[
  {
    "type": "Ride",
    "eftp": 280
  },
  {
    "type": "Run",
    "eftp": 315
  }
]
```

正規化後のイメージです。

```text
wellness_id  sport_type  eftp
2026-09-18   Ride        280
2026-09-18   Run         315
```

`data_json`には、eFTP以外の種目別情報を失わないように元の辞書をJSON形式で保存します。

### `events`

カレンダーイベントと予定ワークアウトを保存します。

主な利用対象は次のとおりです。

```text
start_date_local
name
category
type
description
```

---

## 7. トレーニング指標

| 一般用語 | Intervals.icu上の意味 | 主なフィールド | 現在の処理 |
|---|---|---|---|
| TSS | 負荷 | `icu_training_load` | 期間合計 |
| CTL | フィットネス | `icu_fitness`または`wellness.ctl` | 期間内の最新有効値 |
| ATL | ファティーグ | `icu_fatigue`または`wellness.atl` | 期間内の最新有効値 |
| TSB | フォーム | 計算値 | `CTL - ATL` |
| IF | 強度 | `icu_intensity` | 期間平均 |
| NP | 正規化パワー | `icu_normalized_watts` | 期間平均 |
| eFTP | 推定FTP | `icu_eftp`または`sportInfo.eftp` | 期間内または種目別の最新値 |

### 単位変換

```text
距離：メートル ÷ 1,000 → キロメートル
時間：秒 ÷ 3,600 → 時間
IF：85のような百分率形式は0.85へ変換
NP・eFTP：W
```

---

## 8. 分析期間

Activitiesデータ内の有効な最終日を分析基準日として使用します。

```text
直近7日  : 基準日 - 6日  ～ 基準日
前の7日  : 基準日 - 13日 ～ 基準日 - 7日
直近28日 : 基準日 - 27日 ～ 基準日
```

### 前の7日との比較

次の指標について変化率を計算します。

```text
距離
時間
負荷
```

計算式は次のとおりです。

```text
変化率 = (直近7日の値 - 前の7日の値) ÷ 前の7日の値 × 100
```

前の7日の値が0の場合は、ゼロ除算を避けて「比較不可」とします。

---

## 9. Wellness分析

現在は、Wellnessの最新日を取得し、次の値を分析テキストへ追加します。

```text
CTL
ATL
TSB
体重
安静時心拍
HRV
睡眠スコア
レディネス
歩数
睡眠時間
```

TSBは次の式で計算します。

```text
TSB = CTL - ATL
```

現在は最新1日の表示が中心です。コーチAI化では、直近7日平均と直近28日平均の比較を追加する予定です。

---

## 10. Events分析

現在は、今日以降のEventsを日付順に並べ、最大10件まで次の形式で分析データへ追加します。

```text
日付 / category / type / name
```

現状は予定一覧の整理までです。今後は次を追加します。

- 目標イベントまでの残日数
- 予定ワークアウトの負荷と時間
- レース、ワークアウト、その他イベントの分類
- 次回メニュー提案時の予定競合判定

---

## 11. LangGraph構成

### State

```python
class TrainingAgentState(TypedDict, total=False):
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
```

### 現在のノード

```text
START
  ↓
validate_data
  ↓
prepare_analysis_data
  ↓
build_prompt
  ↓
call_ollama
  ↓
END
```

#### `validate_data`

- Activities DataFrameの有無を確認
- Activitiesが空なら分析を停止
- Activities、Wellness、Eventsの件数を記録

#### `prepare_analysis_data`

- Activitiesの期間別分析テキストを作成
- Wellnessの最新状態を作成
- 今後のEvents一覧を作成
- 3種類のテキストを`analysis_data`へ統合

#### `build_prompt`

- データの意味をOllamaへ説明
- 出力章立てを指定
- 数値の捏造や医学的断定を禁止

#### `call_ollama`

- Ollamaへの接続確認
- 指定モデルの存在確認
- プロンプト送信
- 最終回答をStateへ格納

---

## 12. Streamlit画面

### サイドバー

- Athlete ID
- Ollamaモデル
- SQLiteファイル名
- APIキー設定状況
- Ollama接続確認
- Intervals.icuデータ取得ボタン

### データ概要

- Activities件数
- 合計距離
- 合計時間
- 合計負荷

### SQLite保存データ

次の4タブを表示します。

```text
Activities
Wellness
Wellness Sport Info
Events
```

### AI分析

- 利用者の追加依頼を入力
- LangGraphのノードごとの進捗を表示
- 最終回答をMarkdown形式で表示

---

## 13. 起動方法

### 前提条件

- Python
- Ollama
- 使用モデルのインストール
- Intervals.icu APIキー

### 仮想環境の有効化

```powershell
cd C:\Users\S.OTA\Desktop\python\intervals_ai_app
.venv\Scripts\Activate.ps1
```

### Streamlitの起動

```powershell
python -m streamlit run app.py
```

### 構文確認

```powershell
python -m py_compile app.py
python -m py_compile services\database.py
python -m py_compile services\intervals_client.py
python -m py_compile services\ollama_client.py
python -m py_compile agent\training_graph.py
```

### Ollama確認

```powershell
ollama list
ollama run llama3.1:latest
```

`MODEL`に指定した名前は、`ollama list`の表示と一致させてください。

---

## 14. 既知の課題

### 14.1 Eventsの名前取得処理

`build_events_analysis_data()`のイベント名取得は、次の形式である必要があります。

```python
event_name = str(
    row.get(
        "name",
        "名称なし",
    )
)
```

`row.get("name""名称なし")`のようにカンマがない場合、Pythonでは文字列が連結され、`name名称なし`という存在しない列名を検索してしまいます。

### 14.2 `wellness_sport_info`がLangGraphへ未接続

現在、`wellness_sport_info`はSQLiteへ保存され、Streamlitにも表示されていますが、LangGraphのStateには渡していません。

そのため、種目別eFTPは現在のAI回答へ直接反映されません。

今後は次を追加します。

```text
wellness_sport_info_df
sport_info_analysis
sport_info_builder
```

### 14.3 Wellnessは最新値中心

現在は最新日の値を説明していますが、個人内の通常範囲との比較がありません。

追加候補は次のとおりです。

- 直近7日平均
- 直近28日平均
- 最新値と28日平均の差
- 欠損日数
- HRV、安静時心拍、睡眠の推移

### 14.4 CTL・ATLの基準日

ActivitiesのCTL・ATLとWellnessのCTL・ATLで、最新有効値の日付が異なる可能性があります。

回答では値だけでなく、次の形式で基準日を明記する必要があります。

```text
2026-09-18時点
```

### 14.5 IF・NPは単純平均

現在は期間内の単純平均です。活動時間の差を反映するには時間加重平均を検討します。

### 14.6 SQLite更新方式

現在、Activities、Wellness、Eventsは基本的に全置換保存です。

今後は主キーを明示し、次の方式へ移行します。

```text
新規ID  → INSERT
既存ID  → UPDATE
対象外  → 必要に応じて維持または削除
```

### 14.7 Ollamaの多言語混在

モデルによっては回答中に韓国語などが混在する場合があります。

対応候補は次のとおりです。

- system promptで日本語のみを指定
- 日付表現を`YYYY-MM-DD時点`に固定
- temperatureを低くする
- 出力後にハングルを検出
- 必要に応じて一度だけ再生成

---

## 15. コーチAI化ロードマップ

### STEP 1：Events解析の強化

- レースや目標イベントまでの残日数
- 予定ワークアウトの時間、負荷、構成
- 直近イベントと次回メニューの競合確認

### STEP 2：Wellness推移分析

- HRVの7日平均と28日平均
- 安静時心拍の7日平均と28日平均
- 睡眠時間と睡眠スコアの推移
- 最新値と個人内基準との差

### STEP 3：種目別eFTPの統合

`wellness_sport_info`をLangGraphのStateへ追加します。

```text
activities_df
wellness_df
wellness_sport_info_df
events_df
```

### STEP 4：Coachノードの追加

現在のグラフへ`prepare_coach_data`を追加します。

```text
validate_data
  ↓
prepare_analysis_data
  ↓
prepare_coach_data
  ↓
build_prompt
  ↓
call_ollama
```

Coachノードでは、確定的なルールと数値計算をPython側で処理します。

```text
直近負荷
CTL / ATL / TSB
Wellness推移
種目別eFTP
今後の予定
利用可能時間
```

### STEP 5：具体的な次回メニュー生成

出力例です。

```text
目的：回復促進
種目：Ride
時間：60分
強度：Z1～Z2
想定負荷：35 TSS

構成：
- ウォームアップ 10分
- Z2 40分
- クールダウン 10分

実施条件：
睡眠不足や安静時心拍の上昇が続く場合は30分へ短縮する。
```

### STEP 6：Goalsテーブルの追加

将来的には目標管理を追加します。

```text
goals
├─ id
├─ name
├─ event_date
├─ sport_type
├─ target_time
├─ target_weight
├─ target_ftp
└─ priority
```

これにより、現在状態だけでなく、目標までの残期間を踏まえた提案が可能になります。

---

## 16. 将来構成

```text
Intervals.icu
    │
    ├─ Activities
    ├─ Wellness
    └─ Events
    │
    ▼
SQLite
    ├─ activities
    ├─ wellness
    ├─ wellness_sport_info
    ├─ events
    ├─ goals
    └─ analysis_history
    │
    ▼
LangGraph
    ├─ Activity Analysis
    ├─ Wellness Analysis
    ├─ Event Analysis
    ├─ Goal Analysis
    ├─ Coach Rules
    └─ Answer Generation
    │
    ▼
LLM Provider
    ├─ Ollama
    └─ 将来の外部AI
    │
    ▼
Streamlit / モバイル / チャット連携
```

---

## 17. 設計方針

本システムでは、次の責任分離を維持します。

```text
Python
  正確な取得、変換、集計、判定

SQLite
  履歴と構造化データの保存

LangGraph
  状態、処理順、条件分岐の管理

Ollama
  計算済みデータの説明と自然言語化

Streamlit
  操作、可視化、進行状況、結果表示
```

LLMへ自由な計算や自由な更新SQLを任せず、数値と判断根拠をPython側で確定してから文章化させる方針です。

---

## 18. 現在の到達状況

| 領域 | 状況 |
|---|---|
| Activities取得 | 実装済み |
| Wellness取得 | 実装済み |
| Events取得 | 実装済み |
| SQLite保存 | 実装済み |
| sportInfo正規化 | 実装済み |
| Streamlit表示 | 実装済み |
| 期間別Activities分析 | 実装済み |
| 最新Wellness分析 | 実装済み |
| 今後のEvents一覧 | 実装済み |
| LangGraph基本フロー | 実装済み |
| 種目別eFTPのAI統合 | 未実装 |
| Wellness推移比較 | 未実装 |
| Coachノード | 未実装 |
| 具体的メニュー生成 | 未実装 |
| Goals管理 | 未実装 |

---

## 19. 最終目標

現在の「何を実施し、どのような傾向にあるかを説明する分析AI」から、次の流れを一貫して扱うコーチAIへ発展させます。

```text
何を実施したか
    ↓
現在どのような状態か
    ↓
今後どのような予定があるか
    ↓
目標までに何が必要か
    ↓
次に何を実施するか
```

最終的には、データに裏付けられた具体的なトレーニングメニューを提示し、実施後の結果を再びIntervals.icuデータから評価する循環型のコーチAIを目指します。
