# SO-101による小物移動の模倣学習

SO-101の実演を2台のカメラと関節状態で記録し、ACTを学習するためのコードです。Windows PCで実演収集と実機制御を行い、ThinkStation PGXで学習と遠隔GPU推論を行いました。接続先と使用機器はコマンド引数または環境変数で指定します。

## 収集・学習・推論の構成

| 処理 | 実装 |
| --- | --- |
| 実演収集画面 | `record-so101.py`、`so101_record_ui.py` |
| LeaderからFollowerへの追従 | `follow-so101.py` |
| 俯瞰・手元カメラの接続と画像整形 | `iphone_camera_profile.py`、`windows_camera_devices.py` |
| 別プロセスでのデータ保存 | `training/episode_spool.py`、`training/process_export.py` |
| 保存済み試行の確認・除外 | `so101_episode_review_ui.py`、`training/episode_review_store.py` |
| データ結合・ACT学習・追加学習 | `training/pipeline.py`とLeRobot公式CLI |
| 保存済み実観測によるCPU推論 | `training/infer_act.py` |
| 遠隔GPU推論と実機制御 | `training/run_pgx_inference_client.py`、`training/so101_remote_adapter.py` |
| 推論中の映像・観測・行動の保存 | `training/inference_session_recorder.py` |
| 実画像を使った拡張の比較 | `training/prepare_augmentation.py` |
| カメラ別の画像拡張と追加学習 | `training/camera_augmentation/camera_specific.py`、`train_camera_specific.py` |
| 拡張実験の実行・評価・比較 | `training/camera_augmentation/camera_suite.py`、`evaluate_camera_model.py`、`compare_camera_evaluations.py` |

俯瞰画像はCamo経由で1280×720画素を取得します。縦横比を保って640×360画素に縮小し、上下に60画素ずつ黒い余白を加えます。手元画像は640×480画素です。どちらもRGBとして学習・推論へ渡します。

保存処理は追従制御から分離しています。推論では画像送信を別スレッドで実行し、通信待ちが制御処理を止めない構成です。カメラや通信、行動送信で検出したエラーに対して停止と終了処理を行います。

## 学習設定

初期実験は66エピソード、62,819フレームの実演を使用しました。ACTの画像入力は2視点、関節状態と出力行動は各6次元です。

| 設定 | 値 |
| --- | --- |
| LeRobot | v0.6.1と同梱のデータ読み込み修正 |
| Policy | ACT、ResNet-18 |
| 初期学習 | 100,000ステップ |
| 追加学習 | 公式resumeで累計200,000ステップ |
| Batch size | 8 |
| DataLoader workers | 4 |
| 学習率 | 0.00001 |
| Action chunk | 100 |
| 初期実験の画像拡張 | 無効 |

基本学習の設定は`configs/training_initial.json`、追加学習で使うカメラ別の画像拡張は`configs/camera_augmentation.json`で確認できます。

## 環境の準備

Python 3.12、Git、uv、FFmpegを用意してください。Windowsの収集画面にはTkinterが必要です。俯瞰カメラとしてiPhoneを使う場合は、Camoで映像を確認してから収集を始めてください。

以下はリポジトリのルートで実行します。LeRobotの公式ソースを取得し、初期実験で使用した読み込み修正を適用します。この修正は、画像を不要に復号せず、行動などの必要な列だけを読み込むものです。

```sh
uv venv --python 3.12
git clone --branch v0.6.1 --depth 1 https://github.com/huggingface/lerobot.git vendor/lerobot
git -C vendor/lerobot apply --check ../../patches/lerobot-v0.6.1-dataset-reader.patch
git -C vendor/lerobot apply ../../patches/lerobot-v0.6.1-dataset-reader.patch
uv pip install -e "vendor/lerobot[feetech,async,dataset]" Pillow
```

GPU用PyTorchは、利用するGPU・OS・CUDAに合うものを使ってください。PGXとWindowsでは同じwheelを使えるとは限りません。ここでは機材に依存するPyTorchの配布先を固定していません。通常の収集画面はTkinterで表示します。旧コンソールのOpenCVプレビューを使う場合は、GUI機能を含むOpenCV環境が別途必要です。

## Windowsでの実演収集

使用するLeaderとFollowerを識別し、既存の校正ファイルを次の場所に配置してください。別の個体の校正値は使用しないでください。校正手順は[LeRobot公式のSO-101説明](https://huggingface.co/docs/lerobot/so101)を参照してください。

```text
calibration/leader/so101_leader.json
calibration/follower/so101_follower.json
```

PowerShellでポートとタスクの説明を入力します。接続先を省略して自動識別する場合は、環境変数`SO101_LEADER_SERIAL`と`SO101_FOLLOWER_SERIAL`に自分の機材のUSBシリアル番号を設定してください。設定値をリポジトリへ保存する必要はありません。

```powershell
$env:SO101_LEADER_PORT = Read-Host "LeaderのCOMポート"
$env:SO101_FOLLOWER_PORT = Read-Host "FollowerのCOMポート"
$env:SO101_TASK = Read-Host "記録するタスクの説明"
uv run --no-project --no-sync python record-so101.py --camera-profile iphone --leader-port $env:SO101_LEADER_PORT --follower-port $env:SO101_FOLLOWER_PORT --task $env:SO101_TASK
```

このコマンドは計画を表示します。カメラの役割、ポート、校正値を確認した後、同じコマンドの末尾へ`--execute`を加えると、実際にカメラとアームへ接続します。画面でRを押すと録画を始め、Pで一時停止・再開、Sでその試行を保存します。Fでデータセットを閉じ、Qで追従を停止して終了処理を行います。保存エラーや除外待ちの試行は、確認画面で処理してから学習対象にしてください。

## データ結合とACT学習

収集したデータをPGXへコピーしてから、同じPython環境で実行します。下の`datasets/first`と`datasets/second`には、自分の実データを配置してください。`repo-id`はローカルデータの識別子であり、Hubへの公開先ではありません。

```sh
uv run --no-project --no-sync python training/pipeline.py merge --roots datasets/first datasets/second --repo-ids local/so101-demonstrations local/so101-demonstrations --output datasets/merged
uv run --no-project --no-sync python training/pipeline.py train --dataset datasets/merged --output outputs/act --device cuda
uv run --no-project --no-sync python training/pipeline.py resume --train-config outputs/act/checkpoints/100000/pretrained_model/train_config.json --output outputs/act-resumed --steps 200000
```

既存の出力先は上書きしません。追加学習には、重みだけでなく最初の学習で保存したチェックポイント一式が必要です。モデルやデータをHubへアップロードする設定は無効にしています。

色調変更と切り抜きの初期比較は、実データに対して次のコマンドで実行できます。元データは変更せず、指定した新しい出力先に画像と設定を保存します。このコマンド自体は学習を開始しません。

```sh
uv run --no-project --no-sync python training/prepare_augmentation.py --dataset datasets/merged --output outputs/augmentation-preview
```

## カメラ別画像拡張による追加学習

基本学習後に実機で動かしたところ、マスコットを安定してつかめませんでした。動作の安定性を改善するため、カメラの役割に合わせた画像拡張を導入し、追加学習しました。

追加実験では、200,000ステップ時点の重みを初期値として、画像拡張を加えて100,000ステップ学習しました。オプティマイザーは新しく作り、`resume=false`で実行しています。前節の公式resumeとは異なる実験です。

俯瞰画像には切り抜きと平行移動を適用します。把持対象の位置関係を保つため、手元画像には幾何変換を加えません。画像だけを変換し、関節状態・行動ラベル・元のデータセットは変更しない構成です。

| 変換 | 俯瞰カメラ | 手元カメラ |
| --- | --- | --- |
| 切り抜き | 確率0.5、面積比0.9〜1.0、縦横比を固定 | なし |
| 平行移動 | 確率0.5、幅・高さの約3%を端画素で埋めて切り出し | なし |
| 色調 | 確率0.8、明度・コントラスト0.8〜1.2 | 同じ範囲 |
| ぼかし | 確率0.15、σ=0.1〜0.8 | 確率0.1、σ=0.1〜0.5 |
| 陰影 | 確率0.2、不透明度0.05〜0.2 | 同じ範囲 |
| ガウスノイズ | なし | 確率0.2、標準偏差0.005〜0.015 |

値の範囲が0〜1の画像に変換を適用し、元の型へ戻します。1サンプルの時間方向には同じ変換パラメータを使い、カメラ間とサンプル間では独立に抽選します。LeRobot標準の共通画像変換は無効にし、この処理との二重適用を防いでいます。

次のコマンドは、元の200,000ステップモデルを評価した後、拡張ありの追加学習と評価を実行します。`configs/augmentation_plan.json`には実施済みのcombined条件、100,000ステップ、seed=1000、同時実行数1を記載しています。モデルと実データのパスは自分の保存先に合わせてください。

```sh
uv run --no-project --no-sync python training/camera_augmentation/camera_suite.py --root outputs/camera-augmentation --plan configs/augmentation_plan.json --source outputs/act-resumed/checkpoints/200000/pretrained_model --dataset datasets/merged
uv run --no-project --no-sync python training/camera_augmentation/compare_camera_evaluations.py --root outputs/camera-augmentation --output outputs/camera-augmentation/comparison.csv
```

単一条件を実行する場合は`camera_job.py`、経過を読み取る場合は`camera_progress.py`を使用できます。`camera_benchmark.py`は実学習を1・2・3プロセスで各120ステップ実行する処理速度の計測用です。これらも実データとモデルが必要で、模擬データは生成しません。

保存された評価では、同じ48フレームとseed=1000を使いました。全10条件で変換後画像のSHAが一致しており、同じ入力に対してモデルを比較しています。全条件の集計値は[評価結果JSON](results/evaluation_summary.json)に収録しました。以下は正規化した6関節行動のMAEです。

| 評価条件 | 元の200,000ステップモデル | 拡張あり追加学習後 |
| --- | --- | --- |
| 元画像 | 0.047489 | 0.040668 |
| 俯瞰画像の約3%平行移動 | 0.127327 | 0.041267 |
| 俯瞰画像の切り抜き | 0.071039 | 0.040796 |

各条件では、パディングを除く4,462時点の行動を評価しました。評価画像は学習済みの66エピソードから抽出しています。この結果は学習データ内の画像変化への誤差であり、未見データの性能や実機成功率ではありません。追加100,000ステップを画像拡張なしで学習した対照群がないため、拡張だけの効果は分離できません。

実機ではマスコットをつかみ、目標の人形付近へ運びました。一度でつかめず、再度つかみ直して移動を完了した試行もあります。今後は開始位置と向きを変えた実演を追加し、動作の安定性を改善します。

## 遠隔GPU推論

PGXでLeRobot公式の推論サーバーを起動します。

```sh
uv run --no-project --no-sync python -m lerobot.async_inference.policy_server --host=localhost --port=8080
```

Windowsの別のPowerShellでSSHトンネルを接続します。アカウントや接続先は入力欄で指定し、ソースコードには書き込みません。

```powershell
$env:SO101_SSH_TARGET = Read-Host "SSH接続先をアカウントとホスト名で入力"
ssh -N -L 8080:localhost:8080 $env:SO101_SSH_TARGET
```

推論クライアント用のPowerShellで、PGX上のモデルとカメラ番号を指定します。`--checkpoint-dir`はWindows側の排他制御用フォルダーです。`--skip-checkpoint-verification`を指定すると、クライアント側の事前検証ログを要求しません。重みはPGXのサーバーが実際に読み込みます。

```powershell
$env:SO101_MODEL = Read-Host "PGX上のpretrained_modelディレクトリ"
$env:SO101_FOLLOWER_PORT = Read-Host "FollowerのCOMポート"
$env:SO101_OVERHEAD_CAMERA = Read-Host "CamoのMSMFカメラ番号"
$env:SO101_WRIST_CAMERA = Read-Host "手元カメラのDirectShow番号"
$env:SO101_TASK = Read-Host "収集時と同じタスク説明"
uv run --no-project --no-sync python training/run_pgx_inference_client.py --server-address localhost:8080 --pretrained-path $env:SO101_MODEL --checkpoint-dir outputs/inference-lease --checkpoint-step 200000 --skip-checkpoint-verification --follower-port $env:SO101_FOLLOWER_PORT --overhead-camera-index $env:SO101_OVERHEAD_CAMERA --arm-camera-index $env:SO101_WRIST_CAMERA --task $env:SO101_TASK --recording-directory outputs/inference-recordings
```

このコマンドは計画を表示します。末尾へ`--run`を加えると、実際にFollowerを駆動します。Qで停止し、保存終了を待ってからウィンドウを閉じてください。2本のAVI、フレーム時刻、送信観測、受信した行動列、実際に送った行動を記録します。録画は既に接続したカメラの画像を使い、追加のカメラ接続を行いません。
