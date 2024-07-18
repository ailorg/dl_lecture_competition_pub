# DL基礎講座2024　最終課題

## 概要
### 最終課題内容：3つのタスクから1つ選び，高い性能となるモデルを開発してください（コンペティション形式）
3つのタスクはそれぞれ以下の通りです．必ず**1つ**を選んで提出してください．
- 脳波分類（[`MEG-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/MEG-competition)ブランチ）: 被験者が画像を見ているときの脳波から，その画像がどのクラスに属するかを分類する．
  - サンプル数: 訓練65,728サンプル，検証16,432サンプル，テスト16,432サンプル．
  - 入力: 脳波データ．
  - 出力: 1854クラスのラベル．
  - 評価指標: top-10 accuracy（モデルの予測確率トップ10に正解クラスが含まれているかどうか）．
- Visual Question Answering（VQA）（[`VQA-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/VQA-competition)ブランチ）: 画像と質問から，回答を予測する．
  - サンプル数: 訓練19,873サンプル，テスト4,969サンプル．
  - 入力: 画像データ（RGB，サイズは画像によって異なる），質問文（サンプルごとに長さは異なる）．
  - 出力: 回答文（サンプルごとに長さは異なる）．
  - 評価指標: VQAでの評価指標（[こちら](https://visualqa.org/evaluation.html)を参照）を利用．
- Optical Flow Prediction from Event Camera (EventCamera)（[`event-camera-competition`](https://github.com/ailorg/dl_lecture_competition_pub/tree/event-camera-competition)ブランチ）: イベントカメラのデータから，Optical Flowを予測する．
  - サンプル数: 訓練7,800サンプル，テスト2,100サンプル．
  - 入力: イベントデータ（各時刻，どのピクセルで"log intensity"に変化があったかを記録）．
  - 出力: Optical Flow（連続フレーム間で，各ピクセルの動きをベクトルで表したもの）．
  - 評価指標: Average Endpoint Error（推定したOptical Flowと正解のOptical Flowのユークリッド距離）．

### 最終課題締切：7/18（木）16:00

### 注意点
- 学習するモデルについて制限はありませんが，必ず**提供された訓練データで学習したモデルで予測をしてください**．
  - 訓練データを使って，事前学習済みモデルをfine-tuningしても構いません．
  - 埋め込み抽出モデルなど，モデルの一部を訓練しないケースは構いません．
  - 学習を一切せずにChatGPTなどの基盤モデルを使うのは禁止です．

## 最終課題の取り組み方
### ベースラインコードのダウンロードと実行
ベースラインコードのダウンロードから，提出までの流れの説明です．
1. ベースラインコードがあるリポジトリをforkする．
- [こちら](https://github.com/ailorg/dl_lecture_competition_pub/tree/main)のリポジトリをforkします．
- この際，"Copy the `main` branch only"の**チェックを外してください．**  
2. `git clone`を利用してローカル環境にリポジトリをcloneする．
- 以下のコマンドを利用して，手順1でforkしたリポジトリをローカル環境にcloneします．
- [Github user name]にはご自身のGithubのユーザー名を入力してください．
```bash
$ git clone git@github.com:[Github user name]/dl_lecture_competition_pub
```
3. `git checkout`を利用して自身が参加するコンペティションのブランチに切り替える．
- 以下のコマンドを利用して，参加したいコンペティションのブランチに切り替えてください．
- [Competition name]には`MEG-competition`（脳波分類タスク），`VQA-competition`（VQAタスク），`event-camera-competition`（EventCameraタスク）のいずれかが入ります．
```bash
$ cd dl_lecture_competition_pub
$ git checkout [Competition name]
```
4. README.mdの`環境構築`を参考に環境を作成します．
- README.mdにはconda，もしくはDockerを利用した環境構築の手順を記載しています．
5. README.mdの`ベースラインモデルを動かす`を参考に，ベースラインコードを実行すると，学習や予測が実行され，テストデータに対する予測である`submission.npy`が出力されます．

### 取り組み方
- ベースラインコードを書き換える形で，より性能の高いモデルの作成を目指してください．
  -  基本的には`main.py`などを書き換える形で実装してください．
  -  自分で1から実装しても構いませんが，ベースラインコードと同じ訓練データおよびテストデータを利用し，同じ形式で予測結果を出力してください．
- コンペティションでは，受講生の皆様に`main.py`の中身を書き換えて，より性能の高いモデルを作成していただき，予測結果(`submission.npy`)，工夫点レポート(`.pdf`)，実装したコードのリポジトリのURLを提出いただきます．
- 以下の条件を全て満たした場合に，最終課題提出と認めます．
   - 全ての提出物が提出されていること．
     - 注意：Omicampusで提出した結果が，レポートで書いた内容やコードと大きく異なる場合は，提出と認めない場合があります．
   - Omnicampusでの採点で各タスクのベースライン実装の性能を超えていること．
     - ベースライン性能は各タスクのブランチのREADMEを確認して下さい． 

### Githubへのpush方法
最終課題ではforkしたリポジトリをpublicに設定していただき，皆様のコードを評価に利用いたします．そのため，作成したコードをgithubへpushしていただく必要があります．

以下にgithubにその変更を反映するためのpushの方法を記載します．
1. `git add`
- 以下のように，`git add`に続けて変更を加えたファイル名を空白を入れて羅列します．
```bash
$ git add main.py hogehoge.txt hugahuga.txt
```
2. `git commit`
- `-m`オプションによりメモを残すことができます．その時の変更によって何をしたか，この後何をするのかなど記録しておくと便利です．
```bash
$ git commit -m "hogehoge"
``` 
3. `git push`
- branch nameには提出方法の手順3でcheckoutの後ろで指定したブランチ名を入力します．
```bash
$ git push origin [branch name]
```

## 最終課題の提出方法
Omnicampusから予測結果(`.npy`)，工夫点レポート(`.pdf`)，実装したコードのリポジトリのURLを提出していただきます．

### 予測結果の提出
1. Omnicampusの「宿題」一覧から「最終課題（---）」をクリックします（---を取り組むタスクに読み替えてください）．
2. 「拡張子（.npy）のファイルを提出」をクリックし，予測結果`.npy`をアップロードします．採点は即時実行され，「リーダーボードへ」をクリックすると，順位表を確認できます．
<img width="1540" alt="submission" src="https://github.com/ailorg/dl_lecture_competition/assets/11865486/ce9eb6b1-5ecb-4f63-88fc-55c46084bd25">

### 工夫点レポートとGithubリポジトリURLの提出
1. Omnicampusの「レポート課題」一覧から「最終課題レポート」をクリックします．
2. 「レポートのタイトル」にレポートの題名（任意のタイトルで構いません），「参照URL」に今回のタスクの実装を含んだGithubリポジトリのURLを入力して，「提出ファイル」で提出ファイル`.pdf`を選択し，「提出する」をクリックします．
<img width="912" alt="report" src="https://github.com/ailorg/dl_lecture_competition/assets/11865486/f0adbb35-5268-4bf2-9e54-3e728cc7f195">

注意：
- 締め切りまでに**何回提出しても構いません**．最後に提出された内容で評価します．

## 課題内容に関する質問について
- 最終課題の内容に関する質問は以下で受け付けています．
  - slack「08_受講生間の疑問解決_最終課題」チャネル
- 基本的には受講生間で解決するようにしてください．**必ずしも全ての質問に対してTAや運営から回答する訳ではないので注意してください**．
