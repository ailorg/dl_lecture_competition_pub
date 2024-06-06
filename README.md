# DL基礎講座2024　最終課題「Visual Question Answering（VQA）」


## 環境構築
### Conda
```bash
conda create -n dl_competition python=3.10
pip install -r requirements.txt
```
### Docker
- Dockerイメージのcudaバージョンについては，ご自身が利用するGPUに合わせて変更してください．
```bash
docker build -t <イメージ名> .
docker run -it -v $PWD:/workspace -w /workspace <イメージ名> bash
```

## ベースラインモデルを動かす
### データのダウンロード
- [こちら](https://drive.google.com/drive/folders/1QTcWMATZ_iGsHnxq6-3aXa7D5VZAzs5T?usp=sharing)から各データをダウンロードしてください．
  - train.json: 訓練データのjsonファイル．画像のパス，質問文，回答文がjson形式でまとめられている．
  - valid.json: テストデータのjsonファイル．画像のパス，質問文がjson形式でまとめられている．
  - train.zip: 訓練データの画像ファイル．
  - valid.zip: テストデータの画像ファイル．
- ダウンロード後，train.zipとvalid.zipを解凍し，各データをdataディレクトリ下に置いてください．
### 訓練・提出ファイル作成
```bash
python3 main.py
```
- `main.py`と同様のディレクトリ内に，学習したモデルのパラメータ`model.pth`とテストデータに対する予測結果`submission.npy`ファイルが作成されます．
- ベースラインは非常に単純な手法のため，改善の余地が多くあります．VQAでは**Omnicampusにおいてベースラインのtest accuracy=41%を超えた提出のみ，修了要件として認めることとします．**

## VizWiz(2023 edition) dataset [[link](https://www.kaggle.com/datasets/nqa112/vizwiz-2023-edition)] の詳細
- 24842枚の画像があり，訓練データには各画像に対して1つの質問と10人の回答者による回答．
  - 10人の回答はすべて同じであるとは限らない．
- 24842のデータの内，80%(19873)が訓練データ，20%(4969)がテストデータとして与えられる．
  - テストデータに対する回答は正解ラベルとし，訓練時には与えられない．
  - データ提供元ではtrainとvalに分かれているが，データの配布前に運営の方でtrainとvalをランダムに分け直している．

## タスクの詳細
- 本コンペでは，与えられた画像と質問に対し，適切な回答をモデルに出力させる．
- 評価は[VQA](https://visualqa.org/index.html)(Visual Question Answering)に基づき，以下の式で計算される．
$$\text{Acc}(ans) = min(\frac{humans \ that \ said \ ans}{3}, 1)$$
- 1つのデータに対し，10人の回答の内9人の回答を選択し上記の式で性能を評価した，10パターンのAccの平均をそのデータに対するAccとする．
- 予測結果と正解ラベルを比較する前に，回答をlowercaseにする，冠詞は削除するなどの前処理を行う（[詳細](https://visualqa.org/evaluation.html)）．

## 考えられる工夫の例
- 質問文の前処理．
  - 回答については評価指標に合わせるため前処理をしていますが，質問文には前処理を施していません．そのため同じ単語であっても大文字，小文字で異なる単語として扱われます．大文字・小文字の統一，冠詞の削除など適切な処理をすることで，性能向上が見込めます．
- 質問文の表現．
  - ベースラインでは，質問文をモデルに入力する際にはone-hotベクトルにしています．この表現をtokenizer等を利用して分散表現にすることで，モデルが学習しやすい表現となり，性能向上が見込めます．
- 回答の出力候補の変更．
  - ベースラインは非常に簡素なモデルとなっており，出力は訓練データに存在する回答例の中からどれが適切かを選択する，クラス分類の形式になっています．これでは訓練データには存在しない回答は出力不可能なため，より大きなコーパスを用いると良いでしょう（huggingfaceでは，VizWizに利用できる[class_mapping](https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv)が存在します）．
- 画像の前処理．
  - 画像の前処理には形状を同じにするためのResizeのみを利用しています．第5回の演習で紹介したようなデータ拡張を追加することで，疑似的にデータを増やし汎化性能の向上が見込めます．
