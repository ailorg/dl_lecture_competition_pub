# DL基礎講座2024　最終課題「脳波分類」

## 環境構築

```bash
conda create -n dlbasics python=3.10
conda activate dlbasics
pip install -r requirements.txt
```

## ベースラインモデルを動かす

### 訓練

```bash
python main.py

# オンラインで結果の可視化（wandbのアカウントが必要）
python main.py use_wandb=True
```

- `outputs/{実行日時}/`に重み`model_best.pt`と`model_last.pt`，テスト入力に対する予測`submission.npy`が保存されます．`submission.npy`をOmnicampusに提出することで，test top-10 accuracyが確認できます．

  - `model_best.pt`はvalidation top-10 accuracyで評価

- 訓練時に読み込む`config.yaml`ファイルは`train.py`，`run()`の`@hydra.main`デコレータで指定しています．新しいyamlファイルを作った際は書き換えてください．

- ベースラインは非常に単純な手法のため，改善の余地が多くあります（セクション「考えられる工夫の例」を参考）．そのため，**Omnicampusにおいてベースラインのtest accuracy=1.637%を超えた提出のみ，修了要件として認めることとします．**

### 評価のみ実行

- テストデータに対する評価のみあとで実行する場合．出力される`submission.npy`は訓練で最後に出力されるものと同じです．

```bash
python eval.py model_path={評価したい重みのパス}.pt
```

## データセット[[link](https://openneuro.org/datasets/ds004212/versions/2.0.0)]の詳細

- 1,854クラス，22,448枚の画像（1クラスあたり12枚程度）
  - クラスの例: airplane, aligator, apple, ...

- 各クラスについて，画像を約6:2:2の割合で訓練，検証，テストに分割

- 4人の被験者が存在し，どの被験者からのサンプルかは訓練に利用可能な情報として与えられる (`*_subject_idxs.pt`)．

### データセットのダウンロード

- [こちら](https://drive.google.com/drive/folders/1pgfVamCtmorUJTQejJpF8GhvwXa67rB9?usp=sharing)から`data.zip`をダウンロードし，`data/`ディレクトリに展開してください．

- 画像を事前学習などに用いる場合は，ドライブから`images.zip`をダウンロードし，任意のディレクトリで展開します．{train, val}_image_paths.txtのパスを使用し，自身でデータローダーなどを作成してください．

## タスクの詳細

- 本コンペでは，**被験者が画像を見ているときの脳波から，その画像がどのクラスに属するか**を分類します．

- 評価はtop-10 accuracyで行います．
  - モデルの予測確率トップ10に正解クラスが含まれているかどうか
  - つまりchance levelは10 / 1,854 ≒ 0.54%となります．

## 考えられる工夫の例

- 脳波の前処理
  - 配布したデータに対しては前処理が加えられていません．リサンプリングやフィルタリング，スケーリング，ベースライン補正など，波に対する基本的な前処理を試すことで性能の向上が見込まれます．
- 画像データを用いた事前学習
  - 本コンペのタスクは脳波のクラス分類ですが，配布してある画像データを脳波エンコーダの事前学習に用いることを許可します．
  - 例）CLIP [Radford+ 2021]
- 音声モデルの導入
  - 脳波と同じ波である音声を扱うアーキテクチャを用いることが有効であると知られています．
- 過学習を防ぐ正則化やドロップアウト
- 被験者情報の利用
  - 被験者ごとに脳波の特性が異なる可能性があるため，被験者情報を利用することで性能向上が見込まれます．
  - 例）Subject-specific layer [[Defossez+ 2022](https://arxiv.org/pdf/2208.12266)], domain adaptation
