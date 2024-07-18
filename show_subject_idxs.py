import numpy as np

def display_npy_file(file_path):
    # .npyファイルを読み込む
    data = np.load(file_path)

    # データの内容を表示する
    print("Data from npy file:")
    print(data)

    # データの形状を表示する
    print("\nShape of data:")
    print(data.shape)

if __name__ == "__main__":
    # 表示する.npyファイルのパスを指定する
    file_path = "data/test_subject_idxs/00001.npy"

    display_npy_file(file_path)
