import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

# 修正
def calculate_loss(flow_dict: dict, target: torch.Tensor) -> torch.Tensor:
    total_loss = 0.0
    weight=[0.3, 0.1, 0.02, 0.01]

    for i,key in enumerate(flow_dict):
        flow_output = flow_dict[key]
        upsampled_output = F.interpolate(flow_output, size=target.size()[2:], mode='bilinear', align_corners=False)
        loss = compute_epe_error(upsampled_output, target)
        total_loss += loss*weight[i]

    total_loss /= len(flow_dict)
    
    return total_loss


def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate_fn = train_collate
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''

    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )

    train_set = loader.get_train_dataset()
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

# 修正
    # full_dataset = loader.get_train_dataset()
    # train_size = int(0.8 * len(full_dataset))  # 80%をトレーニングに使用
    # val_size = len(full_dataset) - train_size  # 残りの20%を検証に使用

    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # train_data = DataLoader(train_dataset,
    #                         batch_size=args.data_loader.train.batch_size,
    #                         shuffle=args.data_loader.train.shuffle,
    #                         collate_fn=collate_fn,
    #                         drop_last=False)

    # val_data = DataLoader(val_dataset,
    #                       batch_size=args.data_loader.train.batch_size,  # 検証用のバッチサイズを指定
    #                       shuffle=False,  # 検証データはシャッフルしない
    #                       collate_fn=collate_fn,
    #                       drop_last=False)

    test_set = loader.get_test_dataset()
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    model = EVFlowNet(args.train).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    
    model.train()
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            flow,flow_dict = model(event_image) # [B, 2, 480, 640]
            loss: torch.Tensor = calculate_loss(flow_dict, ground_truth_flow)
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

# 修正
        # model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch in val_data:
        #         event_image = batch["event_volume"].to(device)
        #         ground_truth_flow = batch["flow_gt"].to(device)
        #         flow = model(event_image)
        #         loss = compute_epe_error(flow, ground_truth_flow)
        #         total_val_loss += loss.item()
        
        # avg_val_loss = total_val_loss / len(val_data)
        # print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            flow_,batch_flow = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, flow_), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "/content/drive/MyDrive/DL最終課題event/submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
