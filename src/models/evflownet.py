import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class EVFlowNet(nn.Module):
    def __init__(self, args, no_batch_norm=False):
        super(EVFlowNet,self).__init__()
        self._args = args
        self.no_batch_norm = no_batch_norm

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        # ResNetブロックの数を増やす
        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(4)])

        # Attention機構を追加
        self.attention = SelfAttention(8*_BASE_CHANNELS)

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # イベントボリュームを取得
        event_volume = inputs["event_volume"]

        # encoder
        skip_connections = {}
        x = self.encoder1(event_volume)
        skip_connections['skip0'] = x.clone()
        x = self.encoder2(x)
        skip_connections['skip1'] = x.clone()
        x = self.encoder3(x)
        skip_connections['skip2'] = x.clone()
        x = self.encoder4(x)
        skip_connections['skip3'] = x.clone()

        # transition
        x = self.resnet_block(x)
        x = self.attention(x)  # Attention機構を適用

        # decoder
        flow_predictions = []
        x = torch.cat([x, skip_connections['skip3']], dim=1)
        x, flow = self.decoder1(x)
        flow_predictions.append(flow)

        x = torch.cat([x, skip_connections['skip2']], dim=1)
        x, flow = self.decoder2(x)
        flow_predictions.append(flow)

        x = torch.cat([x, skip_connections['skip1']], dim=1)
        x, flow = self.decoder3(x)
        flow_predictions.append(flow)

        x = torch.cat([x, skip_connections['skip0']], dim=1)
        x, flow = self.decoder4(x)
        flow_predictions.append(flow)

        return {"flow_predictions": flow_predictions}
            

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)
