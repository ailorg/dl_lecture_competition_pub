import math
from pathlib import Path, PurePath
from typing import Dict, Tuple
from time import time
import cv2
import hdf5plugin
import h5py
from numba import jit
import numpy as np
import os
import imageio
imageio.plugins.freeimage.download()
import imageio.v3 as iio
import torch
import torch.utils.data
from torchvision.transforms import RandomCrop
from torchvision import transforms as tf
from torch.utils.data import Dataset


from src.utils import RepresentationType, VoxelGrid, flow_16bit_to_float

VISU_INDEX = 1


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(
            t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        if t_start_ms_idx is None or t_end_ms_idx is None:
            print('Error', 'start', t_start_us, 'end', t_end_us)
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(
            self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(
                self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class Sequence(Dataset):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str = 'test', delta_t_ms: int = 100,
                 num_bins: int = 4, transforms=[], name_idx=0, visualize=False, load_gt=False):
        assert num_bins >= 1
        assert delta_t_ms == 100
        assert seq_path.is_dir()
        assert mode in {'train', 'test'}
        assert representation_type is not None
        '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─seq_1
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─seq_1
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─seq_2
            └─seq_3
        '''
        self.seq_name = PurePath(seq_path).name
        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize
        self.load_gt = load_gt
        self.transforms = transforms
        if self.mode == "test":
            assert load_gt == False
            # Get Test Timestamp File
            ev_dir_location = seq_path / 'events_left'
            timestamp_file = seq_path / 'forward_timestamps.txt'
            flow_path = seq_path / 'flow_forward'
            timestamps_flow = np.loadtxt(
                seq_path / 'forward_timestamps.txt', delimiter=',', dtype='int64')
            self.indices = np.arange(len(timestamps_flow))
            self.timestamps_flow = timestamps_flow[:, 0]

        elif self.mode == "train":
            ev_dir_location = seq_path / 'events_left'
            flow_path = seq_path / 'flow_forward'
            timestamp_file = seq_path / 'forward_timestamps.txt'
            self.flow_png = [Path(os.path.join(flow_path, img)) for img in sorted(
                os.listdir(flow_path))]
            timestamps_flow = np.loadtxt(
                seq_path / 'forward_timestamps.txt', delimiter=',', dtype='int64')
            self.indices = np.arange(len(timestamps_flow))
            self.timestamps_flow = timestamps_flow[:, 0]
        else:
            pass
        assert timestamp_file.is_file()

        file = np.genfromtxt(
            timestamp_file,
            delimiter=','
        )

        self.idx_to_visualize = file[:, 2] if file.shape[1] == 3 else []

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins


        # Set event representation
        self.voxel_grid = VoxelGrid(
                (self.num_bins, self.height, self.width), normalize=True)
        self.delta_t_us = delta_t_ms * 1000

        # Left events only
        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)

        self.h5rect = h5py.File(str(ev_rect_file), 'r')
        self.rectify_ev_map = self.h5rect['rectify_map'][()]


    def events_to_voxel_grid(self, p, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = iio.imread(str(flowfile), plugin='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (
            self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]
    
    def get_data(self, index) -> Dict[str, any]:
        ts_start: int = self.timestamps_flow[index] - self.delta_t_us
        ts_end: int = self.timestamps_flow[index]

        file_index = self.indices[index]

        output = {
            'file_index': file_index,
            'timestamp': self.timestamps_flow[index],
            'seq_name': self.seq_name
        }
        # Save sample for benchmark submission
        output['save_submission'] = file_index in self.idx_to_visualize
        output['visualize'] = self.visualize_samples
        event_data = self.event_slicer.get_events(
            ts_start, ts_end)
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        if self.voxel_grid is None:
            raise NotImplementedError
        else:
            event_representation = self.events_to_voxel_grid(
                p, t, x_rect, y_rect)
            output['event_volume'] = event_representation
        output['name_map'] = self.name_idx
        
        if self.load_gt:
            output['flow_gt'
                ] = [torch.tensor(x) for x in self.load_flow(self.flow_png[index])]

            output['flow_gt'
                ][0] = torch.moveaxis(output['flow_gt'][0], -1, 0)
            output['flow_gt'
                ][1] = torch.unsqueeze(output['flow_gt'][1], 0)
        return output

    def __getitem__(self, idx):
        sample = self.get_data(idx)
        return sample

    def get_voxel_grid(self, idx):

        if idx == 0:
            event_data = self.event_slicer.get_events(
                self.timestamps_flow[0] - self.delta_t_us, self.timestamps_flow[0])
        elif idx > 0 and idx <= self.__len__():
            event_data = self.event_slicer.get_events(
                self.timestamps_flow[idx-1], self.timestamps_flow[idx-1] + self.delta_t_us)
        else:
            raise IndexError

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        return self.events_to_voxel_grid(p, t, x_rect, y_rect)

    def get_event_count_image(self, ts_start, ts_end, num_bins, normalize=True):
        assert ts_end > ts_start
        delta_t_bin = (ts_end - ts_start) / num_bins
        ts_start_bin = np.linspace(
            ts_start, ts_end, num=num_bins, endpoint=False)
        ts_end_bin = ts_start_bin + delta_t_bin
        assert abs(ts_end_bin[-1] - ts_end) < 10.
        ts_end_bin[-1] = ts_end

        event_count = torch.zeros(
            (num_bins, self.height, self.width), dtype=torch.float, requires_grad=False)

        for i in range(num_bins):
            event_data = self.event_slicer.get_events(
                ts_start_bin[i], ts_end_bin[i])
            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            t = (t - t[0]).astype('float32')
            t = (t/t[-1])
            x = x.astype('float32')
            y = y.astype('float32')
            pol = p.astype('float32')
            event_data_torch = {
                'p': torch.from_numpy(pol),
                't': torch.from_numpy(t),
                'x': torch.from_numpy(x),
                'y': torch.from_numpy(y),
            }
            x = event_data_torch['x']
            y = event_data_torch['y']
            xy_rect = self.rectify_events(x.int(), y.int())
            x_rect = torch.from_numpy(xy_rect[:, 0]).long()
            y_rect = torch.from_numpy(xy_rect[:, 1]).long()
            value = 2*event_data_torch['p']-1
            index = self.width*y_rect + x_rect
            mask = (x_rect < self.width) & (y_rect < self.height)
            event_count[i].put_(index[mask], value[mask], accumulate=True)

        return event_count

    @staticmethod
    def normalize_tensor(event_count):
        mask = torch.nonzero(event_count, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = event_count[mask].mean()
            std = event_count[mask].std()
            if std > 0:
                event_count[mask] = (event_count[mask] - mean) / std
            else:
                event_count[mask] = event_count[mask] - mean
        return event_count


class SequenceRecurrent(Sequence):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str = 'test', delta_t_ms: int = 100,
                 num_bins: int = 15, transforms=None, sequence_length=1, name_idx=0, visualize=False, load_gt=False):
        super(SequenceRecurrent, self).__init__(seq_path, representation_type, mode, delta_t_ms, transforms=transforms,
                                                name_idx=name_idx, visualize=visualize, load_gt=load_gt)
        self.crop_size = self.transforms['randomcrop'] if 'randomcrop' in self.transforms else None
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        continuous_seq_idcs = []
        if self.sequence_length > 1:
            for i in range(len(self.timestamps_flow)-self.sequence_length+1):
                diff = self.timestamps_flow[i +
                                            self.sequence_length-1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        else:
            for i in range(len(self.timestamps_flow)-1):
                diff = self.timestamps_flow[i+1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length-1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        return continuous_seq_idcs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        # Valid index is the actual index we want to load, which guarantees a continuous sequence length
        valid_idx = self.valid_indices[idx]

        sequence = []
        j = valid_idx

        ts_cur = self.timestamps_flow[j]
        # Add first sample
        sample = self.get_data_sample(j)
        sequence.append(sample)

        # Data augmentation according to first sample
        crop_window = None
        flip = None
        if 'crop_window' in sample.keys():
            crop_window = sample['crop_window']
        if 'flipped' in sample.keys():
            flip = sample['flipped']

        for i in range(self.sequence_length-1):
            j += 1
            ts_old = ts_cur
            ts_cur = self.timestamps_flow[j]
            assert(ts_cur-ts_old < 100000 + 1000)
            sample = self.get_data_sample(
                j, crop_window=crop_window, flip=flip)
            sequence.append(sample)

        # Check if the current sample is the first sample of a continuous sequence
        if idx == 0 or self.valid_indices[idx]-self.valid_indices[idx-1] != 1:
            sequence[0]['new_sequence'] = 1
            print("Timestamp {} is the first one of the next seq!".format(
                self.timestamps_flow[self.valid_indices[idx]]))
        else:
            sequence[0]['new_sequence'] = 0

        # random crop
        if self.crop_size is not None:
            i, j, h, w = RandomCrop.get_params(
                sample["event_volume_old"], output_size=self.crop_size)
            keys_to_crop = ["event_volume_old", "event_volume_new",
                            "flow_gt_event_volume_old", "flow_gt_event_volume_new", 
                            "flow_gt_next",]

            for sample in sequence:
                for key, value in sample.items():
                    if key in keys_to_crop:
                        if isinstance(value, torch.Tensor):
                            sample[key] = tf.functional.crop(value, i, j, h, w)
                        elif isinstance(value, list) or isinstance(value, tuple):
                            sample[key] = [tf.functional.crop(
                                v, i, j, h, w) for v in value]
        return sequence


class DatasetProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_ms: int = 100, num_bins=4,
                config=None, visualize=False):
        test_path = Path(os.path.join(dataset_path, 'test'))
        train_path = Path(os.path.join(dataset_path, 'train'))
        assert dataset_path.is_dir(), str(dataset_path)
        assert test_path.is_dir(), str(test_path)
        assert delta_t_ms == 100
        self.config = config
        self.name_mapper_test = []

        # Assemble test sequences
        test_sequences = list()
        for child in test_path.iterdir():
            self.name_mapper_test.append(str(child).split("/")[-1])
            test_sequences.append(Sequence(child, representation_type, 'test', delta_t_ms, num_bins,
                                               transforms=[],
                                               name_idx=len(
                                                   self.name_mapper_test)-1,
                                               visualize=visualize))

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

        # Assemble train sequences
        available_seqs = os.listdir(train_path)

        seqs = available_seqs

        train_sequences: list[Sequence] = []
        for seq in seqs:
            extra_arg = dict()
            train_sequences.append(Sequence(Path(train_path) / seq,
                                   representation_type=representation_type, mode="train",
                                   load_gt=True, **extra_arg))
            self.train_dataset: torch.utils.data.ConcatDataset[Sequence] = torch.utils.data.ConcatDataset(train_sequences)

    def get_test_dataset(self):
        return self.test_dataset

    def get_train_dataset(self):
        return self.train_dataset

    def get_name_mapping_test(self):
        return self.name_mapper_test

    def summary(self, logger):
        logger.write_line(
            "================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(
            self.test_dataset.datasets[0].num_bins), True)
        logger.write_line("Number of Train Sequences: {}".format(
            len(self.train_dataset)), True)

def train_collate(sample_list):
    batch = dict()
    for field_name in sample_list[0]:
        if field_name == 'timestamp':
            batch['timestamp'] = [sample[field_name] for sample in sample_list]
        if field_name == 'seq_name':
            batch['seq_name'] = [sample[field_name] for sample in sample_list]
        if field_name == 'new_sequence':
            batch['new_sequence'] = [sample[field_name]
                                     for sample in sample_list]
        if field_name.startswith("event_volume"):
            batch[field_name] = torch.stack(
                [sample[field_name] for sample in sample_list])
        if field_name.startswith("flow_gt"):
            if all(field_name in x for x in sample_list):
                batch[field_name] = torch.stack(
                    [sample[field_name][0] for sample in sample_list])
                batch[field_name + '_valid_mask'] = torch.stack(
                    [sample[field_name][1] for sample in sample_list])

    return batch


def rec_train_collate(sample_list):
    seq_length = len(sample_list[0])
    seq_of_batch = []
    for i in range(seq_length):
        seq_of_batch.append(train_collate(
            [sample[i] for sample in sample_list]))
    return seq_of_batch
