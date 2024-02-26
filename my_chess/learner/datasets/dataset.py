from typing import Tuple, Union, List
from pathlib import Path
import shutil
import json
import pickle

from torch.utils.data import Dataset as DatasetTorch
import numpy as np
import pandas as pd
from ray.util.multiprocessing import Pool
import ray
from torch.multiprocessing import Manager

class Dataset(DatasetTorch):
    AUTONAME = "complete_generated_dataset"
    LBL_FILE_COUNT = "total_label_files"
    LBL_COUNT = "total_labels"
    CURR_FILE_IDX = "current_file"
    LBL_COUNT_BY_FILE = "_label_counts"
    LBL_BINS = "_cum_label_counts"
    def __init__(
            self,
            dataset_dir,
            seed=None,
            reset=False,
            max_labels_per_file=800000
            ) -> None:
        super().__init__()
        self.seed = seed
        self.reset = reset
        self.dataset_dir = Path(dataset_dir)
        self.label_dir = self.dataset_dir/'labels'
        self.data_dir = self.dataset_dir/'data'
        self.max_labels_per_file = max_labels_per_file
        self.label_data = {
            Dataset.LBL_FILE_COUNT:0,
            Dataset.LBL_COUNT:0,
            Dataset.CURR_FILE_IDX:0,
            Dataset.LBL_COUNT_BY_FILE:[0],
            Dataset.LBL_BINS:[0],
            }
        
    def __len__(self):
        return self.get_label_count()

    def get_file_count(self):
        return self.label_data[self.LBL_FILE_COUNT]

    def get_label_count(self):
        return self.label_data[self.LBL_COUNT]

    def get_curr_file_idx(self):
        return self.label_data[self.CURR_FILE_IDX]

    def get_label_count_by_file(self):
        return self.label_data[self.LBL_COUNT_BY_FILE]

    def get_label_bins(self):
        return self.label_data[self.LBL_BINS]

    def necessary_file(self, idx):
        return np.digitize(idx, self.get_label_bins())-1
    
    def create_database(self):
        """
        With ../labels and ../data directories available, process and save label and data files.
        # TODO how to incorporate/enforce max_labels_per_file?
        """
        raise NotImplementedError
    
    def create_structure(self):
        if (self.reset or
            not (self.label_dir/(self.AUTONAME+".json")).exists()):
            if self.label_dir.exists():
                shutil.rmtree(self.label_dir)
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            
            self.label_dir.mkdir(parents=True)
            self.data_dir.mkdir(parents=True)

            self.create_database()
            
            self.label_data[self.LBL_BINS] = np.cumsum(self.label_data[self.LBL_COUNT_BY_FILE]).tolist()

            with open(self.label_dir/(self.AUTONAME+".json"), 'w') as f:
                json.dump(self.label_data, f)

    def getName(self):
        return self.__class__.__name__
    
    def copy(self, dest_dir:str, subset:Union[int, float]=1., random:bool=True):
        # Created to create a copy on a faster hard drive, however, if read/write speed is a bottle neck,
        # this will also be very slow. Likely better to just provide functionality to create a dataset from
        # a subset of a pgn file, as dataset creation from pgn file provided 68B observations in ~5hours.
        has_lock = False
        if hasattr(self, 'main_lock'):
            self.main_lock = None
            has_lock = True
        class LabelsItr:
            def __init__(self, labels) -> None:
                self.current = -1
                self.labels = labels
            
            def __iter__(self):
                return self

            def __next__(self):
                self.current += 1
                if self.current < len(self.labels):
                    return self.labels.iloc[self.current]
                raise StopIteration

        dest_dir = Path(dest_dir)
        label_dest_dir = dest_dir/'labels'
        data_dest_dir = dest_dir/'data'
        new_label_data = {
            self.LBL_FILE_COUNT:0,
            self.LBL_COUNT:0,
            self.CURR_FILE_IDX:0,
            self.LBL_COUNT_BY_FILE:[0],
            self.LBL_BINS:[0],
            }

        def cpy_data(label):
            dd = self.data_dir/label["file_name"]
            dta = None
            with open(dd, 'rb') as f:
                dta = pickle.load(f)
            dd = data_dest_dir/label["file_name"]
            if not dd.parent.exists():
                dd.parent.mkdir()
            with open(dd, 'wb') as f:
                pickle.dump(dta, f)
                    
        label_dest_dir.mkdir(parents=True, exist_ok=True)
        data_dest_dir.mkdir(parents=True, exist_ok=True)

        indices = np.arange(self.get_label_count())

        if subset != 1:
            size = subset
            if isinstance(subset, float):
                assert 0 < subset < 1
                size = int(self.get_label_count()*subset)
            else:
                assert subset > 1
            
            if random:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(indices, size, replace=False, shuffle=False)
                indices.sort()
            else:
                indices = indices[:size]
        new_label_data[self.LBL_COUNT] = len(indices)
        idx_split_mask_src = np.digitize(indices, self.get_label_bins())-1

        ray.init(num_cpus=5)
        pool = Pool()
        latest_file_fill = 0
        holdover_count = 0
        curr_label_file_idx = 0
        for i in range(idx_split_mask_src.max()+1):
            labels = pd.read_json(self.label_dir/"{}-{}.json".format(self.AUTONAME, i), orient="records")
            process_idxs = indices[idx_split_mask_src==i] - self.label_data[self.LBL_BINS][i]
            dest_idx_split = [y*self.max_labels_per_file-latest_file_fill for y in range(1,len(process_idxs)//self.max_labels_per_file+1)]
            dest_idx_split = np.split(process_idxs, dest_idx_split)
            latest_file_fill = len(dest_idx_split[-1]) % self.max_labels_per_file

            for j, group in enumerate(dest_idx_split, curr_label_file_idx):
                group = group
                lbls = None
                if (label_dest_dir/"{}-{}.json".format(self.AUTONAME, j)).exists():
                    lbls = pd.read_json(label_dest_dir/"{}-{}.json".format(self.AUTONAME, j), orient="records")
                    lbls = lbls.append(labels.iloc[group,:], ignore_index=True)
                else:
                    lbls = labels.iloc[group,:]
                lbls.to_json(label_dest_dir/"{}-{}.json".format(self.AUTONAME, j), orient="records")
                # for l in LabelsItr(lbls):
                #     cpy_data(l)
                pool.map(cpy_data, LabelsItr(lbls))

            label_counts = [len(x) for x in dest_idx_split]
            if holdover_count:
                label_counts[0] += holdover_count
            if i != idx_split_mask_src.max():
                holdover_count = label_counts[-1]
                label_counts = label_counts[:-1]
            new_label_data[self.LBL_COUNT_BY_FILE].extend(label_counts)
        ray.shutdown()

        new_label_data[self.LBL_BINS] = np.cumsum(new_label_data[self.LBL_COUNT_BY_FILE]).tolist()
        with open(label_dest_dir/(self.AUTONAME+".json"), 'w') as f:
            json.dump(new_label_data, f)

        if has_lock:
            self.main_lock = Manager().RLock()



    """
    To prevent memory problems with multiprocessing.
    Provided by https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
    See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662 for summary
    """
    # --- UTILITY FUNCTIONS ---
    @staticmethod
    def strings_to_mem_safe_val_and_offset(strings: List[str]) -> Tuple[np.ndarray,np.ndarray]:
        seqs = [Dataset.string_to_sequence(s) for s in strings]
        return Dataset.pack_sequences(seqs)
    
    @staticmethod
    def mem_safe_val_and_offset_to_string(v, o, index:int) -> Tuple[np.ndarray,np.ndarray]:
        '''
        In case labels represented by file_idx are no longer in memory, use arbitrary index from existing file.
        Either the current idx, or 0 if current index is beyond existing labels. This will be rare, and impact a
        small fraction of a percent of data points, and otherwise still supplies a valid data point. Bandaid necessary
        to allow multiprocessing of a partitioned dataset.
        '''
        index = index if index < len(o) else 0
        seq = Dataset.unpack_sequence(v, o, index)
        return Dataset.sequence_to_string(seq)
    
    @staticmethod
    def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
        return np.array([ord(c) for c in s], dtype=dtype)

    @staticmethod
    def sequence_to_string(seq: np.ndarray) -> str:
        return ''.join([chr(c) for c in seq])

    @staticmethod
    def pack_sequences(seqs: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        values = np.concatenate(seqs, axis=0)
        offsets = np.cumsum([len(s) for s in seqs])
        return values, offsets

    @staticmethod
    def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
        off1 = offsets[index]
        if index > 0:
            off0 = offsets[index - 1]
        elif index == 0:
            off0 = 0
        else:
            raise ValueError(index)
        return values[off0:off1]