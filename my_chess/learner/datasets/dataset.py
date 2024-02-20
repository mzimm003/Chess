from typing import Tuple, Union, List

from torch.utils.data import Dataset as DatasetTorch
import numpy as np

class Dataset(DatasetTorch):
    LBL_FILE_COUNT = "total_label_files"
    LBL_COUNT = "total_labels"
    CURR_FILE_IDX = "current_file"
    LBL_COUNT_BY_FILE = "_label_counts"
    LBL_BINS = "_cum_label_counts"
    def __init__(self) -> None:
        super().__init__()
        self.label_data = {
            "total_label_files":0,
            "total_labels":0,
            "current_file":0,
            "_label_counts":[0],
            "_cum_label_counts":[0],
            }
        

    def getName(self):
        return self.__class__.__name__
    
    def copy(self, subset:Union[int, float]=1., random:bool=True):
        pass


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