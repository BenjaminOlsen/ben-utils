import utils
import musdb
from pathlib import Path
from timeit import default_timer as timer

musdb_test = musdb.DB(root="/YOUR/PATH/TO/musdb18", subsets="test")
musdb_train = musdb.DB(root="/YOUR/PATH/TO/musdb18", subsets="train")


SAVE_DATAPATH = Path('/WHERE/YOU/WANT/TO/SAVE/THE/SPECTROGRAMS/')
TEST_PATH = SAVE_DATAPATH / 'test'
TRAIN_PATH = SAVE_DATAPATH / 'train'

t1 = timer()
utils.save_musdb_spectrograms(musdb_test, save_dir=TEST_PATH, spec_len_in_s=5.0, n_fft=2048, win_length=2047, spec_dimension=(1024,1024), power=None)
t2 = timer()
print(f"test data made in {t2-t1:.2f}s")
utils.save_musdb_spectrograms(musdb_train, save_dir=TRAIN_PATH, spec_len_in_s=5.0, n_fft=2048, win_length=2047, spec_dimension=(1024,1024), power=None)
t3 = timer()
print(f"train data made in {t3-t2:.2f}s")