EDF_DIRECTORY_PATH = '..\\data_edf'

# параметры файлов
HYPNOGRAM_FILE_PATTERN = '*Hypnogram.edf'
PSG_FILE_PATTERN = '*PSG.edf'
PSG_FILE_EXTENSION = '-PSG.edf'

EDF_ENCODING = 'ISO-8859-1'

# параметры данных
CHANNEL_NAME = 'EEG Fpz-Cz'
SAMPLING_RATE_INFO_KEY = 'sfreq'

EPOCH_SIZE = 30

W, N1, N2, N3, REM, UNKNOWN = 0, 1, 2, 3, 4, 5

STAGES_TYPES_COUNT = 5

STAGES_TYPES = {
    'Sleep stage W': W,
    'Sleep stage 1': N1,
    'Sleep stage 2': N2,
    'Sleep stage 3': N3,
    'Sleep stage 4': N3,
    'Sleep stage R': REM,
    'Sleep stage ?': UNKNOWN,
    'Movement time': UNKNOWN
}
