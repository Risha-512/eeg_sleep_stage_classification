from os import path, pardir

NPZ_DIRECTORY_PATH = path.join(pardir, 'data_npz')

# параметры файлов
NPZ_FILE_PATTERN = '*.npz'
NPZ_FILE_EXTENSION = '.npz'

# параметры данных
RAW_VALUES_KEY = 'raw_values'
STAGE_VALUES_KEY = 'stage_values'
SAMPLING_RATE_KEY = 'sampling_rate'
CHANNEL_NAME_KEY = 'channel_name'
HEADER_RAW_KEY = 'header_raw'
HEADER_STAGES_KEY = 'header_stages'

NPZ_KEYS = [
    RAW_VALUES_KEY,
    STAGE_VALUES_KEY,
    SAMPLING_RATE_KEY,
    CHANNEL_NAME_KEY,
    HEADER_RAW_KEY,
    HEADER_STAGES_KEY
]
