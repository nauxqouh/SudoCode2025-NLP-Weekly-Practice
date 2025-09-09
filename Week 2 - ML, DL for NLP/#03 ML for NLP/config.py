DATA_PATH = "data/train.csv"
PROCESSED_PATH = "data/processed/"
MODEL_PATH = "models/best_model.pkl"

# Hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

EMOJI_MAP = {
    ":red_heart:": "yêu thích",
    ":relieved_face:": "hài lòng",
    ":beaming_face_with_smiling_eyes:": "cười",
    ":grinning_squinting_face:": "cười",
    ":smiling_face:": "cười",
    ":crying_face:": "buồn"
}

CUSTOM_STOPWORDS = {'thì', 'mà', 'là', 'của', 'với', 'trong', 'trên', 'khi', 'như', 'đã', 'này', 'để', 'tôi'}
