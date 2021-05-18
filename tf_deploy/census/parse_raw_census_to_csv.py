import pandas as pd

ROOT_PATH     = '//'
TRAIN_RAW     = ROOT_PATH + 'census/adult.data'
TEST_RAW      = ROOT_PATH + 'census/adult.test'
TRAIN_PATH    = ROOT_PATH + 'census/train.csv'
EVAL_PATH     = ROOT_PATH + 'census/eval.csv'
TEST_PATH     = ROOT_PATH + 'census/test.csv'
PREDICT_PATH  = ROOT_PATH + 'census/predict.csv'

MODEL_PATH   = '/tmp/adult_model'
EXPORT_PATH  = '/tmp/adult_export_model'

_CSV_COLUMNS = [
    'age','workclass','fnlwgt','education', 'education_num',
    'marital_status','occupation','relationship','race','gender',
    'capital_gain','capital_loss','hours_per_week','native_country','income_bracket'
]

_STRING_COLS = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country',
    'income_bracket',
]

_CSV_COLUMN_DEFAULTS = [
    [0],[''],[0],[''],[0],
    [''],[''],[''],[''],[''],
    [0],[0],[0],[''],[0]
]

train_df = pd.read_csv(TRAIN_RAW,names=_CSV_COLUMNS)
test_df  = pd.read_csv(TEST_RAW,names=_CSV_COLUMNS)

for col in _CSV_COLUMNS:
    if col in _STRING_COLS:
        train_df[col] = train_df[col].map(lambda x: x.replace(' ', ''))
        test_df[col]  = test_df[col].map(lambda x: x.replace(' ', ''))

test_df.sample(frac=1, random_state=2021)
eval  = test_df.loc[: int(len(test_df) * 0.5)]
test  = test_df.loc[int(len(test_df) * 0.5) + 1:]
test.reset_index(drop=True, inplace=True)

train_df.to_csv(TRAIN_PATH, index=False, header=None)
eval.to_csv(EVAL_PATH,      index=False, header=None)
test.to_csv(TEST_PATH,      index=False, header=None)
