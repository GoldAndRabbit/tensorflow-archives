import os
import sys
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.INFO)

ROOT_PATH     = '/home/psdz/repos/tensorflow-archives/tf_deploy/'
TRAIN_NAIVE   = ROOT_PATH + 'census/adult.data'
TEST_NAIVE    = ROOT_PATH + 'census/adult.test'
TRAIN_PATH    = ROOT_PATH + 'census/train.csv'
EVAL_PATH     = ROOT_PATH + 'census/eval.csv'
TEST_PATH     = ROOT_PATH + 'census/test.csv'
PREDICT_PATH  = ROOT_PATH + 'census/test.csv'
MODEL_PATH    = ROOT_PATH + 'tmp/adult_model'
EXPORT_PATH   = ROOT_PATH + 'tmp/adult_export_model'

_CSV_COLUMNS  = [
    'age','workclass','fnlwgt','education','education_num',
    'marital_status','occupation','relationship','race','gender',
    'capital_gain','capital_loss','hours_per_week','native_country', 'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [
    [0],[''],[0],[''],[0],
    [''],[''],[''],[''],[''],
    [0],[0],[0],[''],['']
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

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train':        32561,
    'validation':   16281,
}


def build_model_columns():
    age            = tf.feature_column.numeric_column('age')
    education_num  = tf.feature_column.numeric_column('education_num')
    capital_gain   = tf.feature_column.numeric_column('capital_gain')
    capital_loss   = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education',[
            'Bachelors','HS-grad','11th','Masters','9th','Some-college',
            'Assoc-acdm','Assoc-voc','7th-8th','Doctorate','Prof-school',
            '5th-6th','10th','1st-4th','Preschool','12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',[
            'Married-civ-spouse','Divorced','Married-spouse-absent',
            'Never-married','Separated','Married-AF-spouse','Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',[
            'Husband','Not-in-family','Wife','Own-child','Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',[
            'Self-emp-not-inc','Private','State-gov','Federal-gov',
            'Local-gov','?','Self-emp-inc','Without-pay','Never-worked'])

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation',hash_bucket_size=_HASH_BUCKET_SIZE)

    age_buckets = tf.feature_column.bucketized_column(
        age,boundaries=[18,25,30,35,40,45,50,55,60,65])

    base_columns = [
        education,marital_status,relationship,workclass,occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education','occupation'],hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column(
            [age_buckets,'education','occupation'],
            hash_bucket_size=_HASH_BUCKET_SIZE),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.embedding_column(occupation,dimension=8),
    ]

    return wide_columns,deep_columns


def input_fn(data_path,shuffle,num_epochs,batch_size):
    def parse_csv(value):
        columns = tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        labels = tf.equal(labels, '>50K')
        return features,labels

    dataset = tf.data.TextLineDataset(data_path)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
    dataset = dataset.map(parse_csv,num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def run():
    wide_columns,deep_columns = build_model_columns()
    config    = tf.estimator.RunConfig(save_checkpoints_steps=1000)
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=MODEL_PATH,
        linear_feature_columns=wide_columns,
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[256, 128, 64],
        dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        config=config
    )

    estimator.train(
        input_fn=lambda: input_fn(
            data_path=TRAIN_PATH,
            shuffle=True,
            num_epochs=40,
            batch_size=128
        ),
        steps=4000
    )

    eval_result = estimator.evaluate(
        input_fn=lambda: input_fn(
            data_path=EVAL_PATH,
            shuffle=False,
            num_epochs=1,
            batch_size=40
        )
    )
    print('Test set accuracy:', eval_result)
    pred_dict = estimator.predict(
        input_fn=lambda: input_fn(
            data_path=PREDICT_PATH,
            shuffle=False,
            num_epochs=1,
            batch_size=40
        )
    )
    for pred_res in pred_dict:
        print(pred_res['classes'][0], pred_res['probabilities'][1])

    columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=columns)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(EXPORT_PATH,serving_input_fn)


if __name__ == '__main__':
    run()

