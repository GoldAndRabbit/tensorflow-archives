import os
import random
import pandas as pd
import numpy  as np
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from mmoe import MMoE

DATA_PATH = '/media/psdz/hdd/Download/census/census_data/'
SEED = 2021
np.random.seed(SEED)
random.seed(SEED)

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self,training_data,validation_data,test_data):
        super().__init__()
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self,logs=None):
        if logs is None:
            logs = {}
        return

    def on_train_end(self,logs=None):
        if logs is None:
            logs = {}
        return

    def on_epoch_begin(self,epoch,logs=None):
        if logs is None:
            logs = {}
        return

    def on_epoch_end(self,epoch,logs=None):
        if logs is None:
            logs = {}
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self,batch,logs=None):
        if logs is None:
            logs = {}
        return

    def on_batch_end(self,batch,logs=None):
        if logs is None:
            logs = {}
        return


def data_preparation():
    # The column names are from https://docs.1010data.com/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/
    train_df = pd.read_csv(DATA_PATH + 'census-income.data', sep=',', names=column_names)
    test_df  = pd.read_csv(DATA_PATH + 'census-income.test', sep=',', names=column_names)

    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']
    # One-hot encoding categorical columns
    categorical_columns = [
        'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
        'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
        'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
        'vet_question'
    ]
    train_raw_labels  = train_df[label_columns]
    other_raw_labels  = test_df[label_columns]
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_other = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)

    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    train_income  = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    other_income  = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)


    dict_outputs = {
        'income' : train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income' : train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income' : other_income,
        'marital': other_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices  = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices        = list(set(transformed_other.index) - set(validation_indices))
    validation_data     = transformed_other.iloc[validation_indices]
    validation_label    = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data   = transformed_other.iloc[test_indices]
    test_label  = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data  = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def census_demo():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_features = train_data.shape[1]

    print('Training   data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test       data shape = {}'.format(test_data.shape))
    print(output_info) # [(2, 'income'), (2, 'marital')]

    input_layer = Input(shape=(num_features,)) #[499,]
    mmoe_layers = MMoE(units=4, num_experts=8, num_tasks=2)(input_layer)
    output_layers = []
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer  = layers.Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling()
        )(task_layer)
        output_layer = layers.Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling()
        )(tower_layer)
        output_layers.append(output_layer)

    model = Model(inputs=[input_layer], outputs=output_layers)
    model.summary()
    adam_optimizer = Adam()
    model.compile(
        loss={
            'income' : 'binary_crossentropy',
            'marital': 'binary_crossentropy',
        },
        optimizer=adam_optimizer,
        metrics=[
            'accuracy'
        ]
    )

    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=5,
        # epochs=100,
    )

if __name__ == '__main__':
    census_demo()
