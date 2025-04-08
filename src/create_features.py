import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle


train = pd.read_csv('course/Data/adult.data', header=None)
test = pd.read_csv('course/Data/adult.test', header=None)


cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train.columns = cols
test.columns = cols


train['income'] = train['income'].str.strip()
test['income'] = test['income'].str.strip()


X_train = train.drop('income', axis=1)
y_train = train['income']
X_test = test.drop('income', axis=1)
y_test = test['income']


num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])


selector = SelectPercentile(f_classif, percentile=50)


pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', selector)
])


X_train_processed = pipe.fit_transform(X_train, y_train)
X_test_processed = pipe.transform(X_test)


with open('course/Data/pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)


pd.DataFrame(X_train_processed).to_csv('course/Data/processed_train_data.csv', index=False)
pd.DataFrame(X_test_processed).to_csv('course/Data/processed_test_data.csv', index=False)
