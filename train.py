# %%
# Importing necessary imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
sns.set()


df = pd.read_csv("data/kidney_disease.csv")
df.head()

df.shape
df.columns


# duplicates - cleaning part
df.duplicated().sum()


# missing value checks
df.isnull().sum().sum()


df.isnull().sum()/len(df)*100


df.info()

# %%
# Finding the unique values in the given columns

for i in df.columns:
    print("************************************", i ,  "****************************")
    print()
    print(set(df[i].tolist()))
    print()

# %%
df.dtypes

df['pcv'] = df['pcv'].apply(lambda x:'43' if x=='\t43' else x)
df['pcv'] = df['pcv'].apply(lambda x:'41' if x=='\t?' else x)
df['wc'] = df['wc'].apply(lambda x:'6200' if x=='\t6200' else x)
df['wc'] = df['wc'].apply(lambda x:'8400' if x=='\t8400' else x)
df['wc'] = df['wc'].apply(lambda x:'9800' if x=='\t?' else x)

df['pcv'].mode()[0]

df['rc'] = df['rc'].apply(lambda x:'5.2' if x=='\t?' else x)
df['classification'] = df['classification'].apply(lambda x:'ckd' if x=='ckd\t' else x)
df['cad'] = df['cad'].apply(lambda x:'no' if x=='\tno' else x)
df['dm'] = df['dm'].apply(lambda x:'yes' if x=='\tyes' else x)
df['dm'] = df['dm'].apply(lambda x:'no' if x=='\tno' else x)
df['dm'] = df['dm'].apply(lambda x:'yes' if x==' yes' else x)


for i in df.select_dtypes(exclude=["object"]).columns:
    df[i]=df[i].apply(lambda x:float(x))

# %%
# Finding the unique values in the given columns

for i in df.columns:
    print("************************************", i ,  "****************************")
    print()
    print(set(df[i].tolist()))
    print()

# %%
df.dtypes

# %%
print(df['pcv'].mode()[0])
print()
print(df['wc'].mode()[0])
print()
print(df['rc'].mode()[0])

# %%
df['pcv'] = df['pcv'].fillna(df['pcv'].mode()[0])
df['wc'] = df['wc'].fillna(df['wc'].mode()[0])
df['rc'] = df['rc'].fillna(df['rc'].mode()[0])

# %%
df['pcv'] = df['pcv'].astype('int64')
df['wc'] = df['wc'].astype('int64')
df['rc'] = df['rc'].astype('float64')

# %%
object_columns = df.select_dtypes(include=['object']).columns
print("Object type Columns :")
print(object_columns)


numerical_columns = df.select_dtypes(include=['int64','float64']).columns
print("\nNumerical type Columns :")
print(numerical_columns)


from sklearn.impute import SimpleImputer
imp_mode1 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mode2 = SimpleImputer(missing_values=np.nan, strategy='median')
df_imp1 = pd.DataFrame(imp_mode1.fit_transform(df[object_columns]))
df_imp1.columns = df[object_columns].columns
df_imp2 = pd.DataFrame(imp_mode2.fit_transform(df[numerical_columns]))
df_imp2.columns = df[numerical_columns].columns

df[object_columns].columns


print(df_imp1.isnull().sum().sum())
print()
print(df_imp2.isnull().sum().sum())

# %%
df_imp2 = df_imp2.iloc[:,1:]
df_imp2


sns.boxplot(y='age', data=df_imp2)

# Checking outlier
def boxplots(col):
    sns.boxplot(df_imp2[col])
    plt.show()

for i in list(df_imp2.select_dtypes(exclude=['object']).columns)[0:]:
    boxplots(i)


df_imp1.columns
df_imp2.columns

# %%
df_imp1['test'] = 'test'
df_imp2['test'] = 'test'


table_df = pd.concat([df_imp1,df_imp2], axis=1)
table_df

table_df = table_df.drop(['test'], axis=1)
table_df.columns

# split the data into independent and dependent variables
x = table_df.drop('classification', axis=1)
y = table_df['classification']

y.value_counts()


y = np.where(y=='ckd', 1,0)

# Handing encoding concept 
def classify_features(x):
    categorical_features =[]
    non_categorical_features = []
    discreate_features = []
    continous_features = []
    for column in x.columns:
        if x[column].dtype=='object':
            if x[column].nunique() < 3:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif x[column].dtype in ['int64','float64']:
            if x[column].nunique() < 100:
                discreate_features.append(column)
            else:
                continous_features.append(column)
    return categorical_features, non_categorical_features, discreate_features, continous_features

categorical, non_categorical, discreate, continous = classify_features(x)

# Label encoder
from sklearn import preprocessing
x [categorical]= x[categorical].apply(preprocessing.LabelEncoder().fit_transform)
x[categorical]


x = pd.concat([x[categorical],x[non_categorical],x[discreate],x[continous]],axis = 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4,random_state=10)
model.fit(x_train, y_train)


from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_test)
accuracy_score(y_test, pred_cv)


import joblib
joblib.dump(model,"classifier.pkl")
joblib.dump(scaler,"scaler.pkl")



