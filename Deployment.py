import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\admin\Desktop\kaggle_diabetes.csv")


df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] = df[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'
]].replace(0, np.NAN)

# numerical_features = [features for features in df.columns if df[features].dtype != 'O']
# print(numerical_features, len(numerical_features))

# for feature in feature_with_nan:
#     print(feature, np.round(df[feature].isnull().mean(), 2), ' % of missing values')

# sns.countplot(df['Outcome'])
# plt.legend()
# plt.show()

# for feature in feature_with_nan:
#     data = df.copy()
#     # data[feature].hist(bins=25)
#     # plt.scatter(df[feature], df['Outcome'])
#     # sns.distplot(df[feature])
#     # sns.boxplot(df[feature])
#     data.boxplot(column=feature)
#     plt.xlabel(feature)
#     # plt.ylabel('Count')
#     plt.title(feature)
#     plt.show()


df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)

# Convert last three features into log normal distribution for further converting into Stand. Normal Dist.

# print(df['BMI'])
skewed_features = ['SkinThickness', 'Insulin', 'BMI']

for feature in skewed_features:
    df[feature] = np.log(df[feature])
# print(df['BMI'])

# feature_with_nan = [features for features in df.columns if df[features].isnull().sum() >0]
# print(feature_with_nan)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop(columns=['Outcome'], axis=1)
y = df['Outcome']

scaling = StandardScaler()
scaling.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))


# Cross validation score
# score = cross_val_score(classifier, X, y, cv=5)
# print(score.mean())

# Hyperparameter tuning
# parameter = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80]}

# grid_score = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)
# grid_score.fit(X, y)
# print(grid_score.best_estimator_)
# print(grid_score.best_score_)
# # print(grid_score.best_params_)



# # Creating a pickle file for the classifier
# filename = r'C:\Users\admin\Desktop\diabetes-prediction-rfc-model.pkl'
# pickle.dump(classifier, open(filename, 'wb'))
#


# creating a pickle file for the classifier
filename = r'C:\Users\admin\Desktop\diabetes-prediction-RFC-model.plk'
pickle.dump(classifier, open(filename, 'wb'))