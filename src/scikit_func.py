import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_data():
    return pd.read_csv("resource/cardio_train.csv", sep=';')


def get_sidebar_classifier():
    return 'Decision Tree', 'SVM', 'Random Forest', 'KNN'


def prepare_dataset(dataframe):
    X = dataframe.drop(columns=['cardio'])
    y = dataframe['cardio']
    scalar = MinMaxScaler()
    x_scaled = scalar.fit_transform(X)
    return train_test_split(x_scaled, y, test_size=0.30, random_state=9)


def get_model(classifier, params):
    if classifier == 'Decision Tree':
        return DecisionTreeClassifier(criterion=params['criterion'], max_features=params['max_features'],
                                      max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
    elif classifier == 'SVM':
        return SVC(C=params['C'])
    elif classifier == 'Random Forest':
        return RandomForestClassifier(n_estimators=90)
    return KNeighborsClassifier(n_neighbors=params['K'])


def trigger_classifier(classifier, params, X_train, X_test, y_train, y_test):
    model = get_model(classifier, params)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
