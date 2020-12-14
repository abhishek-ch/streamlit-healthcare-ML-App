import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure

import pyspark_func as pyspark_func
import scikit_func as scikit_func

matplotlib.use("agg")
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

st.set_page_config(
    page_title='Streamlit with Healthcare Data',
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Decision Tree':
        params['criterion'] = st.sidebar.radio("criterion", ('gini', 'entropy'))
        params['max_features'] = st.sidebar.selectbox("max_features", (None, 'auto', 'sqrt', 'log2'))
        params['max_depth'] = st.sidebar.slider('max_depth', 1, 32)
        params['min_samples_split'] = st.sidebar.slider('min_samples_split', 0.1, 1.0)
    return params


def pyspark_buildmodel(pyspark_classifier_name):
    spark = pyspark_func.get_spark_session()
    trainingData, testData = pyspark_func.prepare_dataset(spark, data)
    return pyspark_func.training(spark, pyspark_classifier_name, trainingData, testData)


def pyspark_operation(pyspark_col):
    st.sidebar.subheader('PySpark')
    pyspark_classifier_name = st.sidebar.selectbox(
        'Select classifier',
        pyspark_func.get_sidebar_classifier(), key='pyspark'
    )
    pyspark_col.write(f'Classifier = {pyspark_classifier_name}')
    accuracy = pyspark_buildmodel(pyspark_classifier_name)
    pyspark_col.write(f'Accuracy = {accuracy}')


def create_sidelayout(scipy_col, pyspark_col):
    st.sidebar.title('Machine Learning Options')
    st.sidebar.subheader('Scikit-Learn')
    scikit_classifier_name = st.sidebar.selectbox(
        'Select classifier',
        scikit_func.get_sidebar_classifier(), key='scikit'
    )
    scipy_col.write(f'Classifier = {scikit_classifier_name}')
    params = add_parameter_ui(scikit_classifier_name)
    accuracy = scikit_func.trigger_classifier(scikit_classifier_name, params, X_train, X_test, y_train, y_test)
    scipy_col.write(f'Accuracy =  {accuracy}')

    if pyspark_enabled == 'Yes':
        pyspark_operation(pyspark_col)


def create_subcol():
    """
    Create 2 Center page Columns for Scikit-Learn and Pyspark
    :return:
    """
    scipy_col, pyspark_col = st.beta_columns(2)
    scipy_col.header('''
    __Scikit Learn ML__  
    ![scikit](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)''')
    if pyspark_enabled == 'Yes':
        pyspark_col.header('''
        __PySpark Learn ML__  
        ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
        ''')
    return scipy_col, pyspark_col


def plot():
    """
    Plotting some basic chart to demonstrate the data
    TODO: Add caching to avoid loading the plots for every event
    :return:
    """
    col_plot1, col_plot2 = st.beta_columns(2)
    temp_df = data
    with col_plot1, _lock:
        st.subheader('Age over Number of people with CVD exceed')
        fig = Figure()
        ax = fig.subplots()
        temp_df['years'] = (temp_df['age'] / 365).round().astype('int')
        sns.countplot(x='years', hue='cardio', data=temp_df, palette="Set2", ax=ax)

        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col_plot2, _lock:
        st.subheader('People Exposed to CVD more')
        fig = Figure()
        ax = fig.subplots()
        df_categorical = temp_df.loc[:, ['cholesterol', 'gluc', 'smoke', 'alco', 'active']]
        sns.countplot(x="variable", hue="value", data=pd.melt(df_categorical), ax=ax)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Count')
        st.pyplot(fig)


data = scikit_func.load_data()
X_train, X_test, y_train, y_test = scikit_func.prepare_dataset(data)
st.sidebar.markdown('''
__⛔ PySpark is computational expensive operation __  
Selecting _Yes_ will trigger Spark Session automatically!  
''', unsafe_allow_html=True)
pyspark_enabled = st.sidebar.radio("PySpark_Enabled", ('No', 'Yes'))


def main():
    st.title(
        '''Streamlit ![](https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e0a328bedb754beb8a973f9_logomark_website.png) Healthcare ML Data App''')
    st.subheader(
        'Streamlit Healthcare example By '
        '[Abhishek Choudhary aka ABC](https://www.linkedin.com/in/iamabhishekchoudhary/)')
    st.markdown(
        "**Cardiovascular Disease dataset by Kaggle** 👇"
    )
    st.markdown('''
    This is source of the dataset and problem statement [Kaggle Link](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
    The dataset consists of 70 000 records of patients data, 11 features + target  
    
    ◀️ Running the same data over ___Scikit Learn & Pyspark ML___ - A simple Comparison and Selection  
    ___This is just for demonstration & absolutely not about Best Machine Learning Model!___  
    ''')
    st.dataframe(data=data.head(20), height=200)
    st.write('Shape of dataset:', X_train.shape)
    st.write('number of classes:', len(np.unique(y_test)))
    scipy_col, pyspark_col = create_subcol()
    create_sidelayout(scipy_col, pyspark_col)
    plot()


if __name__ == '__main__':
    main()
