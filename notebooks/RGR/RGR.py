import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import tensorflow as tf


def main():
    # Боковая панель с навигацией
    st.sidebar.title('Навигация')
    page = st.sidebar.selectbox('Выберите страницу', ['Информация о наборе данных', 'Визуализация', 'Предсказания'])

    # Переключение между страницами
    if page == 'Информация о наборе данных':
        show_describe_page()
    elif page == 'Визуализация':
        show_description_page()
    elif page == 'Предсказания':
        show_predict_page()

def show_describe_page():
    st.header('Информация о наборе данных')
    st.write('Датасет, содержащий информацию о срабатывании датчика дыма, предназначен для мониторинга и анализа условий, связанных с дымом и пожаром. Он содержит данные о различных параметрах, которые могут быть полезными для определения возможных пожарных ситуаций или изменений в окружающей среде, связанных с дымом.')
    df = pd.read_csv('./notebooks/RGR/c.csv')

    st.dataframe(df)

    st.header('Атрибуты дата сета')
    
    attribute_description = [
        "UTC: Это временная метка (timestamp) в формате UNIX, представляющая момент срабатывания датчика дыма.",
        "Temperature[C]: Значение температуры в градусах Цельсия в момент срабатывания датчика.",
        "Humidity[%]: Значение влажности в процентах в момент срабатывания датчика.",
        "TVOC[ppb]: Концентрация летучих органических соединений (Total Volatile Organic Compounds) в частях на миллиард (ppb) в момент срабатывания датчика. Летучие органические соединения могут включать различные газы и пары, такие как альдегиды, углеводороды и другие загрязнители в воздухе.",
        "eCO2[ppm]: Концентрация углекислого газа (equivalent Carbon Dioxide) в частях на миллион (ppm) в момент срабатывания датчика. Это показатель качества воздуха и может служить индикатором загрязнения воздуха.",
        "Raw H2: Значение сырого (непреобразованного) показателя концентрации водорода (H2) в момент срабатывания датчика.",
        "Raw Ethanol: Значение сырого (непреобразованного) показателя концентрации этанола в момент срабатывания датчика. Этанол также может быть индикатором загрязнения воздуха.",
        "Pressure[hPa]: Значение атмосферного давления в гектопаскалях (hPa) в момент срабатывания датчика.",
        "PM1.0: Концентрация частиц размером менее 1.0 микрометра в момент срабатывания датчика.",
        "PM2.5: Концентрация частиц размером менее 2.5 микрометров в момент срабатывания датчика. Частицы данного размера могут включать в себя взвешенные частицы дыма.",
        "NC0.5: Количество частиц размером более 0.5 микрометров на см³ в момент срабатывания датчика.",
        "NC1.0: Количество частиц размером более 1.0 микрометров на см³ в момент срабатывания датчика.",
        "NC2.5: Количество частиц размером более 2.5 микрометров на см³ в момент срабатывания датчика.",
        "CNT: Количество обнаруженных частиц в момент срабатывания датчика.",
        "Fire Alarm: Показатель срабатывания сигнала пожарной тревоги. 'No' означает, что тревога не сработала, 'Yes' означает, что сработала."
    ]

    for i, description in enumerate(attribute_description, start = 1):
        st.markdown(f'{i}. {description}')
    
    st.header('Особенности предобработки')
    st.write('Данный датасет нельзя стандартизировать, поскольку это приведет к ухудшению обучения моделей на этих данных. Проверенно на личном опыте.')

def show_description_page():
    st.header('Визуализация')
    st.markdown('<h3>Heatmap</h3>', unsafe_allow_html=True)
    df = pd.read_csv('./notebooks/RGR/c.csv')
    df['Fire Alarm'] = df['Fire Alarm'].replace({'Yes':1, 'No':0})
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта корреляции данных')
    
    st.pyplot(plt)

    plt.close()

    st.markdown('<h3>Гистограмма</h3>', unsafe_allow_html=True)

    plt.hist(df['Temperature[C]'], bins=10)
    plt.xlabel('Temperature [C]')
    plt.ylabel('Count')
    plt.title('Temperature Distribution')

    st.pyplot(plt)

    plt.close()

    st.markdown('<h3>Диаграмма рассеяния между "Humidity[%]" и "Temperature[C]"</h3>', unsafe_allow_html=True)
    plt.scatter(df['Humidity[%]'], df['Temperature[C]'])
    plt.xlabel('Humidity [%]')
    plt.ylabel('Temperature [C]')
    plt.title('Humidity vs Temperature')

    st.pyplot(plt)

    plt.close()

    st.markdown('<h3>Boxplot</h3>', unsafe_allow_html=True)

    plt.boxplot(df['Humidity[%]'])
    plt.xlabel('Temperature')
    plt.ylabel('Values')
    plt.title('Box Plot - Temperature')

    st.pyplot(plt)

    plt.close()

def show_predict_page():
    st.title('Предсказания')

    uploaded_file = st.file_uploader('Загрузите CSV-файл с данными', type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('Загруженные данные:')
        st.dataframe(data)

    model_list = ['KNN', 'Случайный лес', 'Нейронные сети']
    selected_model = st.selectbox('Выберите модель обучения', model_list)

    predict_options = ['Первые пять', 'Все']

    selected_options = st.selectbox('Количество предсказаний', predict_options)

    if selected_options == 'Первые пять':
        num_predictions = 5
    else:
        num_predictions = None
    
    if uploaded_file is not None:
        if st.button('Предсказать'):
            if selected_model == 'KNN':
                with open('./notebooks/RGR/knn_model.pkl','rb') as file:
                    model = pickle.load(file)
                    predictions = model.predict(data)
            elif selected_model == 'Случайный лес':
                with open('./notebooks/RGR/tree_model.pkl','rb') as file:
                    model = pickle.load(file)
                    predictions = model.predict(data)
            elif selected_model == 'Нейронные сети':
                model_regression_restored = tf.keras.models.load_model('./models/ClassificationModel1')
                predict = model_regression_restored.predict(data)
                predictions = []
                for q in predict:
                    if q[0] > q[1]:
                        predictions.append(0)
                    else:
                        predictions.append(1)
            st.write('Предсказания:')
            if num_predictions == 5:
                predictions_table = pd.DataFrame({'Предсказания': predictions[:5]})
            else:
                predictions_table = pd.DataFrame({'Предсказания': predictions})
            
            st.dataframe(predictions_table)
            
# используем модель
        


if __name__ == '__main__':
    main()
