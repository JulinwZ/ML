import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


def main():
    # Боковая панель с навигацией
    with open('./notebooks/RGR/model.pkl','rb') as file: 
                model = pickle.load(file) #----------------------------------------------------------------ЗАМЕНИТЬ----------------------------------
    st.sidebar.title('Навигация')
    page = st.sidebar.selectbox('Выберите страницу', ['Информация о наборе данных', 'Визуализация', 'Информация о построенных моделях', 'Предсказания'])

    # Переключение между страницами
    if page == 'Информация о наборе данных':
        show_describe_page()
    elif page == 'Визуализация':
        show_description_page()
    elif page == 'Предсказания':
        show_predict_page()
    elif page == 'Информация о построенных моделях':
        show_info_about_model_page()

def show_describe_page():
    st.header('Информация о наборе данных')
    st.write("Для перехода на иную страницу можете выбрать слева 👈")
    st.write('Датасет, содержащий информацию о ...')
    df = pd.read_csv('./notebooks/RGR/c.csv') #----------------------------------------------------------------ЗАМЕНИТЬ----------------------------------

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
    st.header("Целевой признак")
    st.write("... : Целевой признак, который овтечает за ...")
    st.write("В ходе анализа выяснилось, что можно использовать всего 3 признака, с помощью которых можно уверенно определить значение целевого признака. Эти 3 признака: ..., ..., ... .")

def show_description_page():
    st.header('Визуализация')
    st.write("Для перехода на иную страницу можете выбрать слева 👈")
    request = st.selectbox("Выберите способ визуализации", ["Heatmap", "График рассеивания целевого признака от остальных", "Диаграмма рассеивания двух признаков", "Boxplot - ящик с усами для каждого признака"])
    df = pd.read_csv('./notebooks/RGR/c.csv') #-------------------------------------------------------------------------------------------------------------------------------------------ЗАМЕНИТЬ----------------------------------
    df['Fire Alarm'] = df['Fire Alarm'].replace({'Yes':1, 'No':0})
    if request == "Heatmap":
        st.markdown('<h3>Heatmap</h3>', unsafe_allow_html=True)
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title('Тепловая карта корреляции данных')
        
        st.pyplot(plt)

        plt.close()
    elif request == "График рассеивания целевого признака от остальных":
        fig, axes = plt.subplots(nrows = 7, ncols = 2, figsize=(30,40)) #-------------------------------------------------------------------------------------------------------------------------------------------ЗАМЕНИТЬ----------------------------------
        for idx, feature in enumerate(df.columns[:-1]):
            df.plot(feature, 'Fire Alarm', subplots=True, kind = 'scatter', ax = axes[idx // 2, idx % 2])#-------------------------------------------------------------------------------------------------------------------------------------------ЗАМЕНИТЬ----------------------------------
        st.write(fig)

        # st.markdown('<h3>Гистограмма</h3>', unsafe_allow_html=True)

        # plt.hist(df['Temperature[C]'], bins=10)
        # plt.xlabel('Temperature [C]')
        # plt.ylabel('Count')
        # plt.title('Temperature Distribution')

        # st.pyplot(plt)

        # plt.close()
    elif request == "Диаграмма рассеивания двух признаков":
        st.markdown('<h3>График рассеивания между двумя признаками</h3>', unsafe_allow_html=True)
        x_axis = st.selectbox("Выберите столбец для оси Ox", df.columns)
        y_axis = st.selectbox("Выберите столбец для оси Oy", df.columns)
        plt.scatter(df[x_axis], df[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'{x_axis} vs {y_axis}')

        st.pyplot(plt)

        plt.close()
    elif request == "Boxplot - ящик с усами для каждого признака":
        st.markdown('<h3>Boxplot</h3>', unsafe_allow_html=True)
        selected = st.selectbox('Выберите признак', df.columns)
        plt.boxplot(df[selected])
        plt.xlabel(selected)
        plt.ylabel('Values')
        plt.title(f'Box Plot - {selected}')

        st.pyplot(plt)

        plt.close()



def show_info_about_model_page():
    st.title('Информация об использованных моделях')
    st.write("Для перехода на иную страницу можете выбрать слева 👈")
    st.markdown("""Было обучено три модели для классификации на отбалансированных данных с отобранными признаками:
* LogisticRegressor - данная модель использует всего эти три признака: sensor04, sensor10, sensor12. Она показала достаточно хорошие результаты и при этом, на мой взгляд, не является переобученной.
Здесь представлен отчет того, как справлялась на тестовой выборке данная модель

            classes    precision   recall   f1-score   support
           0       1.00      0.98      0.99     30986
           1       0.87      0.95      0.91     30782
           2       0.93      0.86      0.89     30859

    accuracy: 0.93
   macro avg: 0.93
weighted avg: 0.93

* GaussianNB - модель обучалась на тех же трех признаках. Ее результаты:

            classes   precision    recall  f1-score   support
           0       0.87      0.97      0.92     30986
           1       0.87      0.99      0.92     30782
           2       0.95      0.71      0.82     30859

    accuracy: 0.89
   macro avg: 0.89
weighted avg: 0.89

Целевой признак:
* BaggingClassifier - ансамблевая модель, которая справилась лучше всего, однако нельзя с уверенностью сказать, что она не является переобученной. Также по результатам она обгоняет предыдущие две:

           classes  precision    recall  f1-score   support
           0       1.00      1.00      1.00     30986
           1       1.00      1.00      1.00     30782
           2       1.00      1.00      1.00     30859

    accuracy: 1.00
   macro avg: 1.00
weighted avg: 1.00

        """)



def show_predict_page():
    with open('./notebooks/RGR/model.pkl','rb') as file: 
        model = pickle.load(file) #----------------------------------------------------------------ЗАМЕНИТЬ----------------------------------
    st.title('Предсказания')
    st.write("Для перехода на иную страницу можете выбрать слева 👈")
    
    sensor4 = st.number_input("Задайте значение ... . Минимальное значение 3, максимальное 800:", 3, 800, 10)
    sensor4 = float(sensor4)

    sensor5 = st.number_input("Задайте значение ... . Минимальное значение 3, максимальное 800:", 3, 800, 10)
    sensor5 = float(sensor4)

    sensor6 = st.number_input("Задайте значение ... . Минимальное значение 3, максимальное 800:", 3, 800, 10)
    sensor6 = float(sensor4)
    
    if st.button('Получить предсказание'):
        frame = [sensor4, sensor5, sensor6]
        frame = np.array(frame).reshape((1, -1))
        data_df = pd.DataFrame(frame)
        pred1 = model.predict(data_df)
        st.write(f"Значение, предсказанное с помощью модели Логистической регресиии: {pred1[0]:.2f}, точность ответа: 0.93")
    # uploaded_file = st.file_uploader('Загрузите CSV-файл с данными', type='csv')
    # if uploaded_file is not None:
    #     data = pd.read_csv(uploaded_file)
    #     st.write('Загруженные данные:')
    #     st.dataframe(data)

    # model_list = ['KNN', 'Случайный лес', 'Нейронные сети']
    # selected_model = st.selectbox('Выберите модель обучения', model_list)

    # predict_options = ['Первые пять', 'Все']

    # selected_options = st.selectbox('Количество предсказаний', predict_options)


    # if st.button('Предсказать'):
    #     if selected_model == 'KNN':
    #         with open('./notebooks/RGR/knn_model.pkl','rb') as file:
    #             model = pickle.load(file)
    #             predictions = model.predict(data)
    #         elif selected_model == 'Случайный лес':
    #             with open('./notebooks/RGR/tree_model.pkl','rb') as file:
    #                 model = pickle.load(file)
    #                 predictions = model.predict(data)
    #         elif selected_model == 'Нейронные сети':
    #             model_regression_restored = tf.keras.models.load_model('./models/ClassificationModel1')
    #             predict = model_regression_restored.predict(data)
    #             predictions = []
    #             for q in predict:
    #                 if q[0] > q[1]:
    #                     predictions.append(0)
    #                 else:
    #                     predictions.append(1)
    #         st.write('Предсказания:')
    #         if num_predictions == 5:
    #             predictions_table = pd.DataFrame({'Предсказания': predictions[:5]})
    #         else:
    #             predictions_table = pd.DataFrame({'Предсказания': predictions})
            
    #         st.dataframe(predictions_table)
            
# используем модель
        


if __name__ == '__main__':
    main()

# !pip install streamlit
#!pip install scikit-learn

#!npm install localtunnel

#!streamlit run /content/app.py &>/content/logs.txt &

#!curl https://ipinfo.io/ip

#!npx localtunnel --port 8501