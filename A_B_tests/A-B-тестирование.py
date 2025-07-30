#!/usr/bin/env python
# coding: utf-8

# # Проведение А/В-тестирования интернет-магазина

# ### Описание проекта
# Для оценки изменений, связанных с внедрением улучшенной рекомендательной системы в интернет-магазине, проведено А/В-тестирование.

# ### Техническое задание проекта
# - Название теста: recommender_system_test ;
# - Группы: А (контрольная), B (новая платёжная воронка);
# - Дата запуска: 2020-12-07;
# - Дата остановки набора новых пользователей: 2020-12-21;
# - Дата остановки: 2021-01-04;
# - Ожидаемое количество участников теста: 15% новых пользователей из региона EU;
# - Назначение теста: тестирование изменений, связанных с внедрением улучшенной рекомендательной системы;
# - Ожидаемый эффект: за 14 дней с момента регистрации в системе пользователи покажут улучшение каждой метрики не менее, чем на 5 процентных пунктов: 
#   - конверсии в просмотр карточек товаров — событие product_page
#   - просмотры корзины — product_cart
#   - покупки — purchase.

# ### Описание данных
# *Структура датасета ab_project_marketing_events.csv - календарь маркетинговых событий на 2020 год:*
# - `name` — название маркетингового события;
# - `regions` — регионы, в которых будет проводиться рекламная кампания;
# - `start_dt` — дата начала кампании;
# - `finish_dt` — дата завершения кампании.
# 
# *Структура датасета final_ab_new_users.csv - все пользователи, зарегистрировавшиеся в интернет-магазине в период с 7 по 21 декабря 2020 года:*
# - `user_id` — идентификатор пользователя;
# - `first_date` — дата регистрации;
# - `region` — регион пользователя;
# - `device` — устройство, с которого происходила регистрация.
# 
# *Структура датасета final_ab_events.csv - все события новых пользователей в период с 7 декабря 2020 по 4 января 2021 года:*
# - `user_id` — идентификатор пользователя;
# - `event_dt` — дата и время события;
# - `event_name` — тип события;
# - `details` — дополнительные данные о событии. Например, для покупок, purchase , в этом поле хранится стоимость покупки в долларах.
# 
# *Структура датасета final_ab_participants.csv - таблица участников тестов:*
# - `user_id` — идентификатор пользователя;
# - `ab_test` — название теста;
# - `group` — группа пользователя.

# ### Цель исследования
# - Проверить данные на соответствие ТЗ
# - Провести исследовательский анализ
# - Проанализировать результаты теста и дать рекомендации

# ### Декомпозиция
# 
# **Шаг 1. Загрузка данных**
# 
# **Шаг 2. Предобработка данных**
# - Исследовать пропущенные значения;
# - Исследовать соответствие типов;
# - Исследовать дубликаты;
# - Проверить корректность наименований колонок;
# - Переименовать колонки, если это необходимо;
# - Удалить дубликаты;
# - Привести типы;
# - Заменить пропущенные значения, если это возможно;
# 
# **Шаг 3. Проверка соответствия данных теста Техническому заданию:**
# - Выделить пользователей участвующих в тесте и проверить: период набора пользователей в тест и его соответствие требованиям технического задания;
# - Проверить следуюзщее условие: все ли попавшие в тест пользователи представляют целевой регион и составляет ли общее количество пользователей из целевого региона 15% от общего числа пользователей из целевого региона, зарегистрированных в период набора пользователей в тест;
# - Проверить равномерность распределения пользователей по группам теста и корректность их формирования;
# - Удостовериться, что нет пересечений с конкурирующим тестом и нет пользователей, участвующих в двух группах теста одновременно.
# 
# 
# **Шаг 4. Изучить данные о пользовательской активности:**
# - проверить, есть пользователи, которые не совершали событий после регистрации, изучите их количество и распределение между группами теста; 
# - Оставить только те события, которые были совершены в первые 14 дней с момента регистрации; принять решение, оставлять ли пользователей, не проживших 14 дней.
# - Оценить, когда пользователи совершают свои первые события каждого вида, построить гистограмму возраста событий.
# - Проверить, какого размера выборки потребуются для получения достоверного результата, при базовой конверсии 50%.
# 
# 
# **Шаг 5. Исследовательский анализ данных**
# - Распределение количества событий на пользователя в разрезе групп теста, сравнить её средние значения между собой у групп теста;
# - Динамика количества событий в группах теста по дням: изучить распределение числа событий по дням и сравнить динамику групп теста между собой.
# - Убедиться, что время проведения теста не совпадает с маркетинговыми и  другими активностями. Настроить автоматическую проверку, выдающую список событий, пересекающихся с тестом. При необходимости оценить воздействие маркетинговых событий на динамику количества событий.
# - Построить простые продуктовые воронки для двух групп теста с учетом логической последовательности совершения событий; 
# - Изучить изменение конверсии в продуктовой воронке тестовой группы, по сравнению с контрольной и ответить на вопрос: наблюдается ли ожидаемый эффект увеличения конверсии в группе В на 10 процентных пунктов, относительно конверсии в группе А?
# 
# **Шаг 6. Провести оценку результатов A/B-тестирования:**
# - Проверить статистическую разницу долей z-критерием по событиям.
# 
# **Шаг 6. Общий вывод:**
# - Общее заключение о корректности проведения теста и рекомендации.

# ### Шаг 1. Загрузка данных

# Импортируем необходимые библиотеки и загружаем данные из файлов в датафреймы:

# In[1]:


# необходимо обновить библиотеку для корректного отображения графиков
get_ipython().system('pip install plotly==5.9.0')


# In[2]:


import pandas as pd
import datetime as dt

import numpy as np
from scipy import stats as st
import math as mth

import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")


# In[3]:


try:
    market_events = pd.read_csv('~/Desktop/учеба яндекс/A_B_tests_final/ab_project_marketing_events.csv')
    users = pd.read_csv('/Desktop/учеба яндекс/final_project/final_ab_new_users.csv')
    ab_events = pd.read_csv('/Desktop/учеба яндекс/final_project/final_ab_events.csv')
    ab_participants = pd.read_csv('/Desktop/учеба яндекс/final_project/final_ab_participants.csv')
except:
    market_events = pd.read_csv('https://code.s3.yandex.net/datasets/ab_project_marketing_events.csv')
    users = pd.read_csv('https://code.s3.yandex.net/datasets/final_ab_new_users.csv')
    ab_events = pd.read_csv('https://code.s3.yandex.net/datasets/final_ab_events.csv')
    ab_participants = pd.read_csv('https://code.s3.yandex.net/datasets/final_ab_participants.csv')


# **Используемые функции:**

# In[4]:


# функция на вывод основных характеристик датасета
def data(data, name):
    # Выведем 5 строк датафреймов
    display(data.head(10).style.set_caption(name))
    # Выведем основные характеристики датафрейма (типы столбцов, пропущенные значения)
    print(data.info())
    print()
    # Проверим, присутствуют ли пропуски в датафрейме:
    print(f'Количество пропусков в датафрейме {name} - {data.isna().sum()}') 
    print()


# In[5]:


# функция на вывод уникальных строковых значений
def unique_rows(data):
    for i in data.columns:
        if data[i].dtype == 'O' and i != 'user_id':
            print(f'Уникальные значения в столбце {i}')
            print( "\n".join(map(str, data[i].unique())), sep='\n')
            print()


# Рассмотрим полученные данные из датафреймов:

# In[6]:


dataframes = [market_events, users, ab_events, ab_participants]
dataframes_names = list(['market_events', 'users', 'ab_events', 'ab_participants'])


# In[7]:


count = 0
for i in dataframes:
    print('Характеристики датафрейма', dataframes_names[count])
    data(i, dataframes_names[count])
    count += 1


# **Вывод:**
# 
# - всего представлены четыре датасета - market_events, users, ab_events и ab_participants;
# - в датасетах market_events, users, ab_events и ab_participants 14, 61733, 440317 и 18268 строк соответственно;
# - 377577 пропусков обнаружено в датасете ab_events, в столбце details (дополнительные данные о событиях);
# - отметим, что в датасетах market_events, users, ab_events столбцы со временем относятся к типу данных object.

# ### Шаг 2. Предобработка данных

# Приведем столбцы датасетов market_events (столбцы `start_dt` и `finish_dt`), users (столбец `first_date`) и ab_events(`event_dt`) к типу времени datetime:

# In[8]:


market_events['start_dt'] = pd.to_datetime(market_events['start_dt'], format='%Y-%m-%d %H:%M:%S')
market_events['finish_dt'] = pd.to_datetime(market_events['finish_dt'], format='%Y-%m-%d %H:%M:%S')

users['first_date'] = pd.to_datetime(users['first_date'], format='%Y-%m-%d %H:%M:%S')

ab_events['event_dt'] = pd.to_datetime(ab_events['event_dt'], format='%Y-%m-%d %H:%M:%S')


# Проверим, присутствуют ли явные дубликаты в датафрейме:

# In[9]:


count = 0
for i in dataframes:
    print(f'Всего дупликатов в датафрейме {dataframes_names[count]} - {i.duplicated().sum()}, ' 
          f'то есть {round(100 * i.duplicated().sum() / len(i), 2)} процентов от всех данных')
    count += 1


# Проверим, присутствуют ли неявные дубликаты в датафреймах market_events:

# In[10]:


unique_rows(market_events)
unique_rows(users)
unique_rows(ab_events)
unique_rows(ab_participants)


# В датасетах не обнаружены неявные дупликаты.
# 
# Рассмотрим столбец details датасета ab_events, в котором были обнаружены пропуски:

# In[11]:


print(f'Количество пропусков в столбце details - {ab_events["details"].isna().sum()}, '
     f'что составляет {round(100 * ab_events["details"].isna().sum() / len(ab_events), 2)} %') 


# In[12]:


ab_events.query('details.isna()').pivot_table(index='event_name', values='details', aggfunc='count')


# Судя по всему, заполнение столбца `details` происходит только при оплате - в столбец вносится сумма покупки. В остальных случаях (например, регистрация), данные не записываются. Следовательно, удалять пропуски в столбце не имеет смысла.

# ### Шаг 3. Проверка данных на соответствие ТЗ:

# - период набора пользователей в тест и его соответствие требованиям технического задания;

# In[13]:


# проверим минимальную и максимальную дату столбца first_date датасета users
print(f'Начало периода - {min(users["first_date"])}, конец периода - {max(users["first_date"])}')


# По ТЗ все пользователи зарегистрировались в период с 7 по 21 декабря 2020 года. В данном случае, данных чуть больше.

# Оставим данные до 21 декабря включительно:

# In[14]:


users = users[users["first_date"].dt.date <= dt.datetime.strptime('2020-12-21', '%Y-%m-%d').date()]


# In[15]:


# проверим минимальную и максимальную дату столбца event_dt датасета ab_events
print(f'Начало периода - {min(ab_events["event_dt"])}, конец периода - {max(ab_events["event_dt"])}')


# По ТЗ события новых пользователей совершались в период с 7 декабря 2020 по 4 января 2021. Отсутствуют данные за 5 дней. Причинами отсутствия могут быть неверная выгрузка данных, намеренная остановка теста ввиду набора достаточного количества данных.

# - Проверим следуюзщее условие: все ли попавшие в тест пользователи представляют целевой регион и составляет ли общее количество пользователей из целевого региона 15% от общего числа пользователей из целевого региона, зарегистрированных в период набора пользователей в тест;

# Найдем идентификаторы участников теста `recommender_system_test` из датасета ab_participants:

# In[16]:


eu = ab_participants.query('ab_test == "recommender_system_test"')['user_id']


# Посчитаем соотношение количества участников теста ко всем пользователям из целевого региона, зарегистрированных в период набора:

# In[17]:


print('Итого получено следующее соотношение:',
     int(round(100 * int(users.query('user_id in @eu').pivot_table(index='region', values='device', aggfunc='count').iloc[2]) / 
           int(users.query('region == "EU"').pivot_table(index='region', values='device', aggfunc='count').iloc[0]), 2)),
     '% количества участников теста ко всем пользователям из целевого региона')


# Полученное соотношение соответствует ТЗ.

# - оценить равномерность распределения пользователей по группам теста и корректность их формирования;

# Оставим в датасете только пользователей из целевого региона:

# In[18]:


users = users.query('region == "EU"')


# In[19]:


(px.histogram(users.query('user_id in @eu').merge(ab_participants.query('ab_test == "recommender_system_test"'), on='user_id'),
             x='first_date', 
             color='group',
             color_discrete_map={'A': '#feacd2', 'B': 'royalblue'},
             barmode="overlay") 
   .update_layout(plot_bgcolor='rgba(0,0,0,0)',
               autosize=False,
               title={
              'text': "Динамика набора пользователей в группы теста",
               'y':0.98,
               'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'
                },
                xaxis_title="Дата",
                yaxis_title="Частота",
                font=dict(
                size=15),
                width=950,
                height=900,
                legend=dict(
                title='Группы пользователей',
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.7
                )) 
   .update_xaxes(showline=True,
              linewidth=2,
              linecolor='black',
              gridcolor='LightGrey',))


# Контрольная и тестовая группа представляют собой две выборки из генеральной совокупностии и должны отражать ее основные характеристики. Судя по гистограмме, события распределены похожим образом по группам А и В, но в группе А регистраций происходит больше.

# Сколько пользователей в каждой экспериментальной группе?

# In[20]:


group_one = (ab_participants.query('ab_test=="recommender_system_test"')
                        .pivot_table(index='group', values='user_id', aggfunc='count').reset_index())
group_one.columns = ['group', 'amount']
group_one['share'] = round(100 *  group_one['amount'] / sum(group_one['amount']), 2)
display(group_one)
print('Всего пользователей в группе А и В -', group_one['amount'].sum())


# Видим, что пользователей из контрольной группы больше.

# In[21]:


print('Всего уникальных пользователей теста recommender_system_test -', 
      ab_participants.query('ab_test=="recommender_system_test"')['user_id'].nunique())


# Пересечений между группами А и В теста `recommender_system_test` нет.
# 
# Проверим, есть ли пересечение пользователей между конкурирующими тестами recommender_system_test и interface_eu_test:

# In[22]:


group_two = (ab_participants.query('ab_test=="interface_eu_test"')
                        .pivot_table(index='group', values='user_id', aggfunc='count').reset_index())
group_two.columns = ['group', 'amount']
group_two['share'] = round(100 *  group_two['amount'] / sum(group_two['amount']), 2)
display(group_two)
print('Всего пользователей в группе А и В теста interface_eu_test -', group_two['amount'].sum())


# In[23]:


print('Всего уникальных пользователей теста recommender_system_test -', ab_participants['user_id'].nunique())


# In[24]:


if group_one['amount'].sum() + group_two['amount'].sum() > ab_participants['user_id'].nunique():
    print('Количество пользователей, которые попали в два теста одновременно -', 
         group_one['amount'].sum() + group_two['amount'].sum() - ab_participants['user_id'].nunique(),
         ', что составляет', 
          round(100 * (group_one['amount'].sum() + group_two['amount'].sum() - ab_participants['user_id'].nunique()) /
         ab_participants['user_id'].nunique(), 2), '%')


# Таких пользователей достаточно много - 9.61%. Удалив их, може потерять в мощности теста. Контрольная группа отличается тем, что на нее не производится никакого воздействия/изменения в отличие от тестовой группы. Поэтому оценим влияние группы В конкурирующего теста на наши группы:

# In[25]:


# найдем отношение пересекающихся пользователей тестовых групп в двух тестах к общему количеству уникальных пользователей
# тестовой группы теста recommender_system_test
round((len(set(ab_participants.query('ab_test == "recommender_system_test" and group == "B"')['user_id'])
     .intersection(ab_participants.query('ab_test == "interface_eu_test" and group == "B"')['user_id'])) / 2877), 3)


# In[26]:


# найдем отношение пересекающихся пользователей тестовой группы конкурирующего теста и контрольной группы нашего теста
# к общему количеству уникальных пользователей контрольной группы теста recommender_system_test
round((len(set(ab_participants.query('ab_test == "recommender_system_test" and group == "A"')['user_id'])
     .intersection(ab_participants.query('ab_test == "interface_eu_test" and group == "B"')['user_id'])) / 3824), 3)


# Доли между собой практически равны, следовательно, конкурирующий тест практически одинаково влияет на пользователей двух групп нашего теста - значит, оставляем пересекающихся пользователей.

# **Вывод:**
# 
# - Привели столбцы датасетов market_events (столбцы start_dt и finish_dt), users (столбец first_date) и ab_events(event_dt) к типу времени datetime
# - явные и неявные дупликаты не обнаружены
# - обнаружены пропуски (85,75%) в столбце details датасета ab_events. Пропуски оставляем как есть, так как заполнение столбца details происходит только при оплате.
# - в датасете users отфильрованы значения по дате регистрации в соответствии с ТЗ - с 7 по 21 декабря 2020 г.
# - в датасете ab_events соыбтия совершаются с 7 по 30 декабря 2020, что не соответствует ТЗ. По ТЗ события новых пользователей совершались в период с 7 декабря 2020 по 4 января 2021. Отсутствуют данные за 5 дней. Причинами отсутствия могут быть неверная выгрузка данных, намеренная остановка теста ввиду набора достаточного количества данных.
# - События распределены схожим образом по группам А и В, но в группе А регистраций происходит больше. Также пользователей больше в группе А.
# - Пересечений между группами А и В теста recommender_system_test нет выявлено.
# - Пересечения между группами двух тестов есть - принято решение сохранить пользователей.

# ### Шаг 4. Изучение данных о пользовательской активности:

# - активность пользователей: есть ли пользователи, которые не совершали событий после регистрации, изучите их количество и распределение между группами теста; 

# Перед исследованием объединим датасеты ab_participants, users и ab_events:

# In[27]:


new_data = (ab_participants.merge(users, how='left', on='user_id')
                          .query('ab_test =="recommender_system_test" and region == "EU"')
                          .merge(ab_events, how='left', on='user_id'))
new_data.head()


# In[28]:


# количество пользователей по тестам
(new_data.pivot_table(index='group', values='user_id', aggfunc='nunique')
        .reset_index())


# In[29]:


print('Всего пользователей, которые не совершали никаких действий после регистрации:',
      new_data.pivot_table(index='user_id', values='event_name', aggfunc='count').reset_index()
      .query('event_name == 0')['user_id'].nunique())


# Рассмотрим, как эти пользователи распределены по группам тестирования:

# In[30]:


(new_data.query('event_dt.isna() and event_name.isna()')
        .pivot_table(index='group', values='device', aggfunc='count')
        .reset_index())


# In[31]:


not_active_users = (new_data.pivot_table(index='group', values='user_id', aggfunc='nunique')
                            .reset_index().merge(new_data.query('event_dt.isna() and event_name.isna()')
                            .pivot_table(index='group', values='user_id', aggfunc='nunique')
                            .reset_index(), on='group'))
not_active_users.columns = ['group', 'all_users', 'not_active']
not_active_users['share'] = round(not_active_users['not_active'] / not_active_users['all_users'], 2)
not_active_users


# Неактивные пользователи распределены неравномерно по датасету. Необходимо понять, почему так много пользователей попали в контрольную группу тестирования. При этом, так как пользователи не совершают никаких действий, нельзя оценить какие-либо изменения, поэтому удалим таких пользователей:

# In[32]:


new_data = new_data.query('~event_dt.isna() and ~event_name.isna()')


# - проверить, что все участники теста имели возможность совершать события все 14 дней с момента регистрации, также оценить, когда пользователи совершают свои первые события каждого вида.

# Добавим новый столбцец - `разницу в днях между датой соверешения события и датой регистрации`:

# In[33]:


new_data['age_event'] = (new_data['event_dt'].dt.date - new_data['first_date'].dt.date).dt.days


# Оценим возраст событий для каждого действия, совершенного пользователем:

# In[34]:


print('Всего пользователей, не прожившие 14 дней -',
      new_data.query('age_event > 14')['user_id'].nunique())


# In[35]:


px.histogram(new_data.query('group == "A"'),
            x='age_event', color='event_name',
            color_discrete_map={'purchase': '#feacd2', 'product_cart': '#9fc5e8', 'product_page':'#d7d3f4', 'login':'#f9dd8e'}, 
             opacity=0.9,
            ) \
.update_layout(plot_bgcolor='rgba(0,0,0,0)',
               autosize=False,
               title={
              'text': "Возраст событий в группе А",
               'y':0.98,
               'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'
                },
                xaxis_title="Возраст события",
                yaxis_title="",
                font=dict(
                size=15),
                width=950,
                height=900,
                legend=dict(
                title='События',
                yanchor="bottom",
                y=0.81,
                xanchor="left",
                x=0.8
                )) \
.update_xaxes(showline=True,
              linewidth=2,
              linecolor='black',
              gridcolor='LightGrey',)


# In[36]:


px.histogram(new_data.query('group == "B"'),
            x='age_event', color='event_name',
            color_discrete_map={'purchase': '#feacd2', 'product_cart': '#9fc5e8', 'product_page':'#d7d3f4', 'login':'#f9dd8e'}, 
             opacity=0.9,
            ) \
.update_layout(plot_bgcolor='rgba(0,0,0,0)',
               autosize=False,
               title={
              'text': "Возраст событий в группе В",
               'y':0.98,
               'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'
                },
                xaxis_title="Возраст события",
                yaxis_title="Частота",
                font=dict(
                size=15),
                width=950,
                height=900,
                legend=dict(
                title='События',
                yanchor="bottom",
                y=0.81,
                xanchor="left",
                x=0.8
                )) \
.update_xaxes(showline=True,
              linewidth=2,
              linecolor='black',
              gridcolor='LightGrey',)


# Большая часть пользователей из группы А и В совершают события ранее 14 дней. Наиболее активно пользователи совершают события в первые 5 дней. Оставим тех пользователей, которые 14 дней не прожили.

# Отфильтруем датасет new_data так,чтобы в него вошли только те события, которые были совершены в первые 14 дней с момента регистрации:

# In[37]:


new_data = new_data.query('age_event < 15')


# - Проверим, какого размера выборки потребуются для получения достоверного результата, при базовой конверсии 50%

# Для этого воспользуемся калькулятором, указав базовую конверсию в 50%:

# ![image.png](attachment:image.png)

# In[38]:


print('Количество пользователей в группе А:', new_data.query('group == "A"')['user_id'].nunique())
print('Количество пользователей в группе B:', new_data.query('group == "B"')['user_id'].nunique())


# Значит, тестовая группа не соответствует требованиям. Возможно, при такой выборке получить достоверный результат не получится.

# **Вывод:**
# 
# - Исключены неактивные пользователи из контрольной и тестовой группы. Пользователи распределены неравномерно по датасету. Необходимо понять, почему так много неактивных пользователей попали в контрольную группу тестирования.
# - Сохранены пользователи, которые не прожили 14 дней ввиду того, что бОльшая часть пользователей совершают события в пределах 14 дней.
# - Исключены события, которые совершены позднее 14 дней.
# - Тестовая группа не соответствует требованиям по размерам выборки. При такой выборке получить достоверный результат не получится.

# ### Шаг 5. Исследовательский анализ данных

# - Распределение количества событий на пользователя в разрезе групп теста:

# In[39]:


events_users = new_data.groupby(['user_id', 'event_name', 'group'], as_index=False).agg({'device':'count'})


# In[40]:


events_users = events_users.groupby(['group', 'event_name'], as_index=False).agg({'device':'mean'})
events_users.columns = ['group', 'event_name', 'mean_value']
events_users['mean_value'] = round(events_users['mean_value'], 2)


# In[41]:


(px.bar(events_users,
        x='event_name',
        y='mean_value',
        color='group',  
        color_discrete_map={'A': '#feacd2', 'B': '#9fc5e8'},
        text_auto=True,
        barmode='group') 
  .update_layout(plot_bgcolor='rgba(0,0,0,0)') 
  .update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='LightGrey') 
  .update_layout(plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=950,
        height=600,
        title = {
                'text': "Среднее количество событий на пользователя",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
        },
        legend=dict(
                title='Группы пользователей',
                yanchor="bottom",
                y=0,
                xanchor="left",
                x=0.7
                ),
        xaxis_title="События",
        yaxis_title="Среднее количество",
        font=dict(
        size=15),
        ))


# В группе А в среднем на пользователя приходится около 3 событий, в группе В данный показатель чуть ниже - около 2.5-2.6 события.

# - Распределение по устройствам:

# In[42]:


device_a = new_data.query('group == "A"').pivot_table(index='device', values='user_id', aggfunc='nunique').reset_index()
device_a['share'] = round(device_a['user_id'] / device_a['user_id'].sum(), 2)

device_b = new_data.query('group == "B"').pivot_table(index='device', values='user_id', aggfunc='nunique').reset_index()
device_b['share'] = round(device_b['user_id'] / device_b['user_id'].sum(), 2)

devices = device_a.merge(device_b, on='device').rename(columns={'user_id_x':'users_a', 
                                                      'share_x': 'share_a',
                                                      'user_id_y':'users_b', 
                                                      'share_y': 'share_b'})
devices


# In[43]:


px.bar(devices,
    x='device',
    y=['share_a', 'share_b'],
    #color='group',  
    color_discrete_map={'share_a': '#feacd2', 'share_b': '#9fc5e8'},
    text_auto=True,
    barmode='group') \
    .update_layout(plot_bgcolor='rgba(0,0,0,0)') \
    .update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='LightGrey') \
    .update_layout(plot_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    width=950,
    height=600,
    title = {
            'text': "Распределение устройств по группам",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
    },
    legend=dict(
                title='Группы пользователей',
                yanchor="bottom",
                y=0.81,
                xanchor="left",
                x=0.7
                ),
    xaxis_title="Устройства",
    yaxis_title="Доля",
    font=dict(
    size=15),
    )


# Почти у половины пользователей группы А и В (44% и 46%) - Android. Типом устройства Mac пользуются не так активно - всего около 10% в обех группах. У iPhone 20% и 21% в группе А и В соответственно. В дальнейшем можно проанализировать, почему пользователей с Android в два раза больше пользователей iPhone, связано ли это с интерфейсом приложения, наличием багов и т.д.

# - Динамика количества событий в группах теста по дням: изучим распределение числа событий по дням и сравним динамику групп теста между собой.

# In[44]:


px.histogram(new_data.pivot_table(index=['group', 'event_dt'], values='user_id', aggfunc='count').reset_index(),
            x='event_dt', color='group',
            color_discrete_map={'A': '#feacd2', 'B': 'royalblue'},
            barmode="overlay") \
.update_layout(plot_bgcolor='rgba(0,0,0,0)',
               autosize=False,
               title={
              'text': "Распределение числа событий по дням",
               'y':0.98,
               'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'
                },
                xaxis_title="Дата",
                yaxis_title="Частота",
                font=dict(
                size=15),
                width=950,
                height=700,
                legend=dict(
                title='Группы пользователей',
                yanchor="bottom",
                y=0.81,
                xanchor="left",
                x=0.07
                )) \
.update_xaxes(showline=True,
              linewidth=2,
              linecolor='black',
              gridcolor='LightGrey',)


# На графике видно, как повлияло удаление неактивных пользователей: в группе В событим много меньше по сравнению с группой А. Резкий подъем количества событий у группы А наблюдается с 14 декабря - можно предположить, что на это могли повлиять новогодние/рождественские праздники - люди совершают больше действий, покупок и т.д. До 21 декабря у групп с каждым днем все больше событий, после этого количество событий снижается.

# - Убедимся, что время проведения теста не совпадает с маркетинговыми и  другими активностями.

# Оставим в датасете с маркетинговыми событиями market_events только те события, которые относятся к целевому региону:

# In[45]:


list = []

count = 0
for i in market_events['regions']:
    if 'EU' in i:
        list.append(market_events.iloc[count])
    count += 1     
market_events_new = pd.DataFrame(list) 
market_events_new


# Зададим два условия - начало маркетингового события меньше или равно дате максимального события датасета new_data (30 декабря) и конец маркетингового события больше или равно дате минимального события датасета new_data (7 декабря):

# In[46]:


a = market_events_new['start_dt'] <= max(new_data['event_dt'])
b = market_events_new['finish_dt'] >= min(new_data['event_dt'])
a_b = (a & b).reset_index()
a_b.columns = ['index', 'bool']


# In[47]:


count = 0
for i in a_b['bool']:
    if i:
        print('Маркетинговые события, пересекающиеся с датой проведения теста:',
              market_events_new.iloc[count]['name'])
    count += 1


# В момент проведения теста попадает одно маркетинговое событие: Christmas&New Year Promo

# In[48]:


fig = px.histogram(new_data.pivot_table(index=['group', 'event_dt'], values='user_id', aggfunc='count').reset_index(),
            x='event_dt', color='group',
            color_discrete_map={'A': '#feacd2', 'B': 'royalblue'},
            barmode="overlay")
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
               autosize=False,
               title={
              'text': "Распределение числа событий и даты маркетингового события",
               'y':0.98,
               'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'
                },
                xaxis_title="Дата",
                yaxis_title="Количество событий",
                font=dict(
                size=15),
                width=950,
                height=700,
                legend=dict(
                title='Группы пользователей',
                yanchor="bottom",
                y=0.81,
                xanchor="left",
                x=0.07
                )) 
fig.update_xaxes(showline=True,
              linewidth=2,
              linecolor='black',
              gridcolor='LightGrey',)

# добавим даты начала маркетиногового события Christmas&New Year Promo
fig.add_vline(x=market_events_new.query('name == "Christmas&New Year Promo"')['start_dt'].iloc[0], 
              line_dash = 'dash', 
              line_color = 'firebrick')
fig.add_vline(x=market_events_new.query('name == "Christmas&New Year Promo"')['finish_dt'].iloc[0], 
              line_dash = 'dash', 
              line_color = 'firebrick')


# В период проведения маркетингового события Christmas&New Year Promo на графике не замечено сильное изменение активности пользователей. Событие началось после предрождественной подготовки, пик активности пользователей пришелся на 21 декабря и после этого активность снижалась. По сути, сильного эффекта данное маркетиногвоое событие не оказало.

# - Построим простые продуктовые воронки для двух групп теста с учетом логической последовательности совершения событий

# Логическая цепочка действий такова: login -> product_page -> product_cart -> purchase.
# 
# В разрезе уникальных пользователей построим воронки:

# In[49]:


funnel_a = (new_data.query('group == "A"').pivot_table(index='event_name', values='user_id', aggfunc='nunique')
                    .sort_values(by='user_id', ascending=False).reset_index())


list = []

step = 1
for i in range(4):
    if i == 0:
        list.append(100 * funnel_a['user_id'][i] / funnel_a['user_id'][i]) 
    else:
        list.append(100 * funnel_a['user_id'][step] / funnel_a['user_id'][step - 1]) 
        step += 1
funnel_a['conv_step'] = list
funnel_a['conv_step'] = round(funnel_a['conv_step'], 2)
funnel_a = funnel_a.reindex([0, 1, 3, 2])
funnel_a


# Отметим, что событий purchase больше событий product_cart - 833 против 782.

# In[50]:


funnel_b = (new_data.query('group == "B"').pivot_table(index='event_name', values='user_id', aggfunc='nunique')
                    .sort_values(by='user_id', ascending=False).reset_index())


list = []

step = 1
for i in range(4):
    if i == 0:
        list.append(100 * funnel_b['user_id'][i] / funnel_b['user_id'][i]) 
    else:
        list.append(100 * funnel_b['user_id'][step] / funnel_b['user_id'][step - 1]) 
        step += 1
funnel_b['conv_step'] = list
funnel_b['conv_step'] = round(funnel_b['conv_step'], 2)
funnel_b = funnel_b.reindex([0, 1, 3, 2])
funnel_b


# Также отметим, что событий purchase больше событий product_cart - 249 против 244.

# In[51]:


fig = make_subplots(rows=1, cols=2,subplot_titles=("Воронка группы А", "Воронка группы В"))


fig.add_trace(
    go.Funnel(
    y = funnel_a['event_name'],
    x = funnel_a['user_id'],
    textposition = "inside",
    hoverinfo = "percent total",
    textinfo = "value+percent previous",
    marker = {"color": "#feacd2"}
),
    row=1, col=1
)

fig.add_trace(
    go.Funnel(
    y = funnel_b['event_name'],
    x = funnel_b['user_id'],
    textposition = "inside",
    hoverinfo = "percent total",
    textinfo = "value+percent previous",
    marker = {"color": "royalblue"}
),
    row=1, col=2
)

fig.update_layout(showlegend=False,height=600, width=900)
fig.show()


# Судя по всему, существует так называется "быстрая покупка", когда пользователь может приобрести товар, не пользуясь корзиной.
# 
# Конверсия переходов к событию product_page и purchase у группы В снизилась по сравнению с группой А. Конверсия в событие product_cart возрасла с 46 по 49%.
# 
# Ожидаемый эффект увеличения конверсии в группе В на 10 процентных пунктов, относительно конверсии в группе А не наблюдается ни у какого события.

# **Вывод:**
# 
# - В группе А в среднем на пользователя приходится около 3 событий, в группе В данный показатель чуть ниже - около 2.5-2.6 события.
# - Почти у половины пользователей группы А и В (44% и 46%) - Android. Типом устройства Mac пользуются не так активно - всего около 10% в обех группах. У iPhone 20% и 21% в группе А и В соответственно. В дальнейшем можно проанализировать, почему пользователей с Android в два раза больше пользователей iPhone, связано ли это с интерфейсом приложения, наличием багов и т.д.
# - В группе В событим много меньше по сравнению с группой А. Резкий подъем количества событий у группы А наблюдается с 14 декабря - можно предположить, что на это могли повлиять новогодние/рождественские праздники - люди совершают больше действий, покупок и т.д. До 21 декабря у групп с каждым днем все больше событий, после этого количество событий снижается.
# - В момент проведения теста попадает одно маркетинговое событие: Christmas&New Year Promo. В период проведения маркетингового события Christmas&New Year Promo на графике не замечено сильное изменение активности пользователей. Сильного эффекта данное маркетиногвоое событие не оказало.
# - Судя по всему, существует так называется "быстрая покупка", когда пользователь может приобрести товар, не пользуясь корзиной.Конверсия переходов к событию product_page и purchase у группы В снизилась по сравнению с группой А. Конверсия в событие product_cart возрасла с 46 по 49%.ьОжидаемый эффект увеличения конверсии в группе В на 10 процентных пунктов, относительно конверсии в группе А не наблюдается ни у какого события.

# ### Шаг 6. Оценка результатов A/B-тестирования:

# - Проверим статистическую разницу долей z-критерием.

# Запишем функцию для проведения статистического теста:

# In[52]:


def stat_analysis(group_one, group_two, group_one_all, group_two_all):
    
    alpha = 0.05 / 3  # критический уровень статистической значимости у четом поправки Бонферони

    successes  = np.array([group_one, group_two])
    trials = np.array([group_one_all, group_two_all])

    # пропорция в первой группе:
    p1 = successes[0] / trials[0]

    # пропорция во второй группе:
    p2 = successes[1] / trials[1]
    
    # пропорция в комбинированном датасете:
    p_combined = (successes[0] + successes[1]) / (trials[0] + trials[1])

    # разница пропорций в датасетах
    difference = p1 - p2 

    z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1 / trials[0] + 1 / trials[1]))

    # задаем стандартное нормальное распределение (среднее 0, ст.отклонение 1)
    distr = st.norm(0, 1)  

    p_value = (1 - distr.cdf(abs(z_value))) * 2

    print('p-значение: ', p_value)

    if p_value < alpha:
        print('Отвергаем нулевую гипотезу: между долями есть значимая разница')
    else:
        print(
            'Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными'
        )


# - Оценим статистическую разницу между группами перехода login -> product_page. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры страницы товара у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры страницы товара есть

# In[53]:


group_one = funnel_a.query('event_name == "product_page"')['user_id']
group_two = funnel_b.query('event_name == "product_page"')['user_id']
group_one_all = funnel_a.query('event_name == "login"')['user_id']
group_two_all = funnel_b.query('event_name == "login"')['user_id']

stat_analysis(group_one, group_two, group_one_all, group_two_all)


# - Оценим статистическую разницу между группами перехода product_page -> product_cart. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями перехода к корзине у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями перехода к корзине товара есть

# In[54]:


group_one = funnel_a.query('event_name == "product_cart"')['user_id']
group_two = funnel_b.query('event_name == "product_cart"')['user_id']
group_one_all = funnel_a.query('event_name == "product_page"')['user_id']
group_two_all = funnel_b.query('event_name == "product_page"')['user_id']

stat_analysis(group_one, group_two, group_one_all, group_two_all)


# - Оценим статистическую разницу между группами перехода purchase -> product_cart. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями к покупке у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями к покупке товара есть

# In[55]:


group_one = funnel_a.query('event_name == "product_cart"')['user_id']
group_two = funnel_b.query('event_name == "product_cart"')['user_id']
group_one_all = funnel_a.query('event_name == "purchase"')['user_id']
group_two_all = funnel_b.query('event_name == "purchase"')['user_id']

stat_analysis(group_one, group_two, group_one_all, group_two_all)


# **Вывод:**
# 
# - Гипотеза 1. Постановка гипотезы:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры страницы товара у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры страницы товара есть
# - Гипотеза 2. Постановка гипотезы:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями перехода к корзине у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями перехода к корзине товара есть
# - Гипотеза 3. Постановка гипотезы:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями к покупке у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями к покупке товара есть
#     
# В результате удалось выявить зависимости:
# - Гипотеза 1 не подтверждена: Отвергаем нулевую гипотезу, между долями есть значимая разница.
# - Гипотеза 2 подтверждена: Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными.    
# - Гипотеза 3 не подтверждена: Отвергаем нулевую гипотезу, между долями есть значимая разница.   

# ### Шаг 6. Общий вывод:
# 
# Был проведен анализ, чтобы проверить данные на соответствие ТЗ, провести исследовательский анализ, а также проанализировать результаты теста после введения улучшенной рекоментательной системы.
# 
# `Перед анализом проведена предобработка данных`:
# - Привели столбцы датасетов market_events (столбцы start_dt и finish_dt), users (столбец first_date) и ab_events(event_dt) к типу времени datetime
# - явные и неявные дупликаты не обнаружены
# - обнаружены пропуски (85,75%) в столбце details датасета ab_events. Пропуски оставляем как есть, так как заполнение столбца details происходит только при оплате.
# - в датасете users отфильрованы значения по дате регистрации в соответствии с ТЗ - с 7 по 21 декабря 2020 г.
# - в датасете ab_events события совершаются с 7 по 30 декабря 2020, что не соответствует ТЗ. По ТЗ события новых пользователей совершались в период с 7 декабря 2020 по 4 января 2021. Отсутствуют данные за 5 дней. Причинами отсутствия могут быть неверная выгрузка данных, намеренная остановка теста ввиду набора достаточного количества данных.
# - События распределены схожим образом по группам А и В, но в группе А регистраций происходит больше. Также пользователей больше в группе А.
# - Пересечений между группами А и В теста recommender_system_test нет выявлено.
# - Пересечения между группами двух тестов есть - принято решение сохранить пользователей.
# 
# 
# - Исключены неактивные пользователи из контрольной и тестовой группы. Пользователи распределены неравномерно по датасету. Необходимо понять, почему так много неактивных пользователей попали в контрольную группу тестирования.
# - Сохранены пользователи, которые не прожили 14 дней ввиду того, что бОльшая часть пользователей совершают события в пределах 14 дней.
# - Исключены события, которые совершены позднее 14 дней.
# - Тестовая группа не соответствует требованиям по размерам выборки. При такой выборке получить достоверный результат не получится.
# 
# 
# `В ходе исследования выявлены следующие закономерности:`
# - В группе А в среднем на пользователя приходится около 3 событий, в группе В данный показатель чуть ниже - около 2.5-2.6 события.
# - Почти у половины пользователей группы А и В (44% и 46%) - Android. Типом устройства Mac пользуются не так активно - всего около 10% в обех группах. У iPhone 20% и 21% в группе А и В соответственно.
# - В группе В событим много меньше по сравнению с группой А. Резкий подъем количества событий у группы А наблюдается с 14 декабря - можно предположить, что на это могли повлиять новогодние/рождественские праздники - люди совершают больше действий, покупок и т.д. До 21 декабря у групп с каждым днем все больше событий, после этого количество событий снижается.
# - В момент проведения теста попадает одно маркетинговое событие: Christmas&New Year Promo. В период проведения маркетингового события Christmas&New Year Promo на графике не замечено сильное изменение активности пользователей. Сильного эффекта данное маркетинговое событие не оказало.
# - Судя по всему, существует так называется "быстрая покупка", когда пользователь может приобрести товар, не пользуясь корзиной. Конверсия переходов к событию product_page и purchase у группы В снизилась по сравнению с группой А. Конверсия в событие product_cart возрасла с 46 по 49%. Ожидаемый эффект увеличения конверсии в группе В на 10 процентных пунктов, относительно конверсии в группе А не наблюдается ни у какого события.
# 
# `Перед проведением исследования были поставлены несколько гипотез:`
# - Гипотеза 1. Постановка гипотезы:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры страницы товара у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры страницы товара есть
# - Гипотеза 2. Постановка гипотезы:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями перехода к корзине у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями перехода к корзине товара есть
# - Гипотеза 3. Постановка гипотезы:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями к покупке у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями к покупке товара есть
#     
# В результате удалось выявить зависимости:
# - Гипотеза 1 не подтверждена: Отвергаем нулевую гипотезу, между долями есть значимая разница.
# - Гипотеза 2 подтверждена: Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными.    
# - Гипотеза 3 не подтверждена: Отвергаем нулевую гипотезу, между долями есть значимая разница.
# 
# 
# 
# 
# 
# **Рекомендации**
# - Необходимо сформировать достаточный размер выборки в тестовой группе для достоверного проведения тестирования, при условии, что пользователи тестовой группы совершают события.
# - По возможности, проводить А/В-тестирование в периодах, не пересекающихся с праздниками, маркетинговыми событиями, чтобы исключить их возможное влияние на активность пользователей.
# - Провести повторное А/В-тестирование, учитывая вышеуказанные рекомендации.
# 
# 
# Также в дальнейшем можно проанализировать, почему пользователей с Android в два раза больше пользователей iPhone, связано ли это с интерфейсом приложения, наличием багов и т.д.
