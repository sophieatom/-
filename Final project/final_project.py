#!/usr/bin/env python
# coding: utf-8

# ## Проект «Анализ поведения пользователей сервиса "Ненужные вещи"»

# ### Описание проекта:
# 
# Команде сервиса «Ненужные вещи» требуется помощь в повышении вовлеченности пользователей - нужно проанализировать различные сценарии поведения пользователей с целью выявления интересных особенностей. В сервисе пользователи продают свои ненужные вещи, размещая их на доске объявлений.
# 
# В датасетах имеются данные о событиях, совершенных в мобильном приложении "Ненужные вещи", а также содержатся данные пользователей, впервые совершивших действия в приложении после 7 октября 2019 года.

# ### Описание данных:
#     
# Структура датасета mobile_sources.csv:     
# - `userId` — идентификатор пользователя,
# - `source` — источник, с которого пользователь установил приложение.
# 
# Структура датасета mobile_dataset.csv :
# - `event.time` — время совершения,
# - `user.id` — идентификатор пользователя,
# - `event.name` — действие пользователя.
# - Виды действий:
#   - `advert_open` — открыл карточки объявления,
#   - `photos_show` — просмотрел фотографии в объявлении,
#   - `tips_show` — увидел рекомендованные объявления,
#   - `tips_click` — кликнул по рекомендованному объявлению,
#   - `contacts_show` и `show_contacts` — посмотрел номер телефона,
#   - `contacts_call` — позвонил по номеру из объявления,
#   - `map` — открыл карту объявлений,
#   - `search_1 — search_7` — разные действия, связанные с поиском по сайту,
#   - `favorites_add` — добавил объявление в избранное.    

# ### Цели исследования
# - Поиск сценариев/"паттерны" для повышения вовлеченности;
# 
# - Получение на основе поведения пользователей гипотезы о том, как можно было бы улучшить приложение с точки зрения пользовательского опыта.

# ### Шаг 1. Открыть файл с данными и изучить общую информацию
# 
# - Импортировать необходимые библиотеки;
# - Открыть файл с данными;
# - Вывести основные характеристики датафрейма;

# ### Шаг 2. Предобработка данных
# - Исследовать пропущенные значения;
# - Исследовать соответствие типов;
# - Исследовать дубликаты;
# - Проверить корректность наименований колонок;
# - Переименовать колонки, если это необходимо;
# - Удалить дубликаты;
# - Привести типы;
# - Заменить пропущенные значения, если это возможно;
# - Объединить действия contacts_show и show_contacts.

# ### Шаг 3. Исследовательский анализ
# 1. События
# - Посчитать, сколько всего событий и пользователей в данных;
# - Определить, сколько в среднем событий приходится на пользователя;
# - Проверить, за какой период имеются данные. Найти максимальную и минимальную дату. Проверить период данных на соответствие с ТЗ; 
# - Определить, с какого момента данные полные и , в случае необходимости, отбросить более старые;
# - Посмотреть, какие события есть в данных, как часто они встречаются. Отсортировать события по частоте.
# 
# 2. Пользовательские сессии:
# - Определить тайм-аут - время, через которое сессия считается завершенной;
# - Сформировать сессии пользователей;
# - Определить основные характеристики: среднее число сессий на пользователя, средняя и медианная продолжительность сессии и т.д.

# ### Шаг 4. Основные вопросы исследования
# - Проанализировать связь целевого события - просмотра контактов — и других действий пользователей:
#  - В разрезе сессий отобрать сценарии\паттерны, которые приводят к просмотру контактов;
#  - Посчитать конверсию в целевое событие в разрезе уникальных пользователей;
#  - Построить воронки по основным сценариям в разрезе уникальных пользователей;
#  - Определить, на каком шаге теряется больше всего пользователей;
#  - Ответить на вопрос, как различается длительность сессий, в которых встречаются события: 
#    - contacts_show с photos_show
#    - contacts_show без photos_show
# - Оценить, какие действия чаще совершают те пользователи, которые просматривают контакты:
#  - Рассчитать относительную частоту событий в разрезе двух групп пользователей:
#    - группа пользователей, которые смотрели контакты contacts_show
#    - группа пользователей, которые не смотрели контакты

# ### Шаг 5. Проверка гипотез
# - Одни пользователи совершают действия `tips_show и tips_click` , другие — только `tips_show`. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры есть
# - Пользователи установливают приложения, переходя из трех источников - Yandex, Google и other. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у пользователей из источников Yandex и Google нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры контактов у пользователей из источников Yandex и Google есть.
# - Одни пользователи совершают действия `favorites_add`, другие — нет. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры есть

# ### Шаг 6. Общий вывод и рекомендации
# - Написать выводы о каждом шаге исследования;
# - На основе результатов дать рекомендации, которые могут помочь в повышении вовлеченности пользователей в приложение.

# ### 1. Загрузка данных

# Импортируем необходимые библиотеки и загружаем данные из файлов в датафреймы:

# In[1]:


import pandas as pd
import datetime as dt

import numpy as np
from scipy import stats as st
import math as mth

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from plotly import graph_objects as go
import warnings

import plotly.io as pio
pio.renderers.default='notebook'

warnings.filterwarnings("ignore")


# In[2]:


try:
    sources = pd.read_csv('/Desktop/учеба яндекс/final_project/mobile_sources.csv')
    data = pd.read_csv('/Desktop/учеба яндекс/final_project/mobile_datase.csv')
except:
    sources = pd.read_csv('https://code.s3.yandex.net/datasets/mobile_sources.csv')
    data = pd.read_csv('https://code.s3.yandex.net/datasets/mobile_dataset.csv')


# Рассмотрим полученные данные из датафреймов:

# In[3]:


# Выведем 5 строк датафреймов
display(sources.head(10).style.set_caption('Данные пользователей'))
display(data.head(10).style.set_caption('Данные о событиях'))


# In[4]:


# Выведем основные характеристики датафрейма (типы столбцов, пропущенные значения)
print('Данные о событиях')
data.info()


# In[5]:


# Выведем основные характеристики датафрейма (типы столбцов, пропущенные значения)
print('Данные пользователей')
sources.info()


# In[6]:


# Проверим, присутствуют ли пропуски в датафрейме:
print(f'Количество пропусков в датафрейме с данными о событиях - {data.isna().sum()}') 


# In[7]:


# Проверим, присутствуют ли пропуски в датафрейме:
print(f'Количество пропусков в датафрейме с данными о пользователях - {sources.isna().sum()}') 


# **Вывод:**
# 
# - всего представлены два датасета - data и sources;
# - в датасетах data и sources 74197 и 4293 строк соотвественно;
# - пропуски не обнаружены;
# - отметим, что в датасете столбец `event.time` относится к типу данных object.

# ### 2. Предобработка данных

# - приведем наименования колонок к единому типу:

# In[8]:


# Переименуем столбцы датафрейма sources и data:
sources = sources.rename(columns={'userId':'user_id'})
data = data.rename(columns={
                             'event.time':'event_time',
                             'event.name':'event_name',
                             'user.id':'user_id'
                            })


# - приведем типы данных там, где необходимо:

# Тип столбца `event_time` небходимо привести к типу datetime:

# In[9]:


data['event_time'] = pd.to_datetime(data['event_time'], format='%Y-%m-%d %H:%M:%S')


# - исследуем данные на наличие явных и неявных дупликатов:

# In[10]:


# Проверим, присутствуют ли дубликаты в датафреймах source и data:
print(f'Всего дупликатов в датафрейме - {sources.duplicated().sum()}, ' 
      f'то есть {round(100 * sources.duplicated().sum() / len(sources), 2)} процентов от всех данных')
print(f'Всего дупликатов в датафрейме - {data.duplicated().sum()}, ' 
      f'то есть {round(100 * data.duplicated().sum() / len(data), 2)} процентов от всех данных')


# Рассмотрим уникальные значения и сколько раз они встречаются в столбцах `event_name и source`:

# In[11]:


print(data['event_name'].value_counts())
print()
print(sources['source'].value_counts())


# In[12]:


print('Период события contacts_show:', 
      data.query("event_name == 'contacts_show'").event_time.dt.date.min(), '-', 
      data.query("event_name == 'contacts_show'").event_time.dt.date.max())
print('Период события show_contacts:', 
      data.query("event_name == 'show_contacts'").event_time.dt.date.min(), '-', 
      data.query("event_name == 'show_contacts'").event_time.dt.date.max())


# Судя по всему, действия *contacts_show* и *show_contacts* в столбце event_name не пересекаются и обозначают одно действие - просмотр контакта. Так как событие contacts_show происходило чаще, приведем эти действия к одному названию - contacts_show.

# In[13]:


data.loc[(data['event_name'] == 'show_contacts') | (data['event_name'] == 'contacts_show'), 'event_name'] = 'contacts_show'


# **Вывод:**
# 
# - наименования столбцов приведены в соответствии с синтаксисом;
# - тип данных столбца event_name датасета data изменен на тип данных datetime;
# - явные и неявные дупликаты не обнаружены;
# - объединены события show_contacts и contacts_show.

# ### 3. Исследовательский анализ

# #### События

# - Сколько всего событий в данных?

# In[14]:


print(f'Всего событий в данных - {data["event_name"].count()}')


# Рассмотрим, какие именно события имеются в данных:

# In[15]:


print('Уникальные значения в столбце событий -', ", ".join(map(str, data['event_name'].unique())), sep='\n')


# In[16]:


# проверим, как распределены события:
data.pivot_table(index='event_name', values='event_time', aggfunc='count').sort_values(by='event_time', ascending=False)


# In[17]:


px.bar(data.pivot_table(index='event_name', values='event_time', aggfunc='count').sort_values(by='event_time'),
       x='event_time', 
       color_discrete_sequence=["royalblue"],
       text_auto=True) \
  .update_layout(plot_bgcolor='rgba(0,0,0,0)') \
  .update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey') \
  .update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=950,
                 height=800,
                 title = {
                            'text': "Распределение событий",
                                'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="Количество",
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )


# Можно отметить, что событие `tips_show` (показ рекомендованных объявлений) происходило чаще всего - 40 055 раз. Также в топ-3 входят события `photos_show` (просмотр фотографий в объявлении) и `advert_open` (открытие карточки объявления) - 10 012 и 6 164 соответственно. Меньше всего пользователи совершали такие связанные с поиском события, как `search_6`, `search_2` и `search_7` - менее 500 раз.

# - Сколько всего пользователей в данных?

# In[18]:


print(f'Всего уникальных пользователей в данных - {data["user_id"].nunique()}')


# - Сколько в среднем событий приходится на пользователя?

# In[19]:


# посчитаем, сколько событий приходится на каждого пользователя и выдедем основную информацию
data.pivot_table(index='user_id', values='event_name', aggfunc='count').describe()


# In[20]:


fig = px.box(data.pivot_table(index='user_id', values='event_name', aggfunc='count'), points="all", template="seaborn") .update_layout(boxmode = "overlay") .update_xaxes(showline=True, 
                linewidth=2, 
                showticklabels=False,
                linecolor='black', 
                gridcolor='LightGrey') \
.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=950,
                 height=800,
                 title = {
                            'text': "Диаграмма размаха",
                                'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="",
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )
fig.show()


# В среднем на пользователя приходится около 17 событий. Минимальное и максимальное количество событий на пользователя - 1 и 478 соответственно. Получился сильный разброс значений. Есть пользователи, которые совершили только 1 событие (возможно, только увидели рекомендации и больше не совершали никаких действий) и пользователи, которые регулярно пользуются приложением. Так как величиная среднего значения не показательна при выбросах, будем ориентироваться на `значение медианы, которое составляет 9 событий`.

# - Рассмотрим источники, с которых пользователь установил приложение

# Сперва объединим датасеты data и source по полю user_id:

# In[21]:


all_data = data.merge(sources, on='user_id', how='left')
all_data.head()


# В разрезе уникальных пользователей рассмотрим, из какого источника чаще переходили пользователи:

# In[22]:


px.bar(all_data.pivot_table(index='source', values='user_id', aggfunc='nunique').sort_values(by='user_id', ascending=False),
       y='user_id', 
       color_discrete_sequence=["royalblue"],
       text_auto=True) \
  .update_layout(plot_bgcolor='rgba(0,0,0,0)') \
  .update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey') \
  .update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=950,
                 height=600,
                 title = {
                            'text': "Количество уникальных пользователей в зависимости от источника",
                                'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="Источник",
                 yaxis_title="Кол-во пользователей",
                 font=dict(
                 size=15),
                    )


# Yandex лидирует по привлечению пользователей - 1934 уникальных пользователя, у источников other и google - 1230 и 1129. Найдем долю уникальных пользователей, который совершали целевое действие contacts_show к общему числу уникальных пользоватеелй в разрезе источников:

# In[23]:


share = all_data.query('event_name == "contacts_show"')                 .pivot_table(index='source', values='user_id', aggfunc='nunique')                 .reset_index()                 .merge(all_data.pivot_table(index='source', values='user_id', aggfunc='nunique'), on='source')                 .rename(columns={'user_id_x':'target_users', 'user_id_y':'all_users'})
share['share'] = round(100 * share['target_users'] / share['all_users'] , 2)
share


# Пользователи чаще всего приходят из источника yandex, но доля пользователей (24.72%), которые совершили целевое действие, почти такая же, что и источника из google (24.36%).

# - Проверим, за какой период имеются данные

# In[24]:


# узнаем миниамальную и максимальную даты в имеющихся данных
print(f'Начало периода - {min(data["event_time"].dt.date)}, конец периода - {max(data["event_time"].dt.date)}')


# In[25]:


fig = px.histogram(data, 
                   x=data["event_time"].dt.date, 
                   nbins=30, 
                   color_discrete_sequence=["royalblue"],
                   text_auto=True
                  )
fig.update_layout(bargap=0.2)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=950,
                 height=600,
                 title = {
                            'text': "Распределение событий по датам",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="Дата",
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )
fig.update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey')
fig.update_traces(textangle=-90)
fig.show()


# In[26]:


fig = px.histogram(data, 
                   x=data["event_time"].dt.day_name(), 
                   nbins=30, 
                   color_discrete_sequence=["royalblue"],
                   text_auto=True
                  )
fig.update_layout(bargap=0.2)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=950,
                 height=600,
                 title = {
                            'text': "Распределение событий по дням недели",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="Дата",
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )
fig.update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey')
fig.update_traces(textangle=-90)
fig.show()


# Имеются данные с 7 октября 2019 по 3 ноября 2019, что соответствует ТЗ. Судя по полученной гистограмме, изменение числа событий носит скачкообразный характер. В разбивке по дням недели наблюдается снижение числа событий с понедельника по субботу - с 11 671 до 9 154, затем в воскресенье количество событий увеличвается до 10 501.

# - Посчитаем, сколько пользователей совершали каждое из событий, отсортируем события по числу пользователей и посчитаем долю пользователей, которые хоть раз совершали событие.

# Как мы выясняли, событие `tips_show` (показ рекомендованных объявлений) происходило чаще всего - 40 055 раз. Также в топ-3 входят события `photos_show` (просмотр фотографий в объявлении) и `advert_open` (открытие карточки объявления) - 10 012 и 6 164 соответственно.

# In[27]:


# Отсортируем события по числу уникальных пользователей и найдем долю каждого события от общего числа
events = data.pivot_table(index='event_name', values='user_id', aggfunc='nunique')              .query('user_id > 0')              .sort_values(by='user_id', ascending=False)              .reset_index()
events['share'] = round(100 * events['user_id'] / data['user_id'].nunique(), 2)
events


# In[28]:


fig = px.bar(events, 
                   x=events['share'], y=events['event_name'],
                   color_discrete_sequence=["royalblue"],
                   text_auto=True
                  )
fig.update_layout(bargap=0.2)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 yaxis = {"categoryorder":"total ascending"},
                 width=950,
                 height=600,
                 title = {
                            'text': "Доля событий в разрезе уникальных пользователей",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="% событий",
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )
fig.update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey')
fig.update_traces(textangle=0)
fig.show()


# В разрезе уникальных пользователей лидирует событие tips_show, а также map и photos_show. Пока что намечается последовательность следующая последовательность событий: tips_show-map-photos_show-contact_show/увидел рекомендованные объявления-открыл карту объявлений-просмотрел фотографии в объявлении-посмотрел номер телефона.

# #### Пользовательские сессии:

# - Определим тайм-аут - время, через которое сессия считается завершенной;

# Будем исходить из того, что тайм-аут сессии по умолчанию установлены в таких системах, как Google Analytics (GA) и Яндекс.Метрика продолжительностью 30 минут:

# - Сформируем сессии пользователей;

# In[29]:


# создадим новый датасет
data_new = all_data
# перед выделением сессий отсортируем датасет по user_id и event_time
data_new = data_new.sort_values(['user_id', 'event_time'])
# установим 
diff = (data.groupby('user_id')['event_time'].diff() > pd.Timedelta('30Min')).cumsum()
data_new['session_id'] = data_new.groupby(['user_id', diff], sort=False).ngroup() + 1


# In[30]:


data_new.tail()


# - Определить основные характеристики: среднее число сессий на пользователя, средняя и медианная продолжительность сессии и т.д.

# Полезно изучить, сколько в среднем сессий приходится на одного пользователя, например, за месяц. Это хороший показатель регулярности использования приложения

# In[31]:


# находим количество сессий и количество пользователей
data_new['month'] = data_new['event_time'].dt.month

sessions_per_user = data_new.groupby(['month']).agg(
    {'session_id': ['count', 'nunique']}
)

# переименовываем колонки
sessions_per_user.columns = ['n_sessions', 'n_users']
# делим число сессий на количество пользователей
sessions_per_user['sessions_per_user'] = (
    sessions_per_user['n_sessions'] / sessions_per_user['n_users']
)

print(sessions_per_user) 


# В октябре на пользователя приходилось 2-3 сессии. За ноябрь данных мало, чтобы делать выводы.
# 
# Средняя продолжительность сессии, или ASL, показывает, сколько в среднем длится сессия пользователя. 

# In[32]:


# считаем ASL
sessions = data_new.groupby('session_id').agg({'event_time':['min', 'max']})
sessions.columns = ['min', 'max']
sessions['diff'] = (sessions['max'] - sessions['min']).dt.seconds

print('Средняя продолжительность сессии составляет', round(sessions['diff'].mean(), 2), 'сек.') 
# рассчитаем медианную длительность сессий
print('Средняя продолжительность сессии составляет', round(sessions['diff'].median(), 2), 'сек.') 
print('Всего сессий:', len(sessions))


# In[33]:


fig = px.histogram(sessions['diff'], 
                   nbins=50, 
                   color_discrete_sequence=["royalblue"]
                  )
fig.update_layout(bargap=0.2)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=950,
                  showlegend=False,
                 height=600,
                 title = {
                            'text': "Гистограмма средней продолжительности сессий",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="Время, сек.",
                 yaxis_title="Частота",
                 font=dict(
                 size=15),
                    )
fig.update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey')
fig.update_traces(textangle=0)
fig.show()


# Секунды, проведённые пользователями в приложении, имеют экспоненциальное распределение с пиком в районе нуля. Возможно, в сервисе сессии заканчиваются техническими ошибками.
# Все «ошибочные» сессии на графике попадают в корзину, которая ближе всего к нулю, а успешные — «размазываются» в длинный хвост. На самом его кончике собираются аномально длинные сессии пользователей, которым очень понравился продукт. Можно отметить, что присутствует аномальное значение более 8000 сек (более 2 часов).

# Избавимся от сессий, в которых продолжительность сессий равна нулю:

# In[34]:


sessions = data_new.groupby('session_id').agg({'event_time':['min', 'max']})
sessions.columns = ['min', 'max']
sessions['diff'] = (sessions['max'] - sessions['min']).dt.seconds
sessions = sessions.reset_index()
sessions_id = sessions.query('diff == 0')['session_id']


# В датасете со сформированными сессиями "отфильтруем" сессии, продолжительность которых нулевая:

# In[35]:


data_new = data_new.query('session_id not in @sessions_id')


# Посмотрим, как изменится показатель средней и медианной продолжительности сессий и гистограмма:

# In[36]:


# считаем ASL
sessions = data_new.groupby('session_id').agg({'event_time':['min', 'max']})
sessions.columns = ['min', 'max']
sessions['diff'] = (sessions['max'] - sessions['min']).dt.seconds

print('Средняя продолжительность сессии составляет', round(sessions['diff'].mean(), 2), 'сек.') 
# рассчитаем медианную длительность сессий
print('Средняя продолжительность сессии составляет', round(sessions['diff'].median(), 2), 'сек.') 
print('Всего сессий:', len(sessions))


# In[37]:


fig = px.histogram(sessions['diff'], 
                   nbins=50, 
                   color_discrete_sequence=["royalblue"]
                  )
fig.update_layout(bargap=0.2)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 showlegend=False,
                 width=950,
                 height=600,
                 title = {
                            'text': "Гистограмма средней продолжительности сессий",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="Время, сек.",
                 yaxis_title="Частота",
                 font=dict(
                 size=15),
                    )
fig.update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey')
fig.update_traces(textangle=0)
fig.show()


# После удаления части сессий (более 11 тысяч), показатели средней и медианной продолжительности сессий изменились с 121,44 до 215,7 сек. и с 17 до 128 сек. 
# 
# Также важно понять, почему более половины сессий - "ошибочны".

# **Выводы:**
# 
# <u>События:<u>
# - Можно отметить, что событие `tips_show` (показ рекомендованных объявлений) происходило чаще всего - 40 055 раз. Также в топ-3 входят события `photos_show` (просмотр фотографий в объявлении) и `advert_open` (открытие карточки объявления) - 10 012 и 6 164 соответственно. Меньше всего пользователи совершали такие связанные с поиском события, как `search_6`, `search_2` и `search_7` - менее 500 раз.
#     
#     
# - В среднем на пользователя приходится около 17 событий. Минимальное и максимальное количество событий на пользователя - 1 и 478 соответственно. Получился сильный разброс значений. Есть пользователи, которые совершили только 1 событие (возможно, только увидели рекомендации и больше не совершали никаких действий) и пользователи, которые регулярно пользуются приложением. Так как величиная среднего значения не показательна при выбросах, будем ориентироваться на `значение медианы, которое составляет 9 событий`.
# 
#     
# - `Yandex` лидирует по привлечению пользователей - 1934 уникальных пользователя, у источников `other и google` - 1230 и 1129. Пользователи чаще всего приходят из источника yandex, но доля пользователей (24.72%), которые совершили целевое действие, почти такая же, что и источника из google (24.36%).
# 
#     
# - Имеются данные с 7 октября 2019 по 3 ноября 2019, что соответствует ТЗ. Судя по полученной гистограмме, изменение числа событий носит скачкообразный характер. В разбивке по дням недели наблюдается снижение числа событий с понедельника по субботу - с 11 671 до 9 154, затем в воскресенье количество событий увеличвается до 10 501.
# 
#     
# - В разрезе уникальных пользователей лидирует событие `tips_show`, а также `map` и `photos_show`. Пока что намечается  следующая последовательность событий: tips_show-map-photos_show-contact_show/увидел рекомендованные объявления-открыл карту объявлений-просмотрел фотографии в объявлении-посмотрел номер телефона.
# 
# <u>Сессии:<u>
# - В октябре на пользователя приходилось 2-3 сессии.
# - После удаления части сессий (более 11 тысяч), показатели средней и медианной продолжительности сессий изменились с 121,44 до 215,7 сек. и с 17 до 128 сек. Также важно понять, почему более половины сессий - "ошибочны".

# ### Шаг 4. Основные вопросы исследования
# - Проанализируем связь целевого события - просмотра контактов — и других действий пользователей. В разрезе сессий отберем сценарии\паттерны, которые приводят к просмотру контактов и определить, на каком шаге теряется больше всего пользователей.

# In[38]:


patterns = data_new.groupby('session_id').agg({'event_name':'unique'}).reset_index()
patterns['event_name'] = patterns['event_name'].apply(', '.join)
patterns.head()


# Отберем сессии, в которых присутствует событие contacts_show:

# In[39]:


list = []

count = 0
for i in patterns['event_name']:
    if 'contacts_show' in i:
        list.append(patterns.iloc[count])
    count += 1     
dframe = pd.DataFrame(list) 
dframe.head()


# In[40]:


dframe.pivot_table(index='event_name', values='session_id', aggfunc='count') .reset_index() .sort_values(by='session_id', ascending=False) .head(15)


# Отберем те последовательности действий, которые заканчиваются просмотром контактов (целевое действие), а также в которых больше хотя бы 2 событий. Получились такие сценарии:
# 
# **Сценарий 1:** `tips_show-map-contact_show`/увидел рекомендованные объявления-открыл карту объявлений-посмотрел номер телефона.

# In[41]:


tips = data.query('event_name == "tips_show"')['user_id'].unique()
tips_rows = data.query('event_name == "tips_show"')['user_id'].unique().shape[0]

maps = data.query('event_name == "map"')['user_id'].unique()
maps_rows = len(set(tips).intersection(maps))

contacts = data.query('event_name == "contacts_show"')['user_id'].unique()
contacts_rows = len(set(tips).intersection(maps).intersection(contacts))


funnel_one = pd.DataFrame(data={'event': ['tips_show', 'map', 'contacts_show'], 
                                'count': [tips_rows, maps_rows, contacts_rows],
                                })
list = []

step = 1
for i in range(3):
    if i == 0:
        list.append(100 * funnel_one['count'][i] / funnel_one['count'][i]) 
    else:
        list.append(100 * funnel_one['count'][step] / funnel_one['count'][step - 1]) 
        step += 1
funnel_one['conv_step'] = list
funnel_one['conv_step'] = round(funnel_one['conv_step'], 2)
funnel_one


# In[42]:


# отобразим воронку событий

fig = go.Figure(go.Funnel(
    y = funnel_one['event'],
    x = funnel_one['count'],
    textinfo = "value+percent initial",
    hoverinfo = "percent total"))

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=800,
                 height=600,
                 title = {
                            'text': "Воронка событий",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )


fig.show()


# С рекомендованных объявлений менее половины пользователей переходят к карте объявлений и только 10% пользователей совершают целевое действие. Низкая конверсия на шаге просмотр карты объявлений-просмотр контактов может быть связан с технической ошибкой.

# Очень часто встречается последовательность событий photos_show, contacts_show.
# 
# **Сценарий 2:** search_1, photos_show, contacts_show/поиском по сайту, просмотрел фотографии в объявлении, посмотрел номер телефона

# In[43]:


searchs = data.query('event_name == "search_1"')['user_id'].unique()
searchs_rows = data.query('event_name == "search_1"')['user_id'].unique().shape[0]

photos = data.query('event_name == "photos_show"')['user_id'].unique()
photos_rows = len(set(photos).intersection(searchs))

contacts = data.query('event_name == "contacts_show"')['user_id'].unique()
contacts_rows = len(set(photos).intersection(searchs).intersection(contacts))


# In[44]:


funnel_two = pd.DataFrame(data={'event': ['searchs', 'photos', 'contacts_show'], 
                                'count': [searchs_rows, photos_rows, contacts_rows],
                                })
list = []

step = 1
for i in range(3):
    if i == 0:
        list.append(100 * funnel_two['count'][i] / funnel_two['count'][i]) 
    else:
        list.append(100 * funnel_two['count'][step] / funnel_two['count'][step - 1]) 
        step += 1
funnel_two['conv_step'] = list
funnel_two['conv_step'] = round(funnel_two['conv_step'], 2)
funnel_two


# In[45]:


# отобразим воронку событий

fig = go.Figure(go.Funnel(
    y = funnel_two['event'],
    x = funnel_two['count'],
    textinfo = "value+percent initial",
    hoverinfo = "percent total"))

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=800,
                 height=600,
                 title = {
                            'text': "Воронка событий",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )


fig.show()


# При переходе c поиска на просмотр фотографии в объявлении конверсия составляет 82%, что является хорошим показателем. А вот конверсия в целевое события с просмотра фотографии мала - всего 24%. Это может быть связано с неудобным интерфейсом приложения/качеством фотографий и т.д.

# **Сценарий 3:** tips_show, advert_open, contacts_show/ увидел рекомендованные объявления-открыл карточки объявления-посмотрел номер телефона

# In[46]:


tips = data.query('event_name == "tips_show"')['user_id'].unique()
tips_rows = data.query('event_name == "tips_show"')['user_id'].unique().shape[0]

adverts = data.query('event_name == "advert_open"')['user_id'].unique()
adverts_rows = len(set(tips).intersection(adverts))

contacts = data.query('event_name == "contacts_show"')['user_id'].unique()
contacts_rows = len(set(tips).intersection(adverts).intersection(contacts))


# In[47]:


funnel_three = pd.DataFrame(data={'event': ['tips', 'adverts', 'contacts_show'], 
                                'count': [tips_rows, adverts_rows, contacts_rows],
                                })
list = []

step = 1
for i in range(3):
    if i == 0:
        list.append(100 * funnel_three['count'][i] / funnel_three['count'][i]) 
    else:
        list.append(100 * funnel_three['count'][step] / funnel_three['count'][step - 1]) 
        step += 1
funnel_three['conv_step'] = list
funnel_three['conv_step'] = round(funnel_three['conv_step'], 2)
funnel_three


# In[48]:


# отобразим воронку событий

fig = go.Figure(go.Funnel(
    y = funnel_three['event'],
    x = funnel_three['count'],
    textinfo = "value+percent initial",
    hoverinfo = "percent total"))

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 width=800,
                 height=600,
                 title = {
                            'text': "Воронка событий",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )


fig.show()


# Только около пятой части пользователей открывают карточки объявления после просмотра рекомендованных объявлений и всего 3% просматривают контакты. Возможно, стоит проработать рекомендации, которые получают пользователи и понять их алгоритм. 
# 
# Также отметим, что сценарий tips_show, tips_click, contacts_show случается не так часто, скорее всего, рекомендации не работают эффективно.

# - Посчитаем конверсию в целевое событие в разрезе уникальных пользователей

# In[49]:


# посчитаем количество уникальных пользователей по событиям
data.pivot_table(index='event_name', values='user_id', aggfunc='nunique')     .sort_values(by='user_id', ascending=False)     .reset_index()
    


# In[50]:


# запишем в переменную conv идентификаторы пользователей, которые совершали целевое событие
conv = data.query('event_name == "contacts_show"')['user_id'].unique()


# In[51]:


# отфилтруем данные по идентфиикаторам пользователей в переменной conv
data.query('user_id in @conv').pivot_table(index='event_name', values='user_id', aggfunc='nunique')     .sort_values(by='user_id', ascending=False).reset_index()


# In[52]:


# объединим полученные данные по полю event_name
data_conv = data.pivot_table(index='event_name', values='user_id', aggfunc='nunique')                 .sort_values(by='user_id', ascending=False)                 .reset_index()                 .merge(data.query('user_id in @conv').pivot_table(index='event_name', values='user_id', aggfunc='nunique')                            .sort_values(by='user_id', ascending=False).reset_index(), on='event_name')                 .rename(columns={'user_id_x':'users', 'user_id_y':'conv_users'})
data_conv['conv'] = round(100 * data_conv['conv_users'] / data_conv['users'], 2)


# In[53]:


fig = px.bar(data_conv.sort_values(by='conv'),
             x=data_conv['conv'], 
             y=data_conv['event_name'],
             color_discrete_sequence=["royalblue"],
             text_auto=True
                  )
fig.update_layout(bargap=0.2)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                 autosize=False,
                 yaxis = {"categoryorder":"total ascending"},
                 width=950,
                 height=600,
                 title = {
                            'text': "Конверсия событий в разрезе уникальных пользователей",
                            'y':0.98,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                 xaxis_title="% событий",
                 yaxis_title="События",
                 font=dict(
                 size=15),
                    )
fig.update_xaxes(showline=True, 
                linewidth=2, 
                linecolor='black', 
                gridcolor='LightGrey')
fig.update_traces(textangle=0)
fig.show()


# Хороший показатель конверсии наблюдается у события favorites_add (добавил объявление в избранное). Значит, после добавления объявления в избранное, более трети пользователей совершают целевое сбытие. Несмотря на то, что событие tips_show попадается пользователем чаще всего, показатель конверсии достаточно низкий - 18,42% 

# - Ответим на вопрос, как различается длительность сессий, в которых встречаются события: 
#    - show_contacts с photos_show
#    - show_contacts без photos_show

# Оставим сессии, в которых есть события `show_contacts и photos_show`:

# In[54]:


list = []

count = 0
for i in patterns['event_name']:
    if 'contacts_show' in i and 'photos_show' in i:
        list.append(patterns.iloc[count])
    count += 1     
dframe = pd.DataFrame(list) 
dframe.head()


# Посчитаем среднюю и медианную продолжительность сессий:

# In[55]:


# считаем ASL
sessions = data_new.query('session_id in @dframe.session_id').groupby('session_id').agg({'event_time':['min', 'max']})
sessions.columns = ['min', 'max']
sessions['diff'] = (sessions['max'] - sessions['min']).dt.seconds

print('Средняя продолжительность сессии составляет', round(sessions['diff'].mean(), 2), 'сек.') 
# рассчитаем медианную длительность сессий
print('Средняя продолжительность сессии составляет', round(sessions['diff'].median(), 2), 'сек.') 
print('Всего сессий:', len(sessions))


# Теперь отберем сессии, в которых есть целевое событие и нет события photos_show:

# In[56]:


list = []

count = 0
for i in patterns['event_name']:
    if 'contacts_show' in i and 'photos_show' not in i:
        list.append(patterns.iloc[count])
    count += 1     
dframe = pd.DataFrame(list) 
dframe.head()


# In[57]:


# считаем ASL
sessions = data_new.query('session_id in @dframe.session_id').groupby('session_id').agg({'event_time':['min', 'max']})
sessions.columns = ['min', 'max']
sessions['diff'] = (sessions['max'] - sessions['min']).dt.seconds

print('Средняя продолжительность сессии составляет', round(sessions['diff'].mean(), 2), 'сек.') 
# рассчитаем медианную длительность сессий
print('Средняя продолжительность сессии составляет', round(sessions['diff'].median(), 2), 'сек.') 
print('Всего сессий:', len(sessions))


# Сессий без события photos_show оказалось больше (1801 против 302), но при этом продолжительность сессий с целевым событием и событием photos_show оказалась длительнее.

# - Оценим, какие действия чаще совершают те пользователи, которые просматривают контакты, рассчитаем относительную частоту событий в разрезе двух групп пользователей:
#  - группа пользователей, которые смотрели контакты contacts_show
#  - группа пользователей, которые не смотрели контакты

# Отберем уникальных пользователей с целевым событием:

# In[58]:


users_contact_show = data_new.query('event_name == "contacts_show"')['user_id'].unique()


# Посчитаем количество событий, совершенных пользователями с целевым событием и пользователями без:

# In[59]:


# Посчитаем количество событий, совершенных пользователями с целевым событием:
data_new.query('user_id in @users_contact_show').pivot_table(index='event_name', values='event_time', aggfunc='count') .reset_index() .sort_values(by='event_time', ascending=False) .head()


# In[60]:


# Посчитаем количество событий, совершенных пользователями с целевым событием:
data_new.query('user_id not in @users_contact_show').pivot_table(index='event_name', values='event_time', aggfunc='count') .reset_index() .sort_values(by='event_time', ascending=False) .head()


# Объединим две таблицы:

# In[61]:


groups = data_new.query('user_id in @users_contact_show')                  .pivot_table(index='event_name', values='event_time', aggfunc='count')                  .reset_index()                  .sort_values(by='event_time', ascending=False)                  .merge(data_new.query('user_id not in @users_contact_show')                                 .pivot_table(index='event_name', values='event_time', aggfunc='count')                                 .reset_index()                                 .sort_values(by='event_time', ascending=False), how='left', on='event_name')                  .rename(columns={'event_time_x':'see_contacts', 'event_time_y':'not_see_contacts'})                  .fillna(0)
groups['share_see_contacts'] = round(groups['see_contacts'] / sum(groups['see_contacts']), 2)
groups['share_not_see_contacts'] = round(groups['not_see_contacts'] / sum(groups['not_see_contacts']), 2)
groups


# In[62]:


px.bar(groups,
       x=['share_see_contacts', 'share_not_see_contacts'],
       y='event_name',
       text_auto=True,
       barmode='group',
       color_discrete_map={'share_see_contacts': '#fb9f3a', 'share_not_see_contacts': 'royalblue'}) \
.update_layout(plot_bgcolor='rgba(0,0,0,0)',
               autosize=False,
               yaxis = {"categoryorder":"total ascending"},
               title={
              'text': "Зависимость частоты событий от группы пользователей",
               'y':0.98,
               'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'
                },
                xaxis_title="Частота событий",
                yaxis_title="",
                font=dict(
                size=15),
                width=900,
                height=900,
                legend=dict(
                title='Группы пользователей',
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.7
                )) \
.update_xaxes(showline=True,
              linewidth=2,
              linecolor='black',
              gridcolor='LightGrey',)
#range=[-0.1, 0.45])


# Группа пользователей, которая совершала целевое действие (просмотр контакта) менее активна - меньше пользуются поиском, реже видят рекомендации, кликают по ним и открывают карточки объявлений.

# **Вывод:**
# 
# <u>Отобраны следующие сценарии/паттерны:<u>
#     
# `Сценарий 1`: tips_show-map-contact_show/увидел рекомендованные объявления-открыл карту объявлений-посмотрел номер телефона.
# С рекомендованных объявлений менее половины пользователей переходят к карте объявлений и только 10% пользователей совершают целевое действие. Низкая конверсия на шаге просмотр карты объявлений-просмотр контактов может быть связан с технической ошибкой.
# 
#     
# `Сценарий 2`: search_1, photos_show, contacts_show/поиском по сайту, просмотрел фотографии в объявлении, посмотрел номер телефона
# При переходе c поиска на просмотр фотографии в объявлении конверсия составляет 82%, что является хорошим показателем. А вот конверсия в целевое события с просмотра фотографии мала - всего 24%. Это может быть связано с неудобным интерфейсом приложения/качеством фотографий и т.д.
# 
#     
# `Сценарий 3`: tips_show, advert_open, contacts_show/ увидел рекомендованные объявления-открыл карточки объявления-посмотрел номер телефона
# Только около пятой части пользователей открывают карточки объявления после просмотра рекомендованных объявлений и всего 3% просматривают контакты. Возможно, стоит проработать рекомендации, которые получают пользователи и понять их алгоритм. 
# Также отметим, что сценарий tips_show, tips_click, contacts_show случается не так часто, скорее всего, рекомендации не работают эффективно.
# 
#     
# - Хороший показатель конверсии наблюдается у события favorites_add (добавил объявление в избранное). Значит, после добавления объявления в избранное, более трети пользователей совершают целевое событие.
# 
#     
# - Сессий без события photos_show оказалось больше (1801 против 302), но при этом продолжительность сессий с целевым событием и событием photos_show оказалась длительнее.
#    
#     
# - Группа пользователей, которая совершала целевое действие (просмотр контакта) менее активна - меньше пользуются поиском, реже видят рекомендации, кликают по ним и открывают карточки объявлений.
# 
#  

# ### Шаг 5. Проверка гипотез

# #### Первая гипотеза:

# - Одни пользователи совершают действия `tips_show и tips_click` , другие — только `tips_show`. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры есть

# Применим гипотезу о равенстве долей. Для этого найдем количество уникальных пользователей, которые совершают действия tips_show и tips_click,а также количество уникальных пользователей, которые совершают действия tips_show, tips_click и целевое действие:

# In[63]:


group_one_all = len(set(data_new.query('event_name == "tips_show"')['user_id'].unique())                     .intersection(set(data_new.query('event_name == "tips_click"')['user_id'].unique())))
group_one = len(set(data_new.query('event_name == "tips_show"')['user_id'].unique())                     .intersection(data_new.query('event_name == "tips_click"')['user_id'].unique())                     .intersection(data_new.query('event_name == "contacts_show"')['user_id'].unique()))


# Теперь найдем количество уникальных пользователей, которые совершают только действие tips_show ,а также количество уникальных пользователей, которые совершают действия tips_show и целевое действие

# In[64]:


group_two_all = len(set(data_new.query('event_name == "tips_show"')['user_id'].unique())                     .intersection(data_new.query('event_name != "tips_click"')['user_id'].unique()))
group_two = len(set(data_new.query('event_name == "tips_show"')['user_id'].unique())                     .intersection(data_new.query('event_name != "tips_click"')['user_id'].unique())
                    .intersection(data_new.query('event_name == "contacts_show"')['user_id'].unique()))


# In[65]:


alpha = 0.05  # критический уровень статистической значимости

successes  = np.array([group_one, group_two])
trials = np.array([group_one_all, group_two_all])

# пропорция в первой группе:
p1 = successes[0]/trials[0]

# пропорция во второй группе:
p2 = successes[1]/trials[1]
# пропорция в комбинированном датасете:
p_combined = (successes[0] + successes[1]) / (trials[0] + trials[1])

# разница пропорций в датасетах
difference = p1 - p2 

z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/trials[0] + 1/trials[1]))

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


# <u>Cтатистически значимая разница между конверсиями в просмотры есть у группы, которая совершают действия tips_show и tips_click и группы, которые совершают действия tips_show.<u>

# #### Вторая гипотеза:

# - Пользователи установливают приложения, переходя из трех источников - Yandex, Google и other. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у пользователей из источников Yandex и Google нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры контактов у пользователей из источников Yandex и Google есть.

# Применим гипотезу о равенстве долей. Для этого найдем количество уникальных пользователей, которые перешли из источника yandex и  количество уникальных пользователей, которые совершают целевое действие, а также найдем количество уникальных пользователей, которые совершают только действие tips_show ,а также количество уникальных пользователей, которые перешли из источника google и  количество уникальных пользователей, которые совершают целевое действие

# In[66]:


group_one_all = len(set(data_new.query('source == "yandex"')['user_id'].unique()))
group_one = len(set(data_new.query('source == "yandex"')['user_id'].unique())                     .intersection(data_new.query('event_name == "contacts_show"')['user_id'].unique()))


# In[67]:


group_two_all = len(set(data_new.query('source == "google"')['user_id'].unique()))
group_two = len(set(data_new.query('source == "google"')['user_id'].unique())                     .intersection(data_new.query('event_name == "contacts_show"')['user_id'].unique()))


# In[68]:


alpha = 0.05  # критический уровень статистической значимости

successes  = np.array([group_one, group_two])
trials = np.array([group_one_all, group_two_all])

# пропорция в первой группе:
p1 = successes[0]/trials[0]

# пропорция во второй группе:
p2 = successes[1]/trials[1]
# пропорция в комбинированном датасете:
p_combined = (successes[0] + successes[1]) / (trials[0] + trials[1])

# разница пропорций в датасетах
difference = p1 - p2 

z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/trials[0] + 1/trials[1])) 
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


# <u>Cтатистически значимой разницы между конверсиями в просмотры контактов у пользователей из источников Yandex и Google нет<u>

# При подсчете конверсии выяснили, что показатель конверсии у события favorites_add выше, чем у других событий - около 40%. Попробуем подтвердим данный вывод с помощью статистического теста

# #### Третья гипотеза:

# - Одни пользователи совершают действия `favorites_add`, другие — нет. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры есть

# In[69]:


group_one_all = len(set(data_new.query('event_name == "favorites_add"')['user_id'].unique()))
group_one = len(set(data_new.query('event_name == "favorites_add"')['user_id'].unique())                     .intersection(data_new.query('event_name == "contacts_show"')['user_id'].unique()))

group_two_all = len(set(data_new.query('event_name != "favorites_add"')['user_id'].unique()))
group_two = len(set(data_new.query('event_name != "favorites_add"')['user_id'].unique())                     .intersection(data_new.query('event_name == "contacts_show"')['user_id'].unique()))


# In[70]:


alpha = 0.05  # критический уровень статистической значимости

successes  = np.array([group_one, group_two])
trials = np.array([group_one_all, group_two_all])

# пропорция в первой группе:
p1 = successes[0]/trials[0]

# пропорция во второй группе:
p2 = successes[1]/trials[1]
# пропорция в комбинированном датасете:
p_combined = (successes[0] + successes[1]) / (trials[0] + trials[1])

# разница пропорций в датасетах
difference = p1 - p2 

z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/trials[0] + 1/trials[1]))

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


# <u>Cтатистически значимая разница между конверсиями в просмотры у данных двух групп есть.<u>

# ### Шаг 6. Общий вывод:
# 
# Было проведен анализ, чтобы найти сценарии/"паттерны" для повышения вовлеченности пользователей в сервис "Ненужные вещи", найти интересные особенности и презентовать полученные результаты, которые помогут улучшить приложение с точки зрения пользовательского опыта.
# 
# 
# `Перед анализом проведена предобработка данных`:
# - наименования столбцов приведены в соответствии с синтаксисом;
# - тип данных столбца event_name датасета data изменен на тип данных datetime;
# - явные и неявные дупликаты не обнаружены;
# - объединены события show_contacts и contacts_show.
# 
# `В ходе исследования выявлены следующие закономерности:`
# 
# <u>События:<u>
# - Можно отметить, что событие `tips_show` (показ рекомендованных объявлений) происходило чаще всего - 40 055 раз. Также в топ-3 входят события `photos_show` (просмотр фотографий в объявлении) и `advert_open` (открытие карточки объявления) - 10 012 и 6 164 соответственно. Меньше всего пользователи совершали такие связанные с поиском события, как `search_6`, `search_2` и `search_7` - менее 500 раз.
#     
#     
# - В среднем на пользователя приходится около 17 событий. Минимальное и максимальное количество событий на пользователя - 1 и 478 соответственно. Получился сильный разброс значений. Есть пользователи, которые совершили только 1 событие (возможно, только увидели рекомендации и больше не совершали никаких действий) и пользователи, которые регулярно пользуются приложением. Так как величиная среднего значения не показательна при выбросах, будем ориентироваться на `значение медианы, которое составляет 9 событий`.
# 
#     
# - `Yandex` лидирует по привлечению пользователей - 1934 уникальных пользователя, у источников `other и google` - 1230 и 1129. Пользователи чаще всего приходят из источника yandex, но доля пользователей (24.72%), которые совершили целевое действие, почти такая же, что и источника из google (24.36%).
# 
#     
# - Имеются данные с 7 октября 2019 по 3 ноября 2019, что соответствует ТЗ. Судя по полученной гистограмме, изменение числа событий носит скачкообразный характер. В разбивке по дням недели наблюдается снижение числа событий с понедельника по субботу - с 11 671 до 9 154, затем в воскресенье количество событий увеличивается до 10 501.
# 
#     
# - В разрезе уникальных пользователей лидирует событие `tips_show`, а также `map` и `photos_show`. Пока что намечается  следующая последовательность событий: tips_show-map-photos_show-contact_show/увидел рекомендованные объявления-открыл карту объявлений-просмотрел фотографии в объявлении-посмотрел номер телефона.
# 
# <u>Сессии:<u>
# - В октябре на пользователя приходилось 2-3 сессии.
# - После удаления части сессий (более 11 тысяч), показатели средней и медианной продолжительности сессий изменились с 121,44 до 215,7 сек. и с 17 до 128 сек. Также важно понять, почему более половины сессий - "ошибочны".
# 
# - `Отобраны следующие сценарии/паттерны:`
#     
#   - Сценарий 1: tips_show-map-contact_show/увидел рекомендованные объявления-открыл карту объявлений-посмотрел номер телефона.
# С рекомендованных объявлений менее половины пользователей переходят к карте объявлений и только 10% пользователей совершают целевое действие. Низкая конверсия на шаге просмотр карты объявлений-просмотр контактов может быть связан с технической ошибкой.
# 
#   - Сценарий 2: search_1, photos_show, contacts_show/поиском по сайту, просмотрел фотографии в объявлении, посмотрел номер телефона
# При переходе c поиска на просмотр фотографии в объявлении конверсия составляет 82%, что является хорошим показателем. А вот конверсия в целевое события с просмотра фотографии мала - всего 24%. Это может быть связано с неудобным интерфейсом приложения/качеством фотографий и т.д.
# 
#   - Сценарий 3: tips_show, advert_open, contacts_show/ увидел рекомендованные объявления-открыл карточки объявления-посмотрел номер телефона
# Только около пятой части пользователей открывают карточки объявления после просмотра рекомендованных объявлений и всего 3% просматривают контакты. Возможно, стоит проработать рекомендации, которые получают пользователи и понять их алгоритм. 
# Также отметим, что сценарий tips_show, tips_click, contacts_show случается не так часто, скорее всего, рекомендации не работают эффективно.
# 
#     
# - Хороший показатель конверсии наблюдается у события favorites_add (добавил объявление в избранное). После добавления объявления в избранное, более трети пользователей совершают целевое событие.
# 
#     
# - Сессий без события photos_show оказалось больше (1801 против 302), но при этом продолжительность сессий с целевым событием и событием photos_show оказалась длительнее.
#    
#     
# - Группа пользователей, которая совершала целевое действие (просмотр контакта) менее активна - меньше пользуются поиском, реже видят рекомендации, кликают по ним и открывают карточки объявлений.
# 
#     
# `Перед проведением исследования были поставлены несколько гипотез:`
# 1. Одни пользователи совершают действия `tips_show и tips_click` , другие — только `tips_show`. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры есть
# 2. Пользователи установливают приложения, переходя из трех источников - Yandex, Google и other. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у пользователей из источников Yandex и Google нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры контактов у пользователей из источников Yandex и Google есть.
# 3. Одни пользователи совершают действия `favorites_add`, другие — нет. Постановка гипотез:
#    - Нулевая гипотеза: статистически значимой разницы между конверсиями в просмотры контактов у двух групп нет
#    - Альтернативная гипотеза: статистически значимая разница между конверсиями в просмотры есть
#     
# В результате удалось выявить зависимости:
# - Гипотеза 1 не подтверждена: Отвергаем нулевую гипотезу: между долями есть значимая разница.
# - Гипотеза 2 подтверждена: Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными.    
# - Гипотеза 3 не подтверждена: Отвергаем нулевую гипотезу: между долями есть значимая разница.   
# 
# 
# **Рекомендации:**
#     
# - Необходимо проработать рекомендательную систему, так как пользователи чаще всего видят рекомендованные объявления и при это конверсия в целевое действия мала. Возможно, рекомендации работают не эффективно и не учитывают запросов пользователей.
# - Исследовать причины технических ошибок, чтобы уменьшить число "ошибочных" сессий, когда их продолжительность равна 0.
