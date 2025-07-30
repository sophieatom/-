#!/usr/bin/env python
# coding: utf-8

# ### Проект SQL

# ### Описание проекта
# 
# Коронавирус застал мир врасплох, изменив привычный порядок вещей. В свободное время жители городов больше не выходят на улицу, не посещают кафе и торговые центры. Зато стало больше времени для книг. Это заметили стартаперы — и бросились создавать приложения для тех, кто любит читать.
# 
# Ваша компания решила быть на волне и купила крупный сервис для чтения книг по подписке. Ваша первая задача как аналитика — проанализировать базу данных.
# В ней — информация о книгах, издательствах, авторах, а также пользовательские обзоры книг. Эти данные помогут сформулировать ценностное предложение для нового продукта.

# ### Описание данных
# 
# Структура таблицы **books**:
# - `book_id` — идентификатор книги;
# - `author_id` — идентификатор автора;
# - `title` — название книги;
# - `num_pages` — количество страниц;
# - `publication_date` — дата публикации книги;
# - `publisher_id` — идентификатор издателя.
# 
# 
# Структура таблицы **authors**:
# - `author_id` — идентификатор автора;
# - `author` — имя автора.
# 
# 
# Структура таблицы **publishers**:
# - `publisher_id` — идентификатор издательства;
# - `publisher` — название издательства;
# 
# 
# Структура таблицы **ratings**:
# - `rating_id` — идентификатор оценки;
# - `book_id` — идентификатор книги;
# - `username` — имя пользователя, оставившего оценку;
# - `rating` — оценка книги.
# 
# 
# Структура таблицы **reviews**:
# - `review_id` — идентификатор обзора;
# - `book_id` — идентификатор книги;
# - `username` — имя пользователя, написавшего обзор;
# - `text` — текст обзора.

# ### Цели исследования

# - проанализировать базу данных сервиса для чтения книг по подписке
# - подготовить выводы по анализу

# ### Задачи:
# 1. Посчитать, сколько книг вышло после 1 января 2000 года;
# 2. Для каждой книги посчитать количество обзоров и среднюю оценку;
# 3. Определить издательство, которое выпустило наибольшее число книг толще 50 страниц, исключив тем самым из анализа брошюры;
# 4. Определить автора с самой высокой средней оценкой книг — учитывать только книги с 50 и более оценками;
# 5. Посчитать среднее количество обзоров от пользователей, которые поставили больше 48 оценок.

# Получим доступ к базе данных:

# In[1]:


# импортируем библиотеки
import pandas as pd
import sqlalchemy as sa

# устанавливаем параметры
db_config = {
            'user': 'praktikum_student', # имя пользователя
            'pwd': 'Sdf4$2;d-d30pp', # пароль
            'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
            'port': 6432, # порт подключения
            'db': 'data-analyst-final-project-db' # название базы данных
}

connection_string = 'postgresql://{user}:{pwd}@{host}:{port}/{db}'.format(**db_config)

# сохраняем коннектор
engine = sa.create_engine(connection_string, connect_args={'sslmode':'require'})
# чтобы выполнить SQL-запрос, пишем функцию с использованием Pandas

def get_sql_data(query:str, engine:sa.engine.base.Engine=engine) -> pd.DataFrame:
    '''Открываем соединение, получаем данные из sql, закрываем соединение'''
    with engine.connect() as con:
        return pd.read_sql(sql=sa.text(query), con = con)


# Исследуем первые строки таблиц в базе данных:

# In[2]:


# запишем функцию для вывода первых 5 строк датафрейма
def data(data):
    print('Первые пять строк датафрейма', data)
    query = '''SELECT * FROM ''' + data + ''' LIMIT 5'''
    display(get_sql_data(query))
    query = '''SELECT COUNT(*) FROM ''' + data + ''' LIMIT 5'''
    print('Всего строк в датафрейме:', int(get_sql_data(query).iloc[0]))
    print()
    print()


# In[3]:


for i in ['books', 'authors', 'publishers', 'ratings', 'reviews']:
    data(i)


# Выведены первые 5 строк каждого датафрейма. Проверим типы столбцов и наличие пропусков в датафреймах:

# In[4]:


# запишем функцию для вывода основных характеристик датафрейма
def data_char(data):
    print('Характеристики датафрейма', data)
    print()
    query = '''SELECT * FROM ''' + data
    display(get_sql_data(query).info())
    print()


# In[5]:


for i in ['books', 'authors', 'publishers', 'ratings', 'reviews']:
    data_char(i)


# Пропуски не обнаружены, типы столбцов соответствуют значениям, которые в них хранятся.

# - **Задание 1**. Посчитать, сколько книг вышло после 1 января 2000 года;

# In[6]:


query = '''SELECT COUNT(book_id) FROM books 
           WHERE publication_date > '2000-01-01'
        '''
print('Количество книг, опубликованное после 1 января 2000 года -', int(get_sql_data(query).iloc[0]))


# - **Задание 2**. Для каждой книги посчитать количество обзоров и среднюю оценку;

# In[7]:


query = '''SELECT books.book_id, COUNT(DISTINCT reviews.review_id), AVG(ratings.rating) FROM books 
           LEFT JOIN reviews ON reviews.book_id = books.book_id
           LEFT JOIN ratings ON ratings.book_id = books.book_id
           GROUP BY books.book_id
           ORDER BY AVG(ratings.rating) DESC,
                    COUNT(DISTINCT reviews.review_id) DESC
        '''
print('Количество обзоров и средняя оценка для каждой книги')
display(get_sql_data(query))
print('Всего книг с обзорами и оценками:', 
      len(get_sql_data(query)))
#get_sql_data(query)


# - **Задание 3**. Определить издательство, которое выпустило наибольшее число книг толще 50 страниц, исключив тем самым из анализа брошюры.

# In[8]:


query = '''SELECT publishers.publisher, COUNT(DISTINCT books.book_id) FROM books 
           LEFT JOIN publishers ON publishers.publisher_id = books.publisher_id
           WHERE books.num_pages > 50
           GROUP BY publishers.publisher
           ORDER BY COUNT(DISTINCT books.book_id) DESC
           LIMIT 1
        '''
display(get_sql_data(query))
print('Издательство, которое выпустило наибольшее число книг толще 50 страниц:',
     get_sql_data(query)['publisher'].iloc[0], '.',
     'Оно выпустило', get_sql_data(query)['count'].iloc[0], 'таких книг.')


# - **Задание 4**. Определить автора с самой высокой средней оценкой книг — учитывать только книги с 50 и более оценками

# In[9]:


query = '''
           SELECT authors.author, one.author_id, one.avg 
           FROM (SELECT books.author_id, AVG(rating) FROM books
                 LEFT JOIN ratings ON ratings.book_id = books.book_id 
                 WHERE books.book_id IN (
                                           SELECT ratings.book_id FROM ratings 
                                           GROUP BY ratings.book_id 
                                           HAVING COUNT(ratings.rating) > 50
                                           )
                GROUP BY books.author_id
                ORDER BY AVG(rating) DESC) AS one
           JOIN authors ON authors.author_id = one.author_id
           ORDER BY avg DESC
        '''
display(get_sql_data(query))


# Автор с самой высокой средней оценкой книг- `J.K. Rowling/Mary GrandPré` со средней оценкой 4.287.

# - **Задание 5**. Посчитайте среднее количество обзоров от пользователей, которые поставили больше 48 оценок.

# In[10]:


query = '''
           SELECT AVG(one.count) FROM (
                                       (SELECT COUNT(DISTINCT review_id) AS count FROM reviews
                                        WHERE username IN (
                                                           SELECT ratings.username FROM ratings 
                                                           GROUP BY ratings.username
                                                           HAVING COUNT(DISTINCT ratings.rating_id) > 48
                                                           )
                                        GROUP BY username)) AS one
        '''
print('Среднее количество обзоров от пользователей, которые поставили больше 48 оценок:', 
      float(get_sql_data(query).iloc[0]))


# **Общий вывод:**
# 
# - Пропуски не обнаружены, типы столбцов соответствуют значениям, которые в них хранятся.
# - Количество книг, опубликованное после 1 января 2000 года - `819`.
# - Всего в базе `1000` книг с определенным количеством обзоров и оценок.
# - Издательство, которое выпустило наибольшее число книг толще 50 страниц - `Penguin Books`, оно выпустило 42 таких книг.
# - Автор с самой высокой средней оценкой книг- `J.K. Rowling/Mary GrandPré` со средней оценкой `4.287`.
# - Среднее количество обзоров от пользователей, которые поставили больше 48 оценок: `24.0`
