
# coding: utf-8

# Домашняя работа №2
# 
# Постановка задачи: [задачка с kaggle](https://www.kaggle.com/c/digit-recognizer)
# Здесь сначала Ваш код, потом мой, с комментариями. Этот документ = ход выполнения дз.
# 
# Импортируем нужные модули:

# In[1]:

# модуль для работы с матрицами
import numpy as np

# для чтения данных
import pandas as pd

# инструменты для работы с картинками
from scipy.ndimage.interpolation import shift, rotate, zoom
from scipy import special

# графика
import matplotlib.pyplot as plt

# импортируем классификаторы
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# модуль для измерения точности классификации
from sklearn import metrics

# модуль для работы с данными для модели
from sklearn import model_selection

# библиотека для измерения времени
import timeit

# установка опций выводы numpy объектов
np.set_printoptions(precision=3, threshold=100, suppress=True)


# Скачайте обучающую и тестовую выборки.

# Давайте прочитаем данные:

# In[2]:

data = pd.read_csv('train.csv')

digits = []
labels = []
for digit in data.as_matrix():
    digits.append(digit[1:])
    labels.append(digit[0])


# Теперь у нас есть два массива: `digits`, где хранятся матрицы в виде векторов с записанными данными о каждом пикселе, и `labels`, где хранятся данные о том, какая цифра изображена на картинке.

# Теперь сделаем функцию, которая позволяет нам вывести некоторое множество картинок, и посмотрим на данные.

# In[3]:

get_ipython().magic(u'matplotlib inline')

def show_digit_imgs(digits, size=3):
    fig, axarr = plt.subplots(size, size, figsize=(1.5 * size, 1.5 * size), sharex=True, sharey=True)

    for i in range(size ** 2):
        axarr[i // size][i % size].imshow(digits[i].reshape(28, 28), cmap='Greys')

    # уберем расстояние между картинками
    fig.subplots_adjust(hspace=0, wspace=0)
    
    # уберем значения координат
    plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
    
    # уберем координатные "насечки"
    [a.xaxis.set_ticks_position('none') for a in fig.axes]
    [a.yaxis.set_ticks_position('none') for a in fig.axes]

    return fig, axarr
    
show_digit_imgs(digits, size=8);


# (**Препроцессинг (до 3 баллов)** -- я не делала)
# 

# **Работа с различными моделями (до трех баллов за анализ каждой модели)**

# Разобьем нашу выборку на две части, тренировочную и тестовую. На первой выборке (или ее части) будем обучать модель, на второй части будем смотреть качество модели.

# In[4]:

train_input, test_input, train_output, test_output = model_selection.train_test_split(digits, labels,
                                                                      test_size=0.5, train_size=0.5,
                                                                      random_state=42)
# train_input = digits[:len(digits) // 2]
# train_output = labels[:len(labels) // 2]
# control_input = digits[len(digits) // 2:]
# control_output = labels[len(labels) // 2:]

len(digits)


# Напишем функцию, позволяющую запустить модель на тренировочной и тестовой выборках и выводящая результат работы этой модели.

# In[5]:

def run_model(model, train_input, train_output, test_input, test_output):
    # выводим информацию о модели
    print(model, '\n')
    
    # обучаем модель
    model.fit(train_input, train_output)
    
    # размечаем с ее помощью тестовую выборку
    predicted_output = model.predict(test_input)
    
    # распечатываем результаты
    print(metrics.classification_report(test_output, predicted_output))
    print(metrics.confusion_matrix(test_output, predicted_output))


# Приведем пример запуска этой функции.

# In[6]:

# в качестве модели берем классификатор по ближайшим соседям, устанавливаем число соседей, равное трем.
model = KNeighborsClassifier(n_neighbors=3, p=1)

# берем только размер обучающей выборки равный 300, для того, чтобы модель работала не слишком долго
run_model(model, train_input[:300], train_output[:300], test_input, test_output)


# Что у нас означают эти таблички?
# 
# Вначале напечатано название модели и параметры, с которыми она была запущена. Затем вывелась табличка, где в первом столбцы указаны метки классов (названия цифр). Далее идут столбцы:
# 
# `precision` - доля предсказаний, которые указали правильную метку (подробнее про этот и следующий параметры можно почитать в [википедии](https://en.wikipedia.org/wiki/Precision_and_recall));
# 
# `recall` - доля элементов класса, для которых предсказание дало правильную метку;
# 
# `f1-score` - общая мера точности модели для данного класса, см. [вики](https://en.wikipedia.org/wiki/F1_score);
# 
# `support` - количество вхождений каждого класса в `control_output`.
# 
# Далее выведена [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

# Также вы можете замерять время работы вашей модели. Для этого надо немного видоизменить код. Теперь внизу еще будет выводиться время работы вашей модели.

# In[8]:

model = KNeighborsClassifier(n_neighbors=3, p=2,weights='distance')

time = timeit.timeit(lambda: run_model(model,
                                       train_input[:300], train_output[:300],
                                       test_input, test_output), number=1)
print('\ntime: {}'.format(time))


# Далее я по очереди исследую модели (запускаю с разными параметрами, смотрю результат):
# `LogisticRegression`,
# `GaussianNB`,
# `KNeighborsClassifier`,
# `DecisionTreeClassifier`,
# `RandomForestClassifier`
# 
# Результаты я заношу в таблицу: https://docs.google.com/spreadsheets/d/16h1wsqE6th5fiUpH0V2vkx4VE0uw8yzDm20ljLC3hk4/edit?usp=sharing
# 

# # I    ИССЛЕДУЮ МОДЕЛЬ 'KNeighborsClassifier'
# 
# С помощью GridSearchCV я перебираю параметры и ищу ту комбинацию параметров, которая далет максимальный score. Для KNeighborsClassifier я выбрала как важные параметры p, weights, n_neighbors.
# При выборе количества соседей менее 3, "наилучший" результат выдавался с одним соседом, что явно ошибочyо, поэтому я ограничила поиск от 3 соседей до 10.
# Сначала прогоним для размера выборки 300:

# In[21]:

from sklearn.model_selection import GridSearchCV
parameters = {'weights':('uniform', 'distance'), 'p':[1,2], 'n_neighbors':range(3,10)}
svr = KNeighborsClassifier()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:300], train_output[:300])
sorted(clf.cv_results_.keys())


# И посмотрим на наилучший скор, и какими параметрами он реализуется:

# In[22]:

print(clf.best_params_)
print(clf.best_score_)



# Заносим результаты в таблицу и двигаемся дальше: делаем прогон для размера выборки 1000:

# In[ ]:

from sklearn.model_selection import GridSearchCV
parameters = {'weights':('uniform', 'distance'), 'p':[1,2], 'n_neighbors':range(3,10)}
svr = KNeighborsClassifier()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:1000], train_output[:1000])
sorted(clf.cv_results_.keys())


# Снова смотрим на наилучший скор, и какими параметрами он реализуется; Заносим всё в таблицу.

# In[22]:

print(clf.best_params_)
print(clf.best_score_)


# Далее, делаем прогон для размера выборки 3000:

# In[ ]:

from sklearn.model_selection import GridSearchCV
parameters = {'weights':('uniform', 'distance'), 'p':[1,2], 'n_neighbors':range(3,10)}
svr = KNeighborsClassifier()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:1000], train_output[:1000])
sorted(clf.cv_results_.keys())


# Снова смотрим на наилучший скор, и какими параметрами он реализуется; Заносим всё в таблицу:

# In[ ]:

print(clf.best_params_)
print(clf.best_score_)


# Делаем прогон для размера выборки 5000:

# In[ ]:

from sklearn.model_selection import GridSearchCV
parameters = {'weights':('uniform', 'distance'), 'p':[1,2], 'n_neighbors':range(3,10)}
svr = KNeighborsClassifier()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:5000], train_output[:5000])
sorted(clf.cv_results_.keys())


# Снова смотрим на наилучший скор, и какими параметрами он реализуется; Заносим всё в таблицу:

# In[ ]:

print(clf.best_params_)
print(clf.best_score_)


# На этом этапе уже хочется посмотреть, как себя ведет качество модели относительно самого важного параметра этой модели = относительно количества соседей. Вдруг, например, эта модель вообще не применима для данной задачи. Строим график score от n_neighbors.

# In[23]:

# таблица с результатами и параметрами
clf.grid_scores_

# отфильтровать из неё записи, где параметр p был 2, а параметр weights был distance и положить результат в переменную means
means = filter(lambda x: x[0]['p'] == 2 and x[0]['weights'] == 'distance', clf.grid_scores_)

# нарисовать график, в котором по оси абсцисс параметр n_neighbors, а по оси ординат score
plt.plot([x[0]['n_neighbors'] for x in means], [x[1] for x in means])


# На графике видно что качество от количества соседей зависит скачками, о плавном выходе на плато говорить не приходится, а значит, модель K Neighbors вообще не подходит для данной задачи классификации.
# (принтскрин графика также лежит по адресу https://drive.google.com/file/d/0B3myH2URBpgIN2F4cXplRmlYVmc/view?usp=sharing)

# На этот моменте я перестаю дальше заниматься моделью Kneighbors (тк она не применима к данной задаче).
# Однако, какой-то результат на kaggle послать все-таки надо.
# Обучаем нашу модель на всей тренировочной выборке, с наилучшими параметрами для 5000. (просто последнее что я посчитала. В принципе для 2000 и для 5000 параметры были одинаковые: {'n_neighbors': 4, 'weights': 'distance', 'p': 2}, наверное и при увеличении выборки они будут такими же).

# Загрузка результатов на kaggle

# In[6]:

# обучим нашу модель при наилучших параметрах ('n_neighbors': 4, 'weights': 'distance', 'p': 2)
# и возьмем обучающую выборку максимального размера
model = KNeighborsClassifier(n_neighbors=4, p=2, weights='distance')

# обучаем модель
model.fit(digits, labels)





# применяю обученную модель на тестовую выборку!

# In[7]:

test_data = pd.read_csv('test.csv')
test_data[:10]

# сохранить предсказания в переменную predictions
predictions = model.predict(test_input)

# создать из предсказаний объект DataFrame (и назвать колонку в нём Label) и сохранить в переменную predictions_df
predictions_df = pd.DataFrame(predictions, columns=['Label'])

# обозвать индексную колонку ImageId
predictions_df.index.name = 'ImageId'

# записать в файлик
predictions_df.to_csv('KNeighb_predictions_astronomka.csv')


#predictions = model.predict(test_data)
#predictions.to_csv('Kneighb_n4_p2_wdist.csv')


# Выгружаю результат в каггл.
# 

# # II    ИССЛЕДУЮ МОДЕЛЬ 'LogisticRegression'

# In[32]:

from sklearn.model_selection import GridSearchCV
parameters = {'C':np.linspace(0.00001, 1, num=1000),'penalty':('l1', 'l2')}
svr = LogisticRegression()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:300], train_output[:300])
sorted(clf.cv_results_.keys())


# In[33]:

print(clf.best_params_)
print(clf.best_score_)
print(clf.cv_results_)



# In[34]:

#посмотрим на время
model = LogisticRegression(C=1.0000000000000001e-05, penalty='l2')
time = timeit.timeit(lambda: run_model(model,
                                       train_input[:300], train_output[:300],
                                       test_input, test_output), number=1)
print('\ntime: {}'.format(time))


# In[36]:

# Тоже самое для 1000
from sklearn.model_selection import GridSearchCV
parameters = {'C':np.linspace(0.00001, 1, num=1000),'penalty':('l1', 'l2')}
svr = LogisticRegression()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:1000], train_output[:1000])
sorted(clf.cv_results_.keys())


# In[37]:

print(clf.best_params_)
print(clf.best_score_)
#print(clf.cv_results_)


# In[38]:

#посмотрим на время
model = LogisticRegression(C=1.0000000000000001e-05, penalty='l2')
time = timeit.timeit(lambda: run_model(model,
                                       train_input[:1000], train_output[:1000],
                                       test_input, test_output), number=1)
print('\ntime: {}'.format(time))


# In[40]:

# Тоже самое для 2000
from sklearn.model_selection import GridSearchCV
parameters = {'C':np.linspace(0.00001, 1, num=1000),'penalty':('l1', 'l2')}
svr = LogisticRegression()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:2000], train_output[:2000])
sorted(clf.cv_results_.keys())


# In[41]:

print(clf.best_params_)
print(clf.best_score_)
#print(clf.cv_results_)


# In[51]:

#посмотрим на время
model = LogisticRegression(C=1.0000000000000001e-05, penalty='l1')
time = timeit.timeit(lambda: run_model(model,
                                       train_input[:2000], train_output[:2000],
                                       test_input, test_output), number=1)
print('\ntime: {}'.format(time))


# In[44]:

# Тоже самое для 5000
from sklearn.model_selection import GridSearchCV
parameters = {'C':np.linspace(0.00001, 1, num=1000),'penalty':('l1', 'l2')}
svr = LogisticRegression()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(train_input[:5000], train_output[:5000])
sorted(clf.cv_results_.keys())


# In[45]:

print(clf.best_params_)
print(clf.best_score_)
#print(clf.cv_results_)


# In[50]:

#посмотрим на время
model = LogisticRegression(C=1.0000000000000001e-05, penalty='l2')
time = timeit.timeit(lambda: run_model(model,
                                       train_input[:5000], train_output[:5000],
                                       test_input, test_output), number=1)
print('\ntime: {}'.format(time))


# In[ ]:




# In[ ]:




# Видно что наилучшие параметры всегда одна и те же. Посмотрим на график зависимости качества от параметра С чтобы сделать вывод о том, применим ли для данной задачи метод логистической регрессии и какое С разумнее брать.

# In[47]:

# для penalty = l2 


get_ipython().magic(u'matplotlib inline')
# таблица с результатами и параметрами
clf.grid_scores_

# отфильтровать из неё записи, где параметр penalty был l2 и положить результат в переменную means
means = filter(lambda x: x[0]['penalty'] == 'l2', clf.grid_scores_)

# нарисовать график, в котором по оси абсцисс параметр C, а по оси ординат score
plt.plot([x[0]['C'] for x in means], [x[1] for x in means])


# In[48]:

# для penalty = l1 


get_ipython().magic(u'matplotlib inline')
# таблица с результатами и параметрами
clf.grid_scores_

# отфильтровать из неё записи, где параметр penalty был l1 и положить результат в переменную means
means = filter(lambda x: x[0]['penalty'] == 'l1', clf.grid_scores_)

# нарисовать график, в котором по оси абсцисс параметр C, а по оси ординат score
plt.plot([x[0]['C'] for x in means], [x[1] for x in means])


# Отчет должен содержать вышеуказанную табличку, выводы о том, почему некоторые модели справляются с данной задачей классификации лучше, почему некоторые модели долго работают. Также нужно обосновать выбор ключевых параметров модели. Не забудьте указать результат 5 лучших классификаторов.
# 
# Баллы за коллективные работы будут снижаться пропорционально количеству участников!

# In[ ]:



