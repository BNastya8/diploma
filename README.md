# __Дипломный проект.__
__Тема: «Модель прогнозирования стоимости жилья для агентства недвижимости»__

# Оглавление
  1. [1. Описание проекта](https://github.com/BNastya8/diploma#1-%D0%BE%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82%D0%B0)
  2. Краткая информация о данных
  3. Этапы работы над проектом
  4. Результаты


# 1. Описание проекта 
Aгентство недвижимости столкнулось с проблемой — риелторы тратят слишком много времени на сортировку объявлений и поиск выгодных предложений. Поэтому скорость их реакции и качество анализа не дотягивают до уровня конкурентов. Это сказывается на финансовых показателях агентства.

___Задача:___
Разработать модель машинного обучения, которая поможет обрабатывать объявления и увеличит число сделок и прибыль агентства.

___Условия решения задачи:___
1) Провести разведывательный анализ и очистку исходных данных. Предстоит отыскать закономерности, самостоятельно расшифровать все сокращения, найти синонимы в данных, обработать пропуски и удалить выбросы.
2) Выделить наиболее значимые факторы, влияющие на стоимость недвижимости.
3) Построить несколько моделей для прогнозирования стоимости недвижимости. Выбрать лучшую.
4) Разработать небольшой веб-сервис, на вход которому поступают данные о некоторой выставленной на продажу недвижимости, а сервис прогнозирует его стоимость.

___Критерии оценивания проекта___
1) Анализ и обработка данных
2) Применение ML и DL
3) Оформление кода

___Практикуемые навыки:___
- предобработка данных;
- разведывательный анализ;
- построение диаграмм;
- обучение разных моделей регрессии;
- применение алгоритмов оптимизации;
- понимание и использование метрик качества моделей;
- подготовка модели к продакшену и деплой на сервер;
- работа с GitHub;
- контейнеризация и работа с Docker и DockerHub.

# 2. Краткая информация о данных

В датасете представлены 377185 объектов недвижимости. Данные предоставлены менторами Skillfactory. Данные реальные, заранее не обработанные, поэтому содержат всевозможные пропуки, ошибки ввода, дубли.
Описание данных:
* 'status' — статус продажи;
* 'private pool' и 'PrivatePool' — наличие собственного бассейна;
* 'propertyType' — тип объекта недвижимости;
* 'street' — адрес объекта;
* 'baths' — количество ванных комнат;
* 'homeFacts' — сведения о строительстве объекта (содержит несколько типов сведений, влияющих на оценку объекта);
* 'fireplace' — наличие камина;
* 'city' — город;
* 'schools' — сведения о школах в районе;
* 'sqft' — площадь в футах;
* 'zipcode' — почтовый индекс;
* 'beds' — количество спален;
* 'state' — штат;
* 'stories' — количество этажей;
* 'mls-id' и 'MlsId' — идентификатор MLS (Multiple Listing Service, система мультилистинга);
* 'target' — цена объекта недвижимости (целевой признак, который необходимо спрогнозировать).

Данные можно получить по ссылке https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view

# 3. Этапы работы над проектом
* Знакомство с данными;
* Предобработка;
* Разведывательный анализ данных;
* Формирование baseline-решения;
* Моделирование и решение задачи регрессии;
* Подготовка модели к продакшену.

# 4. Результаты
В результате работы, я подготовила датасет для обучения модели. Потом создала несколько моделей. Из всех вариантов я выбрала модель случайного леса, которая лучше всего предсказывает цену недвижимости. Проверила работоспособность модели в production_model.ipynb. Так же для проверки я написала сервер(локальный) и клиента, которые позволяет посмотреть модель в работе.



