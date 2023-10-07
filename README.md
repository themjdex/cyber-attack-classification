Cyber attack multi-classification
==============================

Проект для классификации 14 классов угроз по данным трафика.

Организация проекта
------------

    ├── catboost_info      <- Тех. инфа по процессу обучения модели Catboost
    ├── configs            <- YAML-конфиг с параметрами подготовки и обучения данных, включая признаки    
    │   └── config.yaml 
    ├── data
    │   └── raw            <- Датасеты для обучения модели и проверки инференса
    │
    ├── models             <- Файл обученной модели и JSON с метриками
    │
    ├── notebooks          <- Jupyter-тетрадка, где проводилось первоначальное исследование данных и тестовое обучение.
    ├── src                <- Исходный код проекта
    │   │
    │   ├── data           <- Скрипт первичной обработки датасета и разбития на train и valid выборки      
    │   │   └── make_dataset.py 
    │   │
    │   ├── entities       <- Скрипты с датаклассами параметров обучения и создания схемы классов
    │   │   └── feature_params.py
    │   │   └── split_params.py
    │   │   └── train_params.py
    │   │   └── train_pipeline_params.py
    │   │
    │   ├── inference      <- Скрипт создания POST-запроса на получение предсказаний модели по данным из data/raw 
    │   │   └── make_request.py
    │   ├── models         <- Скрипт с обучением, получением предсказаний, получением метрик и сериализации модели
    │   │   ├── model_fit_predict.py
    ├── .dockerignore      <- Список исключений для Docker
    ├── .gitignore         <- Список исключений для Git
    ├── app.py             <- Скрипт создания экземпляра FastAPI, описание endpoints, запуск локального хоста
    ├── Dockerfile         <- Файл Docker для постройки виртуального окружения
    ├── LICENSE            <- Файл лицензии
    ├── main_pipeline.py   <- Скрипт основного пайплайна для обучения модели
    ├── README.md          <- Информация о проекте и его использовании
    ├── requirements.txt   <- Текстовый файл с указанием требуемых к установке зависимостей
    └── setup.py           <- Cкрипт для поддержки установки проекта как пакета


--------
## Описание проекта

### Постановка задачи

Компания онлайн-сервис с высоким уровнем входящего трафика имеет специализированный отдел безопасности, который занимается фильтрацией и анализом трафика. Сотрудники этого отдела обратились за помощью в автоматизации выявления аномального и злонамеренного трафика. Задача — разработать модель, которая будет классифицировать трафик на нормальный и злонамеренный, присваивая нормальный класс или один из вредоносных:
- DoS Hulk
- PortScan
- DDoS
- DoS GoldenEye
- FTP-Patator
- SSH-Patator
- DoS slowloris
- DoS Slowhttptest
- Bot
- Web Attack & Brute Force
- Web Attack & XSS
- Infiltration
- Web Attack & Sql Injection
- Heartbleed

### Спецификация решения

В данном проекте реализован REST API сервис на базе FastAPI. Предполагается пакетный инференс, но может быть реализован синхронный. Frontend-часть отсутствует, предполагается, что данные будут добавлены в `data/raw` на момент запуска обучения или предсказания:

- `network_traffic_data.csv` для обучения модели
- `network_traffic_data_prep.csv` для инференса (в проекте уже добавлен тестовый датасет на 10 объектов)

Проект запускается на `0.0.0.0` и `8000` порту. Проверка доступных запросов и их схемы доступны по доп. адресу в адресной строке `.../docs`

Модель машинного обучения: `CatBoostClassifier`.

### Метрики качества
- F1-мера
- Precision
- Recall
- Accuracy
- ROC-AUC

После обучения модели метрики сохраняются в `metrics.json`. На момент обучения модели получены следующие метрики:

```
{
    "f1_score": 0.9876027757209501, 
    "precision": 0.9872319844212214, 
    "recall": 0.9883398564905415, 
    "accuracy": 0.9883398564905415, 
    "roc_auc_score": 0.9963944175514415
}
```

Метрики вычисляются со взвешенным средним значением, а roc_auc_score cо стратегией `'ovo'` (one-vs-one).

### Запуск проекта
1. Клонировать репозиторий, добавить датасеты (при необходимости)
2. Запустить в корне проекта для сборки образа:

`docker build -t cyber-attack-classification:v1 .`

3. Запустить:

`docker run -p 8000:8000 --rm cyber-attack-classification:v1`

## Заключение
В ходе данного проекта была создана модель классификации для мультиклассов с хорошими значениями метрик, после чего проект был разбит на модульную структуру для создания REST API сервиса. В качестве модели был выбран CatBoostClassifier, показавший хорошие результаты и не требующий слишком большой предобработки данных. Данный проект готов к масштабированию и подключению front-части. Из разделов, которые можно будет реализовать в будущем: покрыть код тестами, подключить отслеживание экспериментов в MLFlow, а также настроить запуск дообучения и инференса через Airflow.
