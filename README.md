# Customer Churn — MLOps-проєкт

## Мета проєкту
Побудувати відтворюваний MLOps-проєкт для прогнозування відтоку клієнтів (Customer Churn).
Задача: бінарна класифікація (`Churn`: 1 — клієнт пішов, 0 — залишився).

## Джерело даних
Набір даних **Telco Customer Churn** (7043 клієнти, 21 колонка):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Сирі дані: `data/raw/churn.csv`.

## Структура проєкту
```
customer-churn-mlops/
├── configs/
│   └── data_schema.yaml          # схема даних для валідації
├── data/
│   ├── raw/churn.csv             # сирі дані
│   └── processed/
│       ├── churn_v1.csv          # версія даних v1
│       └── churn_v1_metadata.json
├── models/                       # збережені моделі (не в Git)
├── mlruns/                       # артефакти MLflow (не в Git)
├── mlflow.db                     # база MLflow: runs, параметри, метрики, реєстр (не в Git)
├── notebooks/
├── reports/
│   ├── data_validation_report.json
│   ├── latest_metrics.json       # метрики останнього запуску
│   └── model_selection.md        # вибір найкращої моделі
├── src/
│   ├── data_preparation.py
│   ├── validate_data.py          # валідація даних
│   ├── version_data.py           # версіонування даних
│   └── train.py                  # навчання + логування в MLflow
├── tests/
├── README.md
└── requirements.txt
```

## Встановлення залежностей
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## Запуск пайплайну
```bash
python src/validate_data.py
python src/version_data.py
python src/train.py --model-type logistic_regression
python src/train.py --model-type random_forest
```

## Валідація даних
Скрипт `src/validate_data.py` перевіряє сирі дані за схемою `configs/data_schema.yaml`:
наявність обов'язкових колонок, дублікати, частку пропусків (не більше 20%),
допустимі значення `Churn` (`Yes` / `No`) та коректність числових колонок.
Звіт: `reports/data_validation_report.json`.

- `passed` — критичних помилок немає, дані можна використовувати для навчання;
- `failed` — знайдено помилки (`errors`), скрипт завершується з кодом 1.

## Версіонування даних
Скрипт `src/version_data.py` створює версію даних **v1** (`data/processed/churn_v1.csv`)
та метадані з SHA-256 хешами (`data/processed/churn_v1_metadata.json`).

## Відстеження експериментів (MLflow)

### Запуск MLflow UI
В окремому терміналі (сервер має працювати весь час):
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Відкрити у браузері: http://127.0.0.1:5000 (режим **Model training**).

### Навчання з логуванням
```bash
python src/train.py --model-type logistic_regression
python src/train.py --model-type random_forest
```
Кожен запуск створює run в експерименті `customer-churn-baseline` і логує:
- параметри: `model_type`, `data_version`, `test_size`, `random_state`;
- метрики: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`;
- артефакт: `reports/latest_metrics.json`;
- модель, яка реєструється в Model Registry як `CustomerChurnClassifier`.

### Де зберігаються дані MLflow
- `mlflow.db` — база SQLite: експерименти, runs, параметри, метрики та Model Registry
  (MLflow 3.x використовує її за замовчуванням);
- `mlruns/` — артефакти запусків (файли звітів і моделей).

Обидва шляхи додано в `.gitignore`.

### Як переглянути Model Registry
У MLflow UI: **Model registry** → `CustomerChurnClassifier`.
Там видно всі версії моделі; актуальна модель позначена alias **@champion**.

## Моделі та результати (data_version: v1)
| Run | accuracy | precision | recall | f1 | roc_auc | Версія в реєстрі |
|-----|----------|-----------|--------|----|---------|------------------|
| logistic_regression_v1 | 0.805 | 0.654 | 0.561 | 0.604 | 0.843 | Version 1 **@champion** |
| random_forest_v1 | 0.785 | 0.618 | 0.497 | 0.551 | 0.825 | Version 2 |

Основна метрика — **recall**. Обрано логістичну регресію. Обґрунтування: `reports/model_selection.md`.

## Висновок
Додано відстеження експериментів за допомогою MLflow: параметри, метрики, артефакти та моделі
кожного запуску зберігаються автоматично і порівнюються в MLflow UI. Проведено два експерименти
на версії даних v1. За основною метрикою recall найкращою виявилась логістична регресія
(recall 0.561, ROC-AUC 0.843), яку зареєстровано в Model Registry як `CustomerChurnClassifier`
Version 1 з alias `@champion`.