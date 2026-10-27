# Customer Churn — MLOps-проєкт

## Мета проєкту
Побудувати відтворюваний MLOps-проєкт для прогнозування відтоку клієнтів (Customer Churn).
Задача: бінарна класифікація (`Churn`: 1 — клієнт пішов, 0 — залишився).
На цьому етапі навчено базову модель (baseline), з якою порівнюватимуться наступні покращення.

## Джерело даних
Набір даних **Telco Customer Churn** (7043 клієнти, 21 колонка):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Файл потрібно розмістити як `data/raw/churn.csv`.
Розподіл цільової змінної: `No` — 5174, `Yes` — 1869.

## Структура проєкту
```
customer-churn-mlops/
├── data/
│   ├── raw/            # початкові дані
│   └── processed/      # підготовлені дані
├── models/             # збережені моделі
├── notebooks/          # експерименти
├── reports/            # метрики
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   └── train.py
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## Встановлення залежностей
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## Запуск навчання
```bash
python src/train.py
```
Результат: `models/baseline_model.joblib` (у Git не додається) та `reports/baseline_metrics.json`.

## Модель
`StandardScaler → LogisticRegression(max_iter=2000)`, поділ 80/20 зі стратифікацією, `random_state=42`.
Під час підготовки даних видалено `customerID`, а `TotalCharges` перетворено на число.

## Отримані метрики
| Метрика   | Значення |
|-----------|----------|
| accuracy  | 0.807 |
| precision | 0.658 |
| recall    | 0.567 |
| f1        | 0.609 |
| roc_auc   | 0.842 |

## Висновок
Базова модель правильно класифікує ~81% клієнтів, ROC-AUC = 0.84.
Recall 0.57 означає, що модель знаходить трохи більше половини клієнтів, які йдуть,
тому надалі варто покращувати саме recall та f1. Ці значення — точка відліку для наступних робіт.