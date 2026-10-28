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
├── notebooks/
├── reports/
│   ├── data_validation_report.json
│   └── baseline_metrics.json
├── src/
│   ├── data_preparation.py
│   ├── validate_data.py          # валідація даних
│   ├── version_data.py           # версіонування даних
│   └── train.py                  # навчання моделі
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
python src/train.py
```

## Валідація даних
Скрипт `src/validate_data.py` перевіряє сирі дані за схемою `configs/data_schema.yaml`:
- наявність обов'язкових колонок;
- дублікати рядків;
- частку пропущених значень (не більше 20%);
- допустимі значення цільової змінної `Churn` (`Yes` / `No`);
- коректність числових колонок (`tenure`, `MonthlyCharges`, `TotalCharges`).

Звіт зберігається у `reports/data_validation_report.json`.

**Статус:**
- `passed` — критичних помилок немає, дані можна використовувати для навчання;
- `failed` — знайдено помилки (`errors`), скрипт завершується з кодом 1, і навчання запускати не можна.

Попередження (`warnings`) не зупиняють пайплайн, але на них варто звернути увагу.
Поточний результат: `passed`, попередження — 11 порожніх значень у `TotalCharges`.

## Версіонування даних
Скрипт `src/version_data.py` створює версію даних **v1**:
- порожні `TotalCharges` замінюються на 0;
- видаляються дублікати;
- результат зберігається у `data/processed/churn_v1.csv`;
- метадані (версія, дата, кількість рядків, SHA-256 хеші вхідного та вихідного файлів) — у `data/processed/churn_v1_metadata.json`.

## Модель
`StandardScaler → LogisticRegression(max_iter=2000)`, поділ 80/20 зі стратифікацією, `random_state=42`.
Модель навчається на версії даних **v1** (`data/processed/churn_v1.csv`), версія записується в метрики.

## Отримані метрики (data_version: v1)
| Метрика   | Значення |
|-----------|----------|
| accuracy  | 0.807 |
| precision | 0.658 |
| recall    | 0.567 |
| f1        | 0.609 |
| roc_auc   | 0.842 |

## Висновок
Додано валідацію даних і версіонування датасету. Перевірка сирих даних за схемою
`configs/data_schema.yaml` завершилась зі статусом `passed` (7043 рядки, 0 дублікатів,
попередження — 11 порожніх значень у `TotalCharges`). Створено версію даних `v1` з метаданими
та SHA-256 хешами, і модель тепер навчається саме на ній, а версія даних записується в метрики.
Якість моделі не змінилась (accuracy 0.807, ROC-AUC 0.842), але тепер для кожного результату
відомо, на якій версії даних його отримано, що забезпечує відтворюваність.
