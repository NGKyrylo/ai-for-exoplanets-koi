# Модель у Orange → далі Space

Ти збираєш і перевіряєш модель тут. Потім ми перенесемо ту саму схему фіч у `joblib` для Gradio / Hugging Face.

## Файл для навчання
`challenge/data/koi_train.csv`

- лише **CONFIRMED** / **FALSE POSITIVE**
- колонки INPUT + TARGET в кінці: `koi_disposition`
- **немає** `koi_score`, `fpflag_*`, імен

INPUT (9 ознак):
`koi_period`, `koi_duration`, `koi_depth`, `koi_impact`, `koi_model_snr`,
`koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`

## Workflow у Orange (мінімум)

1. **File** → `koi_train.csv`  
   - `koi_disposition` = categorical / target-клас  
   - решта numeric
2. **Select Columns** (за потреби) — target = disposition, features = 9 INPUT
3. **Data Sampler** — 70–80% train / rest test (або Cross Validation далі)
4. Learner (обери один, простий і стабільний):
   - **Random Forest**, або
   - **Gradient Boosting**, або
   - **Logistic Regression** (як baseline)
5. **Test & Score** — дивись AUC, CA (accuracy), F1; матрицю через **Confusion Matrix**
6. **Predictions** на test — подивись імовірності класів, якщо віджет показує

## Що зафіксувати і надіслати мені
Після того як модель «ок» на твій смак:

1. який learner + ключові налаштування (скрін Test & Score);
2. метрики (AUC / accuracy хоча б);
3. підтвердження списку 9 фіч (нічого зайвого не додавав);
4. (опційно) скрін Feature Importance, якщо є.

Експорт з Orange напряму на Space не обов’язковий: по твоєму скріну я зберу еквівалент у `sklearn` + `model.joblib` з тими ж фічами.

## Чого не робити
- не брати `CANDIDATE` у train;
- не додавати `koi_score` / fp-flags;
- не ганятись за 99% за рахунок leakage.
