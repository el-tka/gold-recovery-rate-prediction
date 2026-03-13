from src.preprocessing import load_data, prepare_data
from src.features import split_features_target, align_features
from src.model import train_xgboost
from src.metrics import final_smape


# 1. Загрузка данных
train, test = load_data(
    "data/gold_recovery_train.csv",
    "data/gold_recovery_test.csv"
)


# 2. Предобработка
train, test = prepare_data(train, test)


# 3. Выделение признаков и таргета
X_train, y_train = split_features_target(train)
X_test, y_test = split_features_target(test)


# 4. Приведение train/test к одинаковым признакам
X_train, X_test = align_features(X_train, X_test)


# 5. Обучение модели (XGBoost)
model = train_xgboost(X_train, y_train)


# 6. Предсказания
predictions = model.predict(X_test)


# 7. Оценка качества
score = final_smape(
    y_test["rougher.output.recovery"],
    predictions[:, 0],
    y_test["final.output.recovery"],
    predictions[:, 1]
)

print("Final sMAPE:", score)