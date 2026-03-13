from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

RANDOM_STATE = 42


def train_random_forest(X, y):
    """Обучение Random Forest"""

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


def train_xgboost(X, y):
    """Обучение XGBoost"""

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


def cross_validate_model(model, X, y, metric):
    """Кросс-валидация модели"""

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring=metric,
        n_jobs=-1
    )

    return scores.mean()