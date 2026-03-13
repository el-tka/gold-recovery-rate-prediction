import pandas as pd
import numpy as np


def load_data(train_path, test_path):
    """Загрузка обучающего и тестового датасета"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def remove_leakage_features(data):
    """
    Удаление признаков, которые недоступны на момент предсказания
    """
    drop_columns = [col for col in data.columns if 'output' in col and 'recovery' not in col]
    data = data.drop(columns=drop_columns, errors='ignore')
    return data


def handle_missing_values(data):
    """Обработка пропусков"""
    data = data.fillna(method="ffill")
    return data


def prepare_data(train, test):
    """Основная функция подготовки данных"""

    train = handle_missing_values(train)
    test = handle_missing_values(test)

    train = remove_leakage_features(train)
    test = remove_leakage_features(test)

    return train, test