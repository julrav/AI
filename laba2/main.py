import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("ЛАБОРАТОРНАЯ РАБОТА №2")
print("=" * 50)


# Пытаемся найти и загрузить данные
def load_data():
    # Пробуем разные пути к файлу
    possible_paths = [
        "processed_titanic.csv",  # в корневой папке
        "../processed_titanic.csv",  # в папке выше
        "laba1/processed_titanic.csv",  # в папке laba1
        "../laba1/processed_titanic.csv",  # в папке laba1 на уровень выше
        "datasets/train.csv"  # исходные данные
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Найден файл: {path}")
            return pd.read_csv(path)

    print("❌ Файл processed_titanic.csv не найден!")
    print("Создаем демо-данные для тестирования...")
    return create_demo_data()


def create_demo_data():
    """Создает демо-данные если файл не найден"""
    np.random.seed(42)
    n_samples = 1000

    demo_data = {
        'Age': np.random.normal(30, 10, n_samples),
        'Fare': np.random.exponential(20, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples),
        'Sex': np.random.choice([0, 1], n_samples),  # 0-female, 1-male
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }

    # Добавляем некоторые пропущенные значения для реалистичности
    demo_data['Age'][:50] = np.nan
    demo_data['Fare'][:30] = np.nan

    df = pd.DataFrame(demo_data)

    # Заполняем пропущенные значения
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    print("✅ Демо-данные созданы!")
    return df


# Загружаем данные
df = load_data()
print(f"Размер данных: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nОсновные статистики:")
print(df.describe())

# Продолжаем с основным кодом лабораторной №2
print("\n" + "=" * 50)
print("РАЗДЕЛЕНИЕ ДАННЫХ НА ВЫБОРКИ")
print("=" * 50)

# Выбираем числовые колонки
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"Числовые колонки: {list(numeric_columns)}")

# Создаем матрицу признаков X
X = df[numeric_columns].copy()

# Для регрессии: предсказываем Age
if 'Age' in X.columns:
    y_regression = X['Age'].copy()
    X_regression = X.drop('Age', axis=1, errors='ignore')
    print(f"Регрессия: предсказываем Age на основе {X_regression.shape[1]} признаков")
else:
    # Если Age нет, используем другую числовую колонку
    available_targets = [col for col in X.columns if col != 'Survived']
    if available_targets:
        regression_target = available_targets[0]
        y_regression = X[regression_target].copy()
        X_regression = X.drop(regression_target, axis=1)
        print(f"Регрессия: предсказываем {regression_target} на основе {X_regression.shape[1]} признаков")
    else:
        print("❌ Нет подходящих колонок для регрессии")
        exit()

# Для классификации: предсказываем Survived
if 'Survived' in X.columns:
    y_classification = X['Survived'].copy()
    X_classification = X.drop('Survived', axis=1, errors='ignore')
    print(f"Классификация: предсказываем Survived на основе {X_classification.shape[1]} признаков")
else:
    # Если Survived нет, создаем искусственную целевую переменную
    y_classification = (X_regression.iloc[:, 0] > X_regression.iloc[:, 0].median()).astype(int)
    X_classification = X_regression.copy()
    print("Классификация: создана искусственная целевая переменная")

# Разделяем данные на обучающую и тестовую выборки
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42
)

print(f"\nРазмеры выборок:")
print(f"Обучающая (регрессия): {X_train_reg.shape}")
print(f"Тестовая (регрессия): {X_test_reg.shape}")
print(f"Обучающая (классификация): {X_train_clf.shape}")
print(f"Тестовая (классификация): {X_test_clf.shape}")

# РЕГРЕССИЯ
print("\n" + "=" * 50)
print("ЗАДАЧА РЕГРЕССИИ")
print("=" * 50)

# Создаем и обучаем модель регрессии
regression_model = LinearRegression()
regression_model.fit(X_train_reg, y_train_reg)
print("✅ Модель регрессии обучена!")

# Предсказания
y_pred_reg = regression_model.predict(X_test_reg)

# Оценка модели
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nОЦЕНКА МОДЕЛИ РЕГРЕССИИ:")
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R²): {r2:.2f}")

# Визуализация регрессии
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Фактические vs Предсказанные значения')

plt.subplot(1, 2, 2)
errors = y_pred_reg - y_test_reg
plt.hist(errors, bins=30, edgecolor='black')
plt.xlabel('Ошибка предсказания')
plt.ylabel('Частота')
plt.title('Распределение ошибок')

plt.tight_layout()
plt.show()

# КЛАССИФИКАЦИЯ
print("\n" + "=" * 50)
print("ЗАДАЧА КЛАССИФИКАЦИИ")
print("=" * 50)

# Создаем и обучаем модель классификации
classification_model = LogisticRegression(random_state=42, max_iter=1000)
classification_model.fit(X_train_clf, y_train_clf)
print("✅ Модель классификации обучена!")

# Предсказания
y_pred_clf = classification_model.predict(X_test_clf)

# Оценка модели
accuracy = accuracy_score(y_test_clf, y_pred_clf)

print(f"\nОЦЕНКА МОДЕЛИ КЛАССИФИКАЦИИ:")
print(f"Точность (Accuracy): {accuracy:.2f}")

# Детальный отчет
print("\nДЕТАЛЬНЫЙ ОТЧЕТ:")
print(classification_report(y_test_clf, y_pred_clf))

# Визуализация классификации
plt.figure(figsize=(15, 5))

# Матрица ошибок
plt.subplot(1, 3, 1)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_clf, y_pred_clf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанные')
plt.ylabel('Фактические')
plt.title('Матрица ошибок')

# Фактические vs Предсказанные
plt.subplot(1, 3, 2)
plt.scatter(range(len(y_test_clf)), y_test_clf, alpha=0.5, label='Фактические')
plt.scatter(range(len(y_pred_clf)), y_pred_clf, alpha=0.5, label='Предсказанные')
plt.xlabel('Наблюдения')
plt.ylabel('Класс')
plt.legend()
plt.title('Фактические vs Предсказанные классes')

# Вероятности
plt.subplot(1, 3, 3)
y_pred_proba = classification_model.predict_proba(X_test_clf)[:, 1]
plt.hist(y_pred_proba, bins=30, edgecolor='black')
plt.xlabel('Вероятность класса 1')
plt.ylabel('Частота')
plt.title('Распределение вероятностей')

plt.tight_layout()
plt.show()

# УЛУЧШЕНИЕ МОДЕЛЕЙ
print("\n" + "=" * 50)
print("УЛУЧШЕНИЕ МОДЕЛЕЙ")
print("=" * 50)

# Улучшение регрессии со стандартизацией
print("1. УЛУЧШЕНИЕ РЕГРЕССИИ:")
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

regression_model_improved = LinearRegression()
regression_model_improved.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_improved = regression_model_improved.predict(X_test_reg_scaled)

mse_improved = mean_squared_error(y_test_reg, y_pred_reg_improved)
r2_improved = r2_score(y_test_reg, y_pred_reg_improved)

print(f"До улучшения - MSE: {mse:.2f}, R²: {r2:.2f}")
print(f"После улучшения - MSE: {mse_improved:.2f}, R²: {r2_improved:.2f}")

# Улучшение классификации со стандартизацией
print("\n2. УЛУЧШЕНИЕ КЛАССИФИКАЦИИ:")
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

classification_model_improved = LogisticRegression(random_state=42, max_iter=1000)
classification_model_improved.fit(X_train_clf_scaled, y_train_clf)
y_pred_clf_improved = classification_model_improved.predict(X_test_clf_scaled)

accuracy_improved = accuracy_score(y_test_clf, y_pred_clf_improved)

print(f"До улучшения - Accuracy: {accuracy:.2f}")
print(f"После улучшения - Accuracy: {accuracy_improved:.2f}")

# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
print("\n" + "=" * 50)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Сохраняем предсказания
results = pd.DataFrame({
    'Actual_Regression': y_test_reg.values,
    'Predicted_Regression': y_pred_reg,
    'Actual_Classification': y_test_clf.values,
    'Predicted_Classification': y_pred_clf
})

results.to_csv("laba2/predictions_results.csv", index=False)
print("✅ Результаты сохранены в файл: laba2/predictions_results.csv")

print("\n" + "=" * 50)
print("ЛАБОРАТОРНАЯ РАБОТА №2 ВЫПОЛНЕНА!")
print("=" * 50)