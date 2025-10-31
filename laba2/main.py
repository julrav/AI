import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка обработанных данных
df = pd.read_csv('processed_titanic.csv')
print("Данные успешно загружены")
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

# 1. ПОДГОТОВКА ДАННЫХ И РАЗДЕЛЕНИЕ НА ВЫБОРКИ

print("=" * 60)
print("1. ПОДГОТОВКА ДАННЫХ И РАЗДЕЛЕНИЕ НА ВЫБОРКИ")
print("=" * 60)

# Создаем копию датасета для работы
df_processed = df.copy()

# Обрабатываем категориальные переменные
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']

# Извлекаем информацию из кабины (палуба/номер/сторона)
df_processed['Deck'] = df_processed['Cabin'].str.split('/').str[0]
df_processed['Side'] = df_processed['Cabin'].str.split('/').str[2]

# Кодируем категориальные переменные
label_encoders = {}
for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

# Бинарное кодирование для булевых колонок
df_processed['CryoSleep'] = df_processed['CryoSleep'].astype(int)
df_processed['VIP'] = df_processed['VIP'].astype(int)
df_processed['Transported'] = df_processed['Transported'].astype(int)

# Удаляем ненужные колонки
columns_to_drop = ['PassengerId', 'Cabin', 'Name']
df_processed = df_processed.drop(columns=columns_to_drop)

print("Обработанные данные:")
print(df_processed.head())
print(f"\nРазмер обработанного датасета: {df_processed.shape}")

# РАЗДЕЛЕНИЕ ДЛЯ РЕГРЕССИИ: предсказание возраста (Age)
X_reg = df_processed.drop('Age', axis=1)
y_reg = df_processed['Age']

# РАЗДЕЛЕНИЕ ДЛЯ КЛАССИФИКАЦИИ: предсказание Transported
X_clf = df_processed.drop('Transported', axis=1)
y_clf = df_processed['Transported']

# Разделение для регрессии
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Разделение для классификации
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print("\nРазмеры выборок для регрессии (предсказание Age):")
print(f"Обучающая: {X_reg_train.shape}, Тестовая: {X_reg_test.shape}")

print("\nРазмеры выборок для классификации (предсказание Transported):")
print(f"Обучающая: {X_clf_train.shape}, Тестовая: {X_clf_test.shape}")

print(f"\nРаспределение классов в классификации:")
print(y_clf.value_counts())
print(f"Доля Transported=True: {y_clf.mean():.3f}")

# 2. РЕШЕНИЕ ЗАДАЧИ РЕГРЕССИИ (предсказание Age)

print("\n" + "=" * 60)
print("2. РЕШЕНИЕ ЗАДАЧИ РЕГРЕССИИ (предсказание Age)")
print("=" * 60)

# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_reg_train, y_reg_train)
y_reg_pred = linear_model.predict(X_reg_test)

# Оценка модели регрессии
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_pred)

print("РЕЗУЛЬТАТЫ РЕГРЕССИИ (предсказание Age):")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² score: {linear_model.score(X_reg_test, y_reg_test):.4f}")

# Визуализация результатов регрессии
plt.figure(figsize=(15, 5))

# График 1: Фактические vs Предсказанные значения
plt.subplot(1, 3, 1)
plt.scatter(y_reg_test, y_reg_pred, alpha=0.6, color='blue')
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
plt.xlabel('Фактический возраст')
plt.ylabel('Предсказанный возраст')
plt.title('Регрессия: Фактические vs Предсказанные значения')

# График 2: Ошибки предсказания
plt.subplot(1, 3, 2)
errors = y_reg_test - y_reg_pred
plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Ошибка предсказания (годы)')
plt.ylabel('Частота')
plt.title('Распределение ошибок')

# График 3: Важность признаков
plt.subplot(1, 3, 3)
feature_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': abs(linear_model.coef_)
})
feature_importance = feature_importance.sort_values('importance', ascending=True)
plt.barh(feature_importance['feature'], feature_importance['importance'], color='lightgreen')
plt.xlabel('Важность признака')
plt.title('Важность признаков в регрессии')

plt.tight_layout()
plt.show()

# Анализ важности признаков
print("\nТОП-5 самых важных признаков для регрессии:")
print(feature_importance.tail(5).sort_values('importance', ascending=False))

