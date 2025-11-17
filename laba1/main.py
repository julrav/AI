import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# загружаем данные
df = pd.read_csv("datasets/test.csv")

# смотрим
df.head()

# информация о данных
df.info()

print("Основные статистики:")
display(df.describe())

# размер данных
print("Размер данных:", df.shape)

# количество пропущенных значений для каждого столбца
print("Пропущенные значения:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# сохраняем копию для сравнения, исключая колонки
df_filled = df.drop(['Name', 'Cabin', 'PassengerId', 'Spa'], axis=1).copy()

# заполняем числовые колонки медианой
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df_filled[col].isnull().sum() > 0:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

# заполняем категориальные колонки модой
categorical_columns = df_filled.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df_filled[col].isnull().sum() > 0:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0]).astype(str)

# проверяем, что пропущенных значений не осталось
print("Пропущенные значения после заполнения:")
print(df_filled.isnull().sum())

# сравнение до и после
print("\nСравнение до заполнения:")
print(df.isnull().sum().sum())
print("После заполнения:")
print(df_filled.isnull().sum().sum())

scaler = MinMaxScaler()
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns

# создаем копию для нормализованных данных
df_normalized = df_filled.copy()

# применяем нормализацию MinMax
df_normalized[numeric_columns] = scaler.fit_transform(df_filled[numeric_columns])

# показываем результаты нормализации
print("Данные после MinMax нормализации:")
display(df_normalized[numeric_columns].head())
print("\nСтатистики после MinMax нормализации:")
display(df_normalized[numeric_columns].describe())

df_final = df_normalized.copy()

print("Категориальные колонки перед преобразованием:")
categorical_columns = df_final.select_dtypes(include=['object']).columns
print(categorical_columns.tolist())
print(f"Количество категориальных колонок: {len(categorical_columns)}")

print("Анализ категориальных колонок:")
for col in categorical_columns:
    unique_count = df_final[col].nunique()
    print(f"Колонка '{col}': {unique_count} уникальных значений")

# Применяем One-Hot Encoding для категориальных колонок
if len(categorical_columns) > 0:
    print(f"\nПрименяем One-Hot Encoding ко всем категориальным колонкам: {categorical_columns.tolist()}")
    df_final = pd.get_dummies(df_final, columns=categorical_columns, drop_first=True)

    print("Столбцы после One-Hot Encoding:")
    new_columns = [col for col in df_final.columns if any(cat_col in col for cat_col in categorical_columns)]
    print(new_columns)
else:
    print("Нет категориальных колонок для кодирования")

df_final.to_csv("processed_titanic.csv", index=False)

print(f"\nРазмер данных до преобразования: {df_normalized.shape}")
print(f"Размер данных после преобразования: {df_final.shape}")

# результат
print("\nПервые 5 строк после преобразования:")
display(df_final.head())

print("\nТипы данных после преобразования:")
print(df_final.dtypes.value_counts())