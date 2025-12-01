import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

# загружаем данные
df = pd.read_csv("city_lifestyle_dataset.csv")

# смотрим
print("Первые 5 строк данных:")
display(df.head())

# информация о данных
print("\nИнформация о данных:")
df.info()

print("\nОсновные статистики:")
display(df.describe())

# размер данных
print("Размер данных:", df.shape)

# количество пропущенных значений для каждого столбца
print("\nПропущенные значения:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# создаем копию для обработки
df_filled = df.copy()

# заполняем числовые колонки медианой
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df_filled[col].isnull().sum() > 0:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

# заполняем категориальные колонки модой
categorical_columns = df_filled.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df_filled[col].isnull().sum() > 0:
        df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])

# проверяем, что пропущенных значений не осталось
print("\nПропущенные значения после заполнения:")
print(df_filled.isnull().sum())

print("\nПервые 5 строк после заполнения пропущенных значений:")
display(df_filled.head())
print("\nТипы данных после заполнения:")
print(df_filled.dtypes.value_counts())

# сравнение до и после
print("\nСравнение пропущенных значений:")
print(f"До заполнения: {df.isnull().sum().sum()}")
print(f"После заполнения: {df_filled.isnull().sum().sum()}")

# нормализация данных
scaler = MinMaxScaler()
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns

# создаем копию для нормализованных данных
df_normalized = df_filled.copy()

# применяем нормализацию MinMax
df_normalized[numeric_columns] = scaler.fit_transform(df_filled[numeric_columns])

# показываем результаты нормализации
print("\nДанные после MinMax нормализации:")
display(df_normalized[numeric_columns].head())
print("\nСтатистики после MinMax нормализации:")
display(df_normalized[numeric_columns].describe())

# кодирование категориальных переменных
df_final = df_normalized.copy()

print("\nКатегориальные колонки перед преобразованием:")
categorical_columns = df_final.select_dtypes(include=['object']).columns
print(categorical_columns.tolist())
print(f"Количество категориальных колонок: {len(categorical_columns)}")

print("\nАнализ категориальных колонок:")
for col in categorical_columns:
    unique_count = df_final[col].nunique()
    print(f"Колонка '{col}': {unique_count} уникальных значений")
    if unique_count <= 10:  # показываем значения для колонок с небольшим количеством уникальных значений
        print(f"  Значения: {df_final[col].unique()}")

# Применяем One-Hot Encoding для категориальных колонок
if len(categorical_columns) > 0:
    print(f"\nПрименяем One-Hot Encoding ко всем категориальным колонкам: {categorical_columns.tolist()}")
    df_final = pd.get_dummies(df_final, columns=categorical_columns, prefix=categorical_columns)

    print("Столбцы после One-Hot Encoding:")
    new_columns = [col for col in df_final.columns if any(cat_col in col for cat_col in categorical_columns)]
    print(f"Добавлено {len(new_columns)} новых столбцов:")
    for col in new_columns:
        print(f"  - {col}")
else:
    print("Нет категориальных колонок для кодирования")

# сохраняем обработанные данные
df_final.to_csv("processed_city_lifestyle.csv", index=False)

print(f"\nРазмер данных до преобразования: {df_normalized.shape}")
print(f"Размер данных после преобразования: {df_final.shape}")

# результат
print("\nПервые 5 строк после преобразования:")
display(df_final.head())

print("\nТипы данных после преобразования:")
print(df_final.dtypes.value_counts())

print("\nНазвания всех столбцов после обработки:")
print(df_final.columns.tolist())