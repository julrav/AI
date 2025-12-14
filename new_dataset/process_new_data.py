import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ======================== ЗАГРУЗКА ДАННЫХ ========================
filename = "city_lifestyle_dataset.csv"

# Проверка существования файла
if not os.path.exists(filename):
    print(f"ОШИБКА: Файл '{filename}' не найден!")
    print("\nПоиск файлов в текущей директории:")
    files = os.listdir('.')
    for f in files:
        if 'city' in f.lower() or 'lifestyle' in f.lower():
            print(f"  - {f}")
    exit(1)

try:
    # Загружаем данные
    df = pd.read_csv(filename)
    print(f"Файл успешно загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
    print("\nСтолбцы в данных:", list(df.columns))
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    exit(1)

# ======================== ПРОСМОТР ДАННЫХ ========================
print("\n" + "=" * 50)
print("АНАЛИЗ ДАННЫХ")
print("=" * 50)

print("\nПервые 5 строк данных:")
print(df.head())

print("\nИнформация о данных:")
df.info()

print("\nОсновные статистики:")
print(df.describe())

print("\nТипы данных:")
print(df.dtypes.value_counts())

# ======================== ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ========================
print("\n" + "=" * 50)
print("ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 50)

print("\nПропущенные значения в исходных данных:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Копируем данные для обработки
df_filled = df.copy()  # ВАЖНО: создаем копию!

print(f"\nСоздана копия данных: {df_filled.shape[0]} строк, {df_filled.shape[1]} колонок")

# Определяем, какие колонки могут быть в вашем датасете
# ВАЖНО: адаптируйте этот список под ваш реальный датасет
possible_columns_to_drop = ['Name', 'Cabin', 'PassengerId', 'Spa', 'Age', 'CryoSleep', 'VIP',
                            'id', 'ID', 'Index', 'Unnamed: 0']

# Проверяем, какие колонки действительно существуют
columns_to_drop = []
for col in possible_columns_to_drop:
    if col in df_filled.columns:
        columns_to_drop.append(col)

# Удаляем ненужные колонки
if columns_to_drop:
    df_filled = df_filled.drop(columns_to_drop, axis=1)
    print(f"Удалены колонки: {columns_to_drop}")
else:
    print("Не найдено колонок для удаления из списка возможных")

# Проверяем числовые колонки
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
print(f"\nЧисловые колонки ({len(numeric_columns)}): {list(numeric_columns)}")

# Проверяем категориальные колонки
categorical_columns = df_filled.select_dtypes(include=['object']).columns
print(f"Категориальные колонки ({len(categorical_columns)}): {list(categorical_columns)}")

# Заполняем числовые колонки медианой
for col in numeric_columns:
    if df_filled[col].isnull().sum() > 0:
        median_val = df_filled[col].median()
        df_filled[col] = df_filled[col].fillna(median_val)
        print(f"Заполнена колонка '{col}' медианой: {median_val:.4f}")

# Заполняем категориальные колонки модой
for col in categorical_columns:
    if df_filled[col].isnull().sum() > 0:
        mode_val = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
        df_filled[col] = df_filled[col].fillna(mode_val).astype(str)
        print(f"Заполнена колонка '{col}' модой: '{mode_val}'")

# Проверяем результат
total_missing = df_filled.isnull().sum().sum()
print(f"\nВсего пропущенных значений после заполнения: {total_missing}")

# ======================== НОРМАЛИЗАЦИЯ ЧИСЛОВЫХ ДАННЫХ ========================
print("\n" + "=" * 50)
print("НОРМАЛИЗАЦИЯ ДАННЫХ")
print("=" * 50)

# Обновляем список числовых колонок (могли измениться типы)
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns

if len(numeric_columns) > 0:
    scaler = MinMaxScaler()
    df_normalized = df_filled.copy()

    print(f"Нормализуем {len(numeric_columns)} числовых колонок")

    # Применяем нормализацию
    df_normalized[numeric_columns] = scaler.fit_transform(df_filled[numeric_columns])

    print("\nПример данных после нормализации:")
    print(df_normalized[numeric_columns].head())

    print("\nСтатистики после нормализации:")
    print(df_normalized[numeric_columns].describe().round(3))
else:
    df_normalized = df_filled.copy()
    print("Нет числовых колонок для нормализации")

# ======================== КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ ========================
print("\n" + "=" * 50)
print("КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
print("=" * 50)

# Обновляем список категориальных колонок
categorical_columns = df_normalized.select_dtypes(include=['object']).columns

if len(categorical_columns) > 0:
    print(f"Найдено {len(categorical_columns)} категориальных колонок:")

    for col in categorical_columns:
        unique_count = df_normalized[col].nunique()
        sample_values = df_normalized[col].unique()[:5]
        print(f"  '{col}': {unique_count} уникальных значений, примеры: {sample_values}")

    # Применяем One-Hot Encoding
    df_final = pd.get_dummies(df_normalized, columns=categorical_columns, drop_first=True)

    # Получаем новые колонки после кодирования
    new_categorical_cols = [col for col in df_final.columns
                            if any(cat_col in col for cat_col in categorical_columns)]

    print(f"\nСоздано {len(new_categorical_cols)} новых бинарных колонок")

    if len(new_categorical_cols) > 0:
        print("Примеры новых колонок:", new_categorical_cols[:10])
        if len(new_categorical_cols) > 10:
            print(f"... и еще {len(new_categorical_cols) - 10} колонок")

else:
    df_final = df_normalized.copy()
    print("Нет категориальных колонок для кодирования")

# ======================== СОХРАНЕНИЕ РЕЗУЛЬТАТА ========================
print("\n" + "=" * 50)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 50)

output_filename = "processed_city_lifestyle.csv"
df_final.to_csv(output_filename, index=False)

print(f"\nОбработка завершена успешно!")
print(f"Исходный размер данных: {df.shape}")
print(f"Финальный размер данных: {df_final.shape}")
print(f"Результат сохранен в файл: {output_filename}")

print(f"\nТипы данных в финальном датасете:")
print(df_final.dtypes.value_counts())

# Показываем первые строки
print("\nПервые 5 строк обработанных данных:")
print(df_final.head())