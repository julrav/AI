df_final = df_normalized.copy()

print("Категориальные колонки перед преобразованием:")
categorical_columns = df_final.select_dtypes(include=['object']).columns
print(categorical_columns.tolist())
print(f"Количество категориальных колонок: {len(categorical_columns)}")

# для колонок с малым количеством уникальных значений (2-10) используем One-Hot Encoding
# для колонок с большим количеством уникальных значений используем Label Encoding (чтобы избежать переобучения)

low_cardinality_cols = []
high_cardinality_cols = []

for col in categorical_columns:
    unique_count = df_final[col].nunique()
    print(f"Колонка '{col}': {unique_count} уникальных значений")

    if unique_count <= 10:
        low_cardinality_cols.append(col)
    else:
        high_cardinality_cols.append(col)

print(f"\nКолонки для One-Hot Encoding: {low_cardinality_cols}")
print(f"Колонки для Label Encoding: {high_cardinality_cols}")

# применяем Label Encoding для колонок с большим количеством категорий
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in high_cardinality_cols:
    print(f"Применяем Label Encoding к: {col}")
    df_final[col] = label_encoder.fit_transform(df_final[col].astype(str))

# применяем One-Hot Encoding для колонок с малым количеством категорий
if len(low_cardinality_cols) > 0:
    print(f"Применяем One-Hot Encoding к: {low_cardinality_cols}")
    df_final = pd.get_dummies(df_final, columns=low_cardinality_cols, drop_first=True)

    print("Столбцы после One-Hot Encoding:")
    new_columns = [col for col in df_final.columns if any(low_col in col for low_col in low_cardinality_cols)]
    print(new_columns)

print("\nСохранение предобработанных данных в файл...")
df_final.to_csv("processed_titanic.csv", index=False)

print(f"\nРазмер данных до преобразования: {df_normalized.shape}")
print(f"Размер данных после преобразования: {df_final.shape}")

# результат
print("\nПервые 5 строк после преобразования:")
display(df_final.head())

print("\nТипы данных после преобразования:")
print(df_final.dtypes.value_counts())