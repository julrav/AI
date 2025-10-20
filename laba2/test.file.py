import os
import pandas as pd

print("Проверка создания processed_titanic.csv...")

if os.path.exists("processed_titanic.csv"):
    df = pd.read_csv("processed_titanic.csv")
    print("✅ Файл processed_titanic.csv успешно создан!")
    print(f"Размер: {df.shape}")
    print("\nПервые 3 строки:")
    print(df.head(3))
    print("\nСтолбцы:")
    print(df.columns.tolist())
else:
    print("❌ Файл processed_titanic.csv не создан!")
