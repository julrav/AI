import os
import glob

print("Поиск файлов...")

# Ищем все CSV файлы в проекте
csv_files = glob.glob("**/*.csv", recursive=True)

print("Найдены CSV файлы:")
for file in csv_files:
    print(f"📁 {file}")

# Ищем specifically processed_titanic.csv
print("\nПоиск processed_titanic.csv...")
if os.path.exists("processed_titanic.csv"):
    print("✅ Найден в корневой папке!")
elif os.path.exists("laba1/processed_titanic.csv"):
    print("✅ Найден в папке laba1!")
else:
    print("❌ Файл не найден!")

# Покажем текущую рабочую директорию
print(f"\nТекущая рабочая директория: {os.getcwd()}")