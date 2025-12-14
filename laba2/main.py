import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processed_city_lifestyle.csv")

print("Первые 5 строк данных:")
print(df.head())
print(f"Размер данных: {df.shape}")
print(f"Столбцы: {df.columns.tolist()}")

# Регрессия: прогнозирование happiness_score
target_reg = "happiness_score"

# Ищем столбцы, связанные с city_name
city_name_cols = [col for col in df.columns if col.startswith('city_name_')]
cols_to_drop = city_name_cols + [target_reg]

print(f"\nУдаляем столбцы для регрессии: {cols_to_drop}")

X = df.drop(columns=cols_to_drop)
y = df[target_reg]

print(f"\nПризнаки для регрессии: {X.shape}")
print(f"Целевая переменная: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Анализ корреляций
corr_target = df.drop(columns=city_name_cols).corr()['happiness_score'].sort_values(ascending=False)
print(f"\nКорреляции с happiness_score:")
print(corr_target.head(10))

# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_test = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n--- РЕЗУЛЬТАТЫ РЕГРЕССИИ ---")
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Корень среднеквадратичной ошибки (RMSE): {rmse:.4f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
print(f"R2 score: {r2_test:.4f}")

# Классификация: счастлив человек в городе или нет
# Создаем бинарную целевую переменную на основе медианы happiness_score
median_happiness = df['happiness_score'].median()
# Используем другое имя для колонки, чтобы не было конфликта
happiness_class_col = 'is_happy_city'
df[happiness_class_col] = (df['happiness_score'] > median_happiness).astype(int)

print(f"\nПОДГОТОВКА ДЛЯ КЛАССИФИКАЦИИ")
print(f"Медианное значение happiness_score: {median_happiness:.2f}")
print(f"Распределение классов:")
print(df[happiness_class_col].value_counts())
print(f"0 - несчастлив (happiness_score ≤ {median_happiness:.2f})")
print(f"1 - счастлив (happiness_score > {median_happiness:.2f})")

# Удаляем все столбцы, связанные с городами и целевыми переменными
cols_to_drop_clf = city_name_cols + [target_reg, happiness_class_col]
X_clf = df.drop(columns=cols_to_drop_clf)
y_clf = df[happiness_class_col]

print(f"\nПризнаки для классификации: {X_clf.shape}")
print(f"Целевая переменная для классификации: {y_clf.shape}")

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
     X_clf, y_clf, test_size=0.4, random_state=42
)

# Логистическая регрессия
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_clf, y_train_clf)
y_pred_test_clf = logreg_model.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_test_clf)

print(f"\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print(f"Accuracy score: {accuracy:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test_clf, y_pred_test_clf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Несчастлив', 'Счастлив'],
           yticklabels=['Несчастлив', 'Счастлив'])
plt.title('Confusion Matrix - Happiness Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Дополнительная информация
print(f"\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
print(f"Коэффициенты регрессии: {len(linear_model.coef_)}")
print(f"Важные признаки для регрессии:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': linear_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print(feature_importance.head(10))