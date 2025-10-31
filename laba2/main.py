import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка обработанных данных
df = pd.read_csv('processed_titanic.csv')
print("Данные успешно загружены")
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nКолонки в датасете:")
print(df.columns.tolist())

# 1. ПОДГОТОВКА ДАННЫХ И РАЗДЕЛЕНИЕ НА ВЫБОРКИ

print("=" * 60)
print("1. ПОДГОТОВКА ДАННЫХ И РАЗДЕЛЕНИЕ НА ВЫБОРКИ")
print("=" * 60)

# Создаем копию датасета для работы
df_processed = df.copy()

# Удаляем ненужные колонки
columns_to_drop = ['PassengerId']  # Удаляем только PassengerId
df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')

print("Обработанные данные:")
print(df_processed.head())
print(f"\nРазмер обработанного датасета: {df_processed.shape}")
print("Колонки после обработки:", df_processed.columns.tolist())

# Определяем целевую переменную для классификации
# Ищем колонку с Transported в названии
transported_columns = [col for col in df_processed.columns if 'Transported' in col]
if transported_columns:
    target_classification = transported_columns[0]
    print(f"Найдена целевая переменная для классификации: {target_classification}")
else:
    # Если нет Transported, используем первую булеву колонку
    bool_columns = df_processed.select_dtypes(include=[bool]).columns
    if len(bool_columns) > 0:
        target_classification = bool_columns[0]
        print(f"Используем булеву колонку для классификации: {target_classification}")
    else:
        # Создаем искусственную целевую переменную на основе Age
        target_classification = 'IsAdult'
        df_processed[target_classification] = (df_processed['Age'] > 18).astype(int)
        print(f"Создана искусственная целевая переменная: {target_classification}")

# РАЗДЕЛЕНИЕ ДЛЯ РЕГРЕССИИ: предсказание возраста (Age)
X_reg = df_processed.drop('Age', axis=1)
y_reg = df_processed['Age']

# РАЗДЕЛЕНИЕ ДЛЯ КЛАССИФИКАЦИИ
X_clf = df_processed.drop(target_classification, axis=1)
y_clf = df_processed[target_classification]

# Преобразуем булевы колонки в числовые для моделей
bool_columns_reg = X_reg.select_dtypes(include=[bool]).columns
X_reg[bool_columns_reg] = X_reg[bool_columns_reg].astype(int)

bool_columns_clf = X_clf.select_dtypes(include=[bool]).columns
X_clf[bool_columns_clf] = X_clf[bool_columns_clf].astype(int)
y_clf = y_clf.astype(int)

print(f"\nЦелевая переменная для регрессии: Age")
print(f"Целевая переменная для классификации: {target_classification}")

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

print("\nРазмеры выборок для классификации:")
print(f"Обучающая: {X_clf_train.shape}, Тестовая: {X_clf_test.shape}")

print(f"\nРаспределение классов в классификации:")
print(y_clf.value_counts())
print(f"Доля положительного класса: {y_clf.mean():.3f}")

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
plt.xlabel('Ошибка предсказания')
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

# 3. ОЦЕНКА И УЛУЧШЕНИЕ РЕГРЕССИОННОЙ МОДЕЛИ

print("\n" + "=" * 60)
print("3. ОЦЕНКА И УЛУЧШЕНИЕ РЕГРЕССИОННОЙ МОДЕЛИ")
print("=" * 60)

models_results = []

# Базовая линейная регрессия
models_results.append({
    'Модель': 'Линейная регрессия',
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae
})

# Ridge регрессия (L2 регуляризация)
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_reg_train, y_reg_train)
y_reg_pred_ridge = ridge_model.predict(X_reg_test)

mse_ridge = mean_squared_error(y_reg_test, y_reg_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_reg_test, y_reg_pred_ridge)

print("Ridge регрессия (L2 регуляризация):")
print(f"MSE: {mse_ridge:.4f}, RMSE: {rmse_ridge:.4f}, MAE: {mae_ridge:.4f}")

models_results.append({
    'Модель': 'Ridge (L2)',
    'MSE': mse_ridge,
    'RMSE': rmse_ridge,
    'MAE': mae_ridge
})

# Lasso регрессия (L1 регуляризация)
lasso_model = Lasso(alpha=0.1, random_state=42, max_iter=1000)
lasso_model.fit(X_reg_train, y_reg_train)
y_reg_pred_lasso = lasso_model.predict(X_reg_test)

mse_lasso = mean_squared_error(y_reg_test, y_reg_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_reg_test, y_reg_pred_lasso)

print("Lasso регрессия (L1 регуляризация):")
print(f"MSE: {mse_lasso:.4f}, RMSE: {rmse_lasso:.4f}, MAE: {mae_lasso:.4f}")

models_results.append({
    'Модель': 'Lasso (L1)',
    'MSE': mse_lasso,
    'RMSE': rmse_lasso,
    'MAE': mae_lasso
})

# Полиномиальная регрессия
try:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_reg_train_poly = poly.fit_transform(X_reg_train)
    X_reg_test_poly = poly.transform(X_reg_test)

    poly_model = LinearRegression()
    poly_model.fit(X_reg_train_poly, y_reg_train)
    y_reg_pred_poly = poly_model.predict(X_reg_test_poly)

    mse_poly = mean_squared_error(y_reg_test, y_reg_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    mae_poly = mean_absolute_error(y_reg_test, y_reg_pred_poly)

    print("Полиномиальная регрессия (степень 2):")
    print(f"MSE: {mse_poly:.4f}, RMSE: {rmse_poly:.4f}, MAE: {mae_poly:.4f}")

    models_results.append({
        'Модель': 'Полиномиальная',
        'MSE': mse_poly,
        'RMSE': rmse_poly,
        'MAE': mae_poly
    })
except Exception as e:
    print(f"Полиномиальная регрессия не удалась: {e}")

# Сравнение моделей
comparison_df = pd.DataFrame(models_results)
print("\nСРАВНЕНИЕ МОДЕЛЕЙ РЕГРЕССИИ:")
print(comparison_df.round(4))

# Визуализация сравнения моделей
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(comparison_df['Модель'], comparison_df['MSE'], color=['blue', 'orange', 'green', 'red'][:len(comparison_df)])
plt.title('Сравнение MSE')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.bar(comparison_df['Модель'], comparison_df['RMSE'], color=['blue', 'orange', 'green', 'red'][:len(comparison_df)])
plt.title('Сравнение RMSE')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.bar(comparison_df['Модель'], comparison_df['MAE'], color=['blue', 'orange', 'green', 'red'][:len(comparison_df)])
plt.title('Сравнение MAE')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Вывод лучшей модели
best_model_idx = comparison_df['RMSE'].idxmin()
best_model = comparison_df.loc[best_model_idx]
print(f"\nЛУЧШАЯ МОДЕЛЬ РЕГРЕССИИ: {best_model['Модель']}")
print(f"Лучший RMSE: {best_model['RMSE']:.4f}")

# 4. РЕШЕНИЕ ЗАДАЧИ КЛАССИФИКАЦИИ

print("\n" + "=" * 60)
print("4. РЕШЕНИЕ ЗАДАЧИ КЛАССИФИКАЦИИ")
print("=" * 60)

# Логистическая регрессия
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_clf_train, y_clf_train)
y_clf_pred = logreg_model.predict(X_clf_test)
y_clf_pred_proba = logreg_model.predict_proba(X_clf_test)[:, 1]

# Оценка модели классификации
accuracy = accuracy_score(y_clf_test, y_clf_pred)
precision = precision_score(y_clf_test, y_clf_pred)
recall = recall_score(y_clf_test, y_clf_pred)
f1 = f1_score(y_clf_test, y_clf_pred)

print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_clf_test, y_clf_pred)

plt.figure(figsize=(15, 5))

# График 1: Матрица ошибок
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')

# График 2: Важность признаков в классификации
plt.subplot(1, 3, 2)
feature_importance_clf = pd.DataFrame({
    'feature': X_clf.columns,
    'importance': abs(logreg_model.coef_[0])
})
feature_importance_clf = feature_importance_clf.sort_values('importance', ascending=True)
plt.barh(feature_importance_clf['feature'], feature_importance_clf['importance'], color='orange')
plt.xlabel('Важность признака')
plt.title('Важность признаков в классификации')

# График 3: Распределение вероятностей
plt.subplot(1, 3, 3)
if len(np.unique(y_clf_test)) == 2:
    plt.hist(y_clf_pred_proba[y_clf_test == 0], bins=30, alpha=0.7, label='Класс 0')
    plt.hist(y_clf_pred_proba[y_clf_test == 1], bins=30, alpha=0.7, label='Класс 1')
else:
    plt.hist(y_clf_pred_proba, bins=30, alpha=0.7, color='purple')
plt.xlabel('Вероятность положительного класса')
plt.ylabel('Частота')
plt.legend()
plt.title('Распределение вероятностей')

plt.tight_layout()
plt.show()

# Детальный отчет по классификации
print("\nДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ:")
print(classification_report(y_clf_test, y_clf_pred))

# Анализ важности признаков
print("\nТОП-5 самых важных признаков для классификации:")
print(feature_importance_clf.tail(5).sort_values('importance', ascending=False))

# 5. ОЦЕНКА И УЛУЧШЕНИЕ КЛАССИФИКАЦИОННОЙ МОДЕЛИ

print("\n" + "=" * 60)
print("5. ОЦЕНКА И УЛУЧШЕНИЕ КЛАССИФИКАЦИОННОЙ МОДЕЛИ")
print("=" * 60)

classification_results = []

# Базовая логистическая регрессия
classification_results.append({
    'Модель': 'Базовая',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1
})

# Масштабирование признаков
scaler = StandardScaler()
X_clf_train_scaled = scaler.fit_transform(X_clf_train)
X_clf_test_scaled = scaler.transform(X_clf_test)

# Логистическая регрессия с масштабированием
logreg_scaled = LogisticRegression(random_state=42, max_iter=1000)
logreg_scaled.fit(X_clf_train_scaled, y_clf_train)
y_clf_pred_scaled = logreg_scaled.predict(X_clf_test_scaled)

accuracy_scaled = accuracy_score(y_clf_test, y_clf_pred_scaled)
precision_scaled = precision_score(y_clf_test, y_clf_pred_scaled)
recall_scaled = recall_score(y_clf_test, y_clf_pred_scaled)
f1_scaled = f1_score(y_clf_test, y_clf_pred_scaled)

print("Логистическая регрессия с масштабированием:")
print(f"Accuracy: {accuracy_scaled:.4f}, Precision: {precision_scaled:.4f}, Recall: {recall_scaled:.4f}, F1: {f1_scaled:.4f}")

classification_results.append({
    'Модель': 'С масштабированием',
    'Accuracy': accuracy_scaled,
    'Precision': precision_scaled,
    'Recall': recall_scaled,
    'F1-score': f1_scaled
})

# Логистическая регрессия с L2 регуляризацией
logreg_l2 = LogisticRegression(penalty='l2', C=0.1, random_state=42, max_iter=1000)
logreg_l2.fit(X_clf_train_scaled, y_clf_train)
y_clf_pred_l2 = logreg_l2.predict(X_clf_test_scaled)

accuracy_l2 = accuracy_score(y_clf_test, y_clf_pred_l2)
precision_l2 = precision_score(y_clf_test, y_clf_pred_l2)
recall_l2 = recall_score(y_clf_test, y_clf_pred_l2)
f1_l2 = f1_score(y_clf_test, y_clf_pred_l2)

print("Логистическая регрессия с L2 регуляризацией:")
print(f"Accuracy: {accuracy_l2:.4f}, Precision: {precision_l2:.4f}, Recall: {recall_l2:.4f}, F1: {f1_l2:.4f}")

classification_results.append({
    'Модель': 'L2 регуляризация',
    'Accuracy': accuracy_l2,
    'Precision': precision_l2,
    'Recall': recall_l2,
    'F1-score': f1_l2
})

# Логистическая регрессия с балансировкой классов
logreg_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
logreg_balanced.fit(X_clf_train_scaled, y_clf_train)
y_clf_pred_balanced = logreg_balanced.predict(X_clf_test_scaled)

accuracy_balanced = accuracy_score(y_clf_test, y_clf_pred_balanced)
precision_balanced = precision_score(y_clf_test, y_clf_pred_balanced)
recall_balanced = recall_score(y_clf_test, y_clf_pred_balanced)
f1_balanced = f1_score(y_clf_test, y_clf_pred_balanced)

print("Логистическая регрессия с балансировкой классов:")
print(f"Accuracy: {accuracy_balanced:.4f}, Precision: {precision_balanced:.4f}, Recall: {recall_balanced:.4f}, F1: {f1_balanced:.4f}")

classification_results.append({
    'Модель': 'С балансировкой',
    'Accuracy': accuracy_balanced,
    'Precision': precision_balanced,
    'Recall': recall_balanced,
    'F1-score': f1_balanced
})

# Сравнение моделей классификации
clf_comparison = pd.DataFrame(classification_results)
print("\nСРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ:")
print(clf_comparison.round(4))

# Визуализация сравнения моделей
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
plt.figure(figsize=(12, 8))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.bar(clf_comparison['Модель'], clf_comparison[metric], color=['blue', 'orange', 'green', 'red'])
    plt.title(f'Сравнение {metric}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Вывод лучшей модели
best_clf_idx = clf_comparison['F1-score'].idxmax()
best_clf_model = clf_comparison.loc[best_clf_idx]
print(f"\nЛУЧШАЯ МОДЕЛЬ КЛАССИФИКАЦИИ: {best_clf_model['Модель']}")
print(f"Лучший F1-score: {best_clf_model['F1-score']:.4f}")

