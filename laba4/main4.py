import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

df=pd.read_csv("processed_dataset.csv")

X=df[["population_density", "avg_income", "internet_penetration", "avg_rent", "air_quality_index", "public_transport_score", "green_space_ratio"]]
y=df["country_Asia"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf=RandomForestClassifier(n_estimators=200, max_depth=5, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)

print("Accuracy (RF):", accuracy_score(y_test, y_pred_rf))

ab=AdaBoostClassifier(n_estimators=100, random_state=42)
ab.fit(X_train, y_train)

y_pred_ab=ab.predict(X_test)
y_pred_proba_ab=ab.predict_proba(X_test)

print("Accuracy (AdaBoost):", accuracy_score(y_test, y_pred_ab))

gb=GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

y_pred_gb=gb.predict(X_test)
y_pred_proba_gb=gb.predict_proba(X_test)

print("Accuracy (Gradient Boosting):", accuracy_score(y_test, y_pred_gb))

fpr_rf, tpr_rf, _=roc_curve(y_test, y_pred_proba_rf[:, 1])
fpr_ada, tpr_ada, _=roc_curve(y_test, y_pred_proba_ab[:, 1])
fpr_gb, tpr_gb, _=roc_curve(y_test, y_pred_proba_gb[:, 1])

auc_rf=auc(fpr_rf, tpr_rf)
auc_ada=auc(fpr_ada, tpr_ada)
auc_gb=auc(fpr_gb, tpr_gb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', linewidth=2)
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {auc_ada:.4f})', linewidth=2)
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {auc_gb:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые для различных моделей')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()