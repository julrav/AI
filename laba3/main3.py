import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt

df=pd.read_csv("processed_city_lifestyle.csv")

print(df.head(10))

cols=["population_density", "avg_income", "internet_penetration", "avg_rent", "air_quality_index", "public_transport_score", "green_space_ratio"]
X_reg=df[cols]
y_reg=df["happiness_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

reg_tree=DecisionTreeRegressor(
    max_depth=5, random_state=42
)

reg_tree.fit(X_train, y_train)

y_pred=reg_tree.predict(X_test)

print(f"MSE={mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE={root_mean_squared_error(y_test, y_pred):.4f}")

plt.figure(figsize=(50, 40))
plot_tree(reg_tree, filled=True, feature_names=X_reg.columns)
plt.title("Дерево решений (регрессия happiness_score)")
plt.savefig("tree.png")
plt.show()

target_clf = "country_Europe"
cols = ["happiness_score", "population_density", "avg_income", "internet_penetration", "avg_rent", "air_quality_index", "public_transport_score", "green_space_ratio"]

X_clf = df[cols]
y_clf = df[target_clf]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

clf_tree=DecisionTreeClassifier(
    max_depth=5, random_state=42
)

clf_tree.fit(X_train_clf, y_train_clf)

y_proba=clf_tree.predict_proba(X_test_clf)[:, 1]

fpr, tpr, thresholds=roc_curve(y_test_clf, y_proba)
roc_auc=auc(fpr, tpr)

print(f"ROC-AUC: {roc_auc:.4f}")

plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.savefig("roc.png")
plt.show()
