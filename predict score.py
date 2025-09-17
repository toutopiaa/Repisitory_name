import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

math_data = pd.read_csv("C:/Users/Lenovo/Desktop/student+performance/student/student-mat.csv", sep=";")
por_data = pd.read_csv("C:/Users/Lenovo/Desktop/student+performance/student/student-por.csv", sep=";")


merged_data = pd.merge(
    math_data, 
    por_data, 
    on=["school", "sex", "age", "address", "famsize", "Pstatus", 
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"],
    suffixes=("_math", "_por") 
)


features = merged_data.drop(columns=['G3_por', 'G1_math', 'G2_math', 'G3_math'])
target_raw = merged_data['G3_por']
target = (target_raw >= 10).astype(int)


label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column].astype(str))
    label_encoders[column] = le


imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(
    features_imputed, 
    target, 
    test_size=0.2, 
    random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f'准确率（Accuracy）：{accuracy:.2f}')
print(f'均方根误差（RMSE）：{rmse:.2f}')  
print(f'决定系数（R²）：{r2:.2f}')