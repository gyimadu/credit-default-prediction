import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_excel('default of credit card clients.xls', skiprows=1)

df = df.dropna()


# PREPROCESSING
# Select features for model and target variable
df = df.drop(columns=['ID'])
X = df.drop(columns=['default payment next month'])
y = df['default payment next month']

# Scale features to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert categorical values into 0s and 1s (hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL TRAINING
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# classification report
print(classification_report(y_test, y_pred))

# calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'ROC-AUC Score: {roc_auc:.4f}')

new_data = [[60000, 1, 1, 1, 23, 2000, 1500, 5000, 2000, 1500, 5000, 3500, 4300, 5000, 3500, 3500, 4300,  0, 0, 1, 1, 0]]

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)
print('Prediction (0 = No Default, 1 = Default):', prediction[0])






