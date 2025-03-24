# Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load cleaned dataset
df = pd.read_csv("../dataset/LFS_cleaned.csv")

# Clean and prepare the target column
df['PUFNEWEMPSTAT'] = df['PUFNEWEMPSTAT'].astype(str).str.strip()
df = df[df['PUFNEWEMPSTAT'].isin(['1', '2', '3'])]
df['PUFNEWEMPSTAT'] = df['PUFNEWEMPSTAT'].astype(int)

# Create binary target: 1 = Employed, 0 = Not Employed (includes Unemployed and Not in Labor Force)
df['TARGET'] = df['PUFNEWEMPSTAT'].map({1: 1, 2: 0, 3: 0})

# Select relevant features (numerical + categorical)
features = [
    'PUFC05_AGE',         
    'PUFC18_PNWHRS',      
    'PUFC19_PHOURS',     
    'PUFC25_PBASIC',      
    'PUFC28_THOURS',     
    'PUFC07_GRADE',       
    'PUFC23_PCLASS'       
]

df = df[features + ['TARGET']]

# Encode categorical variables
le_grade = LabelEncoder()
df['PUFC07_GRADE'] = le_grade.fit_transform(df['PUFC07_GRADE'].astype(str))

le_industry = LabelEncoder()
df['PUFC23_PCLASS'] = le_industry.fit_transform(df['PUFC23_PCLASS'].astype(str))

# Split into features and target
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(model, "logreg_model.pkl")
print("\nModel saved as logreg_model.pkl")
