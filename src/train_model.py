
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
MODEL_DIR = PROJECT_DIR / "model"

# 2. Load datasets
a_data = pd.read_csv(DATA_DIR / "A.csv")
b_data = pd.read_csv(DATA_DIR / "B.csv")
c_data = pd.read_csv(DATA_DIR / "C.csv")
d_data = pd.read_csv(DATA_DIR / "D.csv")
e_data = pd.read_csv(DATA_DIR / "E.csv")
f_data = pd.read_csv(DATA_DIR / "F.csv")
g_data = pd.read_csv(DATA_DIR / "G.csv")
h_data = pd.read_csv(DATA_DIR / "H.csv")
i_data = pd.read_csv(DATA_DIR / "I.csv")
k_data = pd.read_csv(DATA_DIR / "K.csv")
l_data = pd.read_csv(DATA_DIR / "L.csv")
m_data = pd.read_csv(DATA_DIR / "M.csv")
n_data = pd.read_csv(DATA_DIR / "N.csv")
o_data = pd.read_csv(DATA_DIR / "O.csv")
p_data = pd.read_csv(DATA_DIR / "P.csv")
q_data = pd.read_csv(DATA_DIR / "Q.csv")
r_data = pd.read_csv(DATA_DIR / "R.csv")
s_data = pd.read_csv(DATA_DIR / "S.csv")
t_data = pd.read_csv(DATA_DIR / "T.csv")
u_data = pd.read_csv(DATA_DIR / "U.csv")
v_data = pd.read_csv(DATA_DIR / "V.csv")
w_data = pd.read_csv(DATA_DIR / "W.csv")
x_data = pd.read_csv(DATA_DIR / "X.csv")
y_data = pd.read_csv(DATA_DIR / "Y.csv")


# 3. Combine data
Trained=[a_data, b_data, c_data, d_data, e_data, f_data, g_data,h_data,i_data,k_data,l_data,m_data,n_data,o_data
         ,p_data,q_data,r_data,s_data,t_data,u_data,v_data,w_data,x_data,y_data]
data = pd.concat(Trained, ignore_index=True)

# 4. Separate features and labels
X = data.drop(columns=["label"])
y = data["label"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train the classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 7. Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 8. Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR / "sign_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully at: {model_path}")