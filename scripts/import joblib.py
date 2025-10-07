import joblib
from sbs_model_train import clf, le

joblib.dump((clf, le), r"C:\Medispeak\scripts\sbs_model.pkl")
print("✅ Model saved successfully!")
