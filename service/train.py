from catboost import CatBoostClassifier, Pool
import shap
import pandas as pd
import os
from service.features import FEATURE_COLUMNS, TARGET_COLUMN, CAT_FEATURES

MODEL_PATH = "models/catboost_model.cbm"


class LookalikeModel:
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=200, 
            learning_rate=0.1, 
            eval_metric='AUC',
            verbose=False,
            allow_writing_files=False
        )
        self.cat_features = CAT_FEATURES 
        self.explainer = None

    def fit(self, df):
        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMN]
        
        X[self.cat_features] = X[self.cat_features].astype(str).fillna("Unknown")
        
        self.model.fit(X, y, cat_features=self.cat_features)
        
        os.makedirs("models", exist_ok=True)
        self.model.save_model(MODEL_PATH)
        self.explainer = shap.TreeExplainer(self.model)

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.model.load_model(MODEL_PATH)
            self.explainer = shap.TreeExplainer(self.model)

    def predict(self, df):
        X = df[FEATURE_COLUMNS].copy()
        X[self.cat_features] = X[self.cat_features].astype(str).fillna("Unknown")
        return self.model.predict_proba(X)[:, 1]

    def get_reasons(self, df):
        X = df[FEATURE_COLUMNS].copy()
        X[self.cat_features] = X[self.cat_features].astype(str).fillna("Unknown")
        
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
            
        shap_values = self.explainer.shap_values(X)
        
        reasons_list = []
        for i in range(len(X)):
            # Берем топ-3 фичи по абсолютному влиянию
            feature_impacts = [(FEATURE_COLUMNS[j], float(shap_values[i][j])) for j in range(len(FEATURE_COLUMNS))]
            top_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)[:3]
            reasons_list.append([{"feature": f, "impact": round(v, 4)} for f, v in top_features])
            
        return reasons_list