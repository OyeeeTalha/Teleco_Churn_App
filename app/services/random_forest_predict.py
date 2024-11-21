import pandas as pd
import joblib
import os

class RandomForestPredict:
    def __init__(self):
        self.rf_model = joblib.load('D:/GitRepos/Teleco_Churn_App/app/ml_models/random_forest_model.pkl')
        self.standard_scaler = joblib.load('D:/GitRepos/Teleco_Churn_App/app/ml_models/standard_scaler_model.pkl')
    def __data_preprocessing__(self,data):
        # cols = ['gender','SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        #        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling','PaymentMethod','tenure','MonthlyCharges','TotalCharges']
        df = pd.DataFrame([data])
        df.replace('No internet service', 'No', inplace=True)
        df.replace('No phone service', 'No', inplace=True)


        df.replace({'Yes': 1, 'No': 0}, inplace=True)
        df['gender'].replace({'Male': 0, 'Female': 1}, inplace=True)

        InternetServiceDummy = ['DSL', 'Fiber optic', 'No']
        ContractDummy = ['Month-to-month', 'One year', 'Two years']
        PaymentMethodsDummy = ['Bank transfer (automatic)','Credit card (automatic)','Electronic check', 'Mailed check']


        for category in InternetServiceDummy[1:]:  # Skip the first category to avoid multicollinearity
            df[f"InternetService_{category}"] = (df["InternetService"] == category).astype(int)

        for category in ContractDummy[1:]:  # Skip the first category to avoid multicollinearity
            df[f"Contract_{category}"] = (df["Contract"] == category).astype(int)

        for category in PaymentMethodsDummy[1:]:  # Skip the first category to avoid multicollinearity
            df[f"PaymentMethod_{category}"] = (df["PaymentMethod"] == category).astype(int)

        df = df.drop(columns=["InternetService", "Contract", "PaymentMethod"])

        scaler = self.standard_scaler
        columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        return df
    def Predict(self,data):
        processed_data = self.__data_preprocessing__(data)
        for cols in processed_data.columns:
            print(cols)
        model = self.rf_model
        y_pred = model.predict(processed_data.values)
        prediction = "The customer is likely to churn." if y_pred[0] == 1 else "The customer is not likely to churn."
        print(prediction)
        return prediction






