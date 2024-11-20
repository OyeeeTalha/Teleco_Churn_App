from fastapi import APIRouter, Form , Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from services.random_forest_predict import RandomForestPredict

predict_router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@predict_router.post("/predict/")
async def predict(request: Request,
    gender: str = Form(...),
    partner: str = Form(...),
    dependents: str = Form(...),
    phone_service: str = Form(...),
    multiple_lines: str = Form(...),
    online_security: str = Form(...),
    online_backup: str = Form(...),
    device_protection: str = Form(...),
    tech_support: str = Form(...),
    streaming_tv: str = Form(...),
    streaming_movies: str = Form(...),
    internet_service: str = Form(...),
    contract: str = Form(...),
    paperless_billing: str = Form(...),
    payment_method: str = Form(...),
    senior_citizen: int = Form(...),
    tenure: int = Form(...),
    monthly_charges: float = Form(...),
    total_charges: float = Form(...)
):
    user_response = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    # Combine all form data into a dictionary for easier processing

    # Example: Add a dummy prediction logic here
    random_forest = RandomForestPredict()
    prediction = "The Customer is Going to " + random_forest.Predict(user_response)

    # Return the user data and the prediction
    # return JSONResponse(content={"user_response": user_response, "prediction": prediction})
    return templates.TemplateResponse("main.html", {
        "request": request,
        "initial_page": 'prediction',
        "prediction_result": prediction,
        "chatbot_response": ""
    })


