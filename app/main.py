from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


# Model for prediction data (you can modify this)
class PredictionRequest(BaseModel):
    data: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main page with navigation options.
    """
    return templates.TemplateResponse("main.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, data: str = Form(...)):
    """
    Handle model prediction requests.
    """
    # Add your prediction logic here
    prediction = f"Prediction for input '{data}'"
    return templates.TemplateResponse(
        "main.html", {"request": request, "prediction_result": prediction}
    )



@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, prompt: str = Form(...)):
    """
    Handle chatbot prompt requests.
    """
    # Add your chatbot logic here
    response = f"Response to prompt '{prompt}'"
    return templates.TemplateResponse(
        "main.html", {"request": request, "chatbot_response": response}
    )

