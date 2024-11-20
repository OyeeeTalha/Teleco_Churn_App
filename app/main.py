from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from routes.predict_route import predict_router
from routes.chatbot_route import chatbot_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


# Model for prediction data (you can modify this)
class PredictionRequest(BaseModel):
    data: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, initial_page: str = "prediction", prediction_result: str = "", chatbot_response: str = ""):
    """
    Render the main page with navigation options.
    """
    return templates.TemplateResponse("main.html", {
        "request": request,
        "initial_page": initial_page,
        "prediction_result": prediction_result,
        "chatbot_response": chatbot_response
    })

app.include_router(predict_router)
app.include_router(chatbot_router)

