from fastapi import APIRouter , Form ,Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from services.chatbot import Chatbot


chatbot_router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@chatbot_router.post("/chat")
async def chat(request : Request,prompt: str = Form(...)):

    chatbot = Chatbot()
    response = chatbot.Query_Chatbot(prompt)
    # return JSONResponse(content={"user_prompt": prompt, "response": response})
    return templates.TemplateResponse("main.html", {
        "request": request,
        "initial_page": 'chatbot',
        "prediction_result": "",
        "chatbot_response": response
    })