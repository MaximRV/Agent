from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import BankDepositModel

router = APIRouter()
templates = Jinja2Templates(directory="templates")
model = BankDepositModel()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, term: int = Form(...)):
    predicted_rate = model.predict_rate(term)
    recommendations = model.get_recommendations(term)

    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "rate": predicted_rate,
        "recommendations": recommendations
    })
