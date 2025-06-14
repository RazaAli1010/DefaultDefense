from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


pipeline = joblib.load("model/pipeline.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    income: float = Form(...),
    loanAmount: float = Form(...),
    creditScore: float = Form(...),
    monthsEmployed: int = Form(...),
    numCreditLines: int = Form(...),
    interestRate: float = Form(...),
    loanTerm: int = Form(...),
    monthlyDebt: float = Form(...),
    education: str = Form(...),
    employmentType: str = Form(...),
    maritalStatus: str = Form(...),
    hasMortgage: str = Form(...),
    hasDependents: str = Form(...),
    loanPurpose: str = Form(...),
    hasCoSigner: str = Form(...)
):
    dti = monthlyDebt / income if income > 0 else 0

    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loanAmount,
        "CreditScore": creditScore,
        "MonthsEmployed": monthsEmployed,
        "NumCreditLines": numCreditLines,
        "InterestRate": interestRate,
        "LoanTerm": loanTerm,
        "DTIRatio": dti,
        "Education": education,
        "EmploymentType": employmentType,
        "MaritalStatus": maritalStatus,
        "HasMortgage": hasMortgage,
        "HasDependents": hasDependents,
        "LoanPurpose": loanPurpose,
        "HasCoSigner": hasCoSigner
    }])

    prediction = pipeline.predict(input_data)[0]

    if prediction == 1:
        status = "High Risk of Default"
        message = " Warning: The applicant is at high risk of loan default. Please evaluate carefully before approval."
        style = "warning"
    else:
        status = "Low Risk of Default"
        message = " Great news! The applicant is unlikely to default on the loan based on current data."
        style = "success"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": status,
        "message": message,
        "style": style
    })

