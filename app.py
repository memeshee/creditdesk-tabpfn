"""CreditDesk demo web app: fill the form, TabPFN-3.5 decides. Run: uvicorn app:app --port 8321"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from creditdesk.agent import underwrite_one

app = FastAPI(title="CreditDesk — TabPFN-3.5 SME underwriting")


class Applicant(BaseModel):
    business_description: str = "noodle cart near bus station, cash only"
    sector: str = "street food stall"
    region: str = "central"
    annual_revenue_usd: float = 18000
    years_in_business: int = 1
    debt_to_income: float = 0.6
    late_payments_12m: int = 2
    avg_monthly_cashflow_usd: float = 900
    employees: int = 1
    requested_loan_usd: float = 6000
    thinking: bool = False


@app.post("/api/underwrite")
def api_underwrite(a: Applicant):
    d = a.model_dump()
    thinking = d.pop("thinking")
    return underwrite_one(d, thinking=thinking)


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>CreditDesk — TabPFN-3.5 SME underwriting</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}label{display:block;margin:.4rem 0}.row{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}input,select{width:100%;padding:.4rem}button{padding:.6rem 1.2rem;font-size:1rem;margin-top:.6rem}#out{background:#f6f6f6;padding:1rem;white-space:pre-wrap;margin-top:1rem}.pill{display:inline-block;padding:.2rem .7rem;border-radius:1rem;color:#fff}</style></head><body>
<h1>CreditDesk</h1><p>Thin-file SME underwriting powered by <b>TabPFN-3.5</b>. No training pipeline — 120 portfolio rows in context, instant posterior.</p>
<div class=row>
<label>Description<input id=business_description value="noodle cart near bus station, cash only"></label>
<label>Sector<select id=sector><option>street food stall</option><option>motorbike repair shop</option><option>tailor shop</option><option>phone accessories kiosk</option><option>smallholder coffee farm</option><option>hair salon</option><option>corner grocery</option><option>furniture workshop</option></select></label>
<label>Region<select id=region><option>central</option><option>north</option><option>south</option><option>highlands</option></select></label>
<label>Annual revenue USD<input id=annual_revenue_usd type=number value=18000></label>
<label>Years in business<input id=years_in_business type=number value=1></label>
<label>Debt/income<input id=debt_to_income type=number step=0.01 value=0.6></label>
<label>Late payments 12m<input id=late_payments_12m type=number value=2></label>
<label>Monthly cashflow<input id=avg_monthly_cashflow_usd type=number value=900></label>
<label>Employees<input id=employees type=number value=1></label>
<label>Requested loan<input id=requested_loan_usd type=number value=6000></label>
</div>
<label><input id=thinking type=checkbox> thinking mode (extra compute)</label>
<button onclick="go()">Underwrite</button><div id=out>…</div>
<script>async function go(){const ids=["business_description","sector","region","annual_revenue_usd","years_in_business","debt_to_income","late_payments_12m","avg_monthly_cashflow_usd","employees","requested_loan_usd"];const b={};ids.forEach(i=>{let v=document.getElementById(i).value;if(!isNaN(+v)&&v.trim()!==""&&isNaN(document.getElementById(i).value[0]||"x")&&i!=="business_description"&&i!=="sector"&&i!=="region")v=+v;b[i]=v});b.thinking=document.getElementById("thinking").checked;
const r=await fetch("/api/underwrite",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});const j=await r.json();
const c=j.decision==="APPROVE"?"green":j.decision==="REVIEW"?"orange":"crimson";
document.getElementById("out").innerHTML=`<span class=pill style="background:${c}">${j.decision}</span> default ${(j.default_probability*100).toFixed(1)}% · expected loss $${j.expected_loss_usd} · margin $${j.expected_margin_usd}\\n${j.terms}\\n\\n`+j.rationale.map(x=>"• "+x).join("\\n")}</script>
</body></html>"""
