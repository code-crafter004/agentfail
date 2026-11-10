import os
from dotenv import load_dotenv
load_dotenv(".env")
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

models = {
    "gemini": ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        api_version="v1beta", temperature=0),
    "groq": ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        temperature=0),
}
for name, m in models.items():
    try:
        r = m.bind_tools([add]).invoke("What is 17 + 25? Use the tool.")
        print(name, "OK", r.tool_calls)
    except Exception as e:
        print(name, "FAIL", repr(e)[:300])
