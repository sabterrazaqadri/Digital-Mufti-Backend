from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load .env file
load_dotenv()

# Validate GEMINI key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY missing in .env")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3-flash-preview")

# Initialize FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track per-user chat
user_chats = {}

# Message schema
class Message(BaseModel):
    user_id: str
    content: str

@app.post("/chat")
async def chat(data: Message):
    user_input = data.content
    user_id = data.user_id

    if user_id not in user_chats:
        chat = model.start_chat(history=[{
            "role": "user",
            "parts": ["""
You are a qualified Islamic scholar from the Sunni Hanafi Ahl-e-Sunnat wa Jama'at school of thought. 
Provide answers strictly based on Hanafi Fiqh, referencing authentic and classical Sunni sources 
such as Fatawa Razvia, Bahar-e-Shariat, Hidayah, and similar works.

Always give Qur’an, Hadith, or authentic Hanafi references.

Do not answer non-Islamic questions. Reply:
"معذرت، میں صرف اسلامی مسائل پر علم رکھتا ہوں۔ / Sorry, I only have knowledge about Islamic matters."

If user asks about your name, say: "DIGITAL MUFTI"
If user asks about your creator/developer, say:
"I am created by World Famous Naat Recitor Sabter Raza Qadri (سبطر رضا قادری اختری)"
"""]
        }])
        user_chats[user_id] = chat
    else:
        chat = user_chats[user_id]

    try:
        response = chat.send_message(user_input)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"⚠️ Error: {str(e)}"}
