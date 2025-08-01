# SpamSlam Real AI Backend
# Install: pip install fastapi uvicorn google-generativeai python-multipart

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional
import json
import re

app = FastAPI(title="SpamSlam AI Backend", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
genai.configure(api_key=AIzaSyAKAQWRYQQIE6x68Zz7X2Yu2lIgXU4hO2k)
model = genai.GenerativeModel('gemini-pro')

# Request/Response models
class SpamAnalysisRequest(BaseModel):
    transcript: str

class SpamAnalysisResponse(BaseModel):
    spam_type: str
    confidence: float
    sarcastic_reply: str
    explanation: Optional[str] = None

# Spam detection prompt
DETECTION_PROMPT = """
Analyze this phone call transcript and determine if it's spam. If it is spam, classify the type.

Transcript: "{transcript}"

Respond with a JSON object containing:
- "is_spam": boolean
- "spam_type": string (one of: "loan_scam", "lottery_scam", "tech_support_scam", "warranty_scam", "irs_scam", "insurance_scam", "robocall", "telemarketing", "other_spam", "not_spam")
- "confidence": float (0.0 to 1.0)
- "key_indicators": list of strings explaining why this is spam

Examples of spam types:
- loan_scam: Personal loans, debt consolidation, credit offers
- lottery_scam: Prize winnings, lottery, sweepstakes
- tech_support_scam: Computer viruses, Microsoft/Apple support
- warranty_scam: Car warranty, extended warranty
- irs_scam: Tax issues, government threats
- insurance_scam: Health/life insurance offers
- robocall: Automated messages, press 1 to continue
- telemarketing: Sales calls, product offers
- other_spam: Other suspicious calls
- not_spam: Legitimate calls

Only respond with valid JSON, no additional text.
"""

# Sarcastic response generation prompt
SARCASM_PROMPT = """
Generate a witty, sarcastic response to this spam call. The response should be:
- Funny and clever
- Obviously sarcastic
- Safe for work
- 1-2 sentences maximum
- Exposes the absurdity of the scam

Spam type: {spam_type}
Original transcript: "{transcript}"

Make it specific to the type of scam. Examples:
- For loan scams: Reference ridiculous investments or financial situations
- For lottery scams: Mock the "random winner" concept
- For tech support: Play dumb about technology
- For warranty scams: Reference absurd vehicles

Just return the sarcastic response, nothing else.
"""

@app.get("/")
async def root():
    return {"message": "SpamSlam AI Backend is running! 🤖📞"}

@app.post("/analyze-spam", response_model=SpamAnalysisResponse)
async def analyze_spam(request: SpamAnalysisRequest):
    """
    Analyze transcript for spam and generate sarcastic response
    """
    try:
        transcript = request.transcript.strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")
        
        # Step 1: Detect spam type using Gemini
        detection_prompt = DETECTION_PROMPT.format(transcript=transcript)
        
        try:
            detection_response = model.generate_content(detection_prompt)
            detection_text = detection_response.text.strip()
            
            # Clean up response (remove markdown formatting if present)
            detection_text = re.sub(r'```json\n?|```\n?', '', detection_text)
            
            # Parse JSON response
            detection_data = json.loads(detection_text)
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to simple classification if Gemini fails
            detection_data = simple_spam_detection(transcript)
        
        if not detection_data.get("is_spam", False):
            return SpamAnalysisResponse(
                spam_type="not_spam",
                confidence=detection_data.get("confidence", 0.0),
                sarcastic_reply="This actually sounds legitimate! No sarcasm needed.",
                explanation="Not detected as spam"
            )
        
        # Step 2: Generate sarcastic response
        spam_type = detection_data.get("spam_type", "other_spam")
        sarcasm_prompt = SARCASM_PROMPT.format(
            spam_type=spam_type,
            transcript=transcript
        )
        
        try:
            sarcasm_response = model.generate_content(sarcasm_prompt)
            sarcastic_reply = sarcasm_response.text.strip()
            
            # Clean up quotes if present
            sarcastic_reply = sarcastic_reply.strip('"\'')
            
        except Exception as e:
            # Fallback responses
            sarcastic_reply = get_fallback_response(spam_type)
        
        return SpamAnalysisResponse(
            spam_type=spam_type,
            confidence=detection_data.get("confidence", 0.8),
            sarcastic_reply=sarcastic_reply,
            explanation=f"Detected as {spam_type.replace('_', ' ')} with {len(detection_data.get('key_indicators', []))} indicators"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def simple_spam_detection(transcript: str) -> dict:
    """
    Fallback spam detection using keywords
    """
    transcript_lower = transcript.lower()
    
    patterns = {
        "loan_scam": ["loan", "credit", "debt", "money", "finance", "interest", "borrow"],
        "lottery_scam": ["won", "winner", "prize", "lottery", "congratulations", "claim"],
        "tech_support_scam": ["computer", "virus", "microsoft", "technical", "support", "infected"],
        "warranty_scam": ["warranty", "car", "vehicle", "auto", "expire", "coverage"],
        "irs_scam": ["irs", "tax", "government", "legal", "arrest", "warrant"],
        "insurance_scam": ["insurance", "health", "medical", "coverage", "policy"]
    }
    
    best_match = "other_spam"
    max_matches = 0
    
    for spam_type, keywords in patterns.items():
        matches = sum(1 for keyword in keywords if keyword in transcript_lower)
        if matches > max_matches:
            max_matches = matches
            best_match = spam_type
    
    is_spam = max_matches > 0 or any(suspicious in transcript_lower for suspicious in 
                                    ["call now", "limited time", "act fast", "free", "guaranteed"])
    
    return {
        "is_spam": is_spam,
        "spam_type": best_match if is_spam else "not_spam",
        "confidence": min(0.9, max_matches * 0.2 + 0.3) if is_spam else 0.1,
        "key_indicators": [f"Found {max_matches} matching keywords"] if is_spam else []
    }

def get_fallback_response(spam_type: str) -> str:
    """
    Fallback sarcastic responses if AI generation fails
    """
    fallbacks = {
        "loan_scam": "Oh fantastic! I was just thinking of investing in invisible yachts. Do you finance those?",
        "lottery_scam": "OMG I won?! This is amazing! I didn't even know I entered the 'Random Stranger Phone Call' lottery!",
        "tech_support_scam": "Oh no, my computer has a virus? That's weird, I'm talking to you on a toaster.",
        "warranty_scam": "My car warranty? Oh good! I was wondering when someone would call about my horse and buggy.",
        "irs_scam": "The IRS? Oh no! Should I be worried about my lemonade stand from 1995?",
        "insurance_scam": "Health insurance? Perfect! Does it cover my crippling addiction to online shopping?",
        "other_spam": "This sounds super legit and not at all like something I should hang up on immediately."
    }
    
    return fallbacks.get(spam_type, fallbacks["other_spam"])

@app.post("/generate-new-response")
async def generate_new_response(request: SpamAnalysisRequest):
    """
    Generate a new sarcastic response for the same transcript
    """
    # Re-analyze to get a potentially different response
    return await analyze_spam(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_available": bool(GEMINI_API_KEY != "your-gemini-api-key-here")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# To run:
# 1. Get Gemini API key from Google AI Studio
# 2. Set environment variable: export GEMINI_API_KEY=your-actual-key
# 3. Install dependencies: pip install fastapi uvicorn google-generativeai
# 4. Run: python main.py
# 5. API will be available at http://localhost:8000
