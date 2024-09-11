from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.cv_service import extract_keywords_from_cv
from app.services.workig_with_cv_data import categorize_skills
import os

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Adjust this to include the URL of your frontend application
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specifies the origins that are allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

router = APIRouter()

@router.get("/analyze-cv/", response_model=dict)
async def analyze_cv():
    file_path = 'C:\\dev\\git\\uploaded-docs\\cv.pdf'
    if not os.path.exists(file_path) or not file_path.endswith('.pdf'):
        raise HTTPException(status_code=404, detail="File not found or not a PDF.")
    
    keywords = await extract_keywords_from_cv(file_path)
    # categories = categorize_skills(keywords,'svm')
    categories = categorize_skills(keywords,'logistic')

    # return list(keywords)
    return categories

# Include the router in your FastAPI app
app.include_router(router)
