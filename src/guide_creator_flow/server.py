from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Dict
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from .main import GuideOutline, ContentCrew

# Load environment variables
load_dotenv()

app = FastAPI(title="Guide Creator API")

class GuideRequest(BaseModel):
    topic: str
    target_audience: str

# Store for tracking running tasks
tasks_store: Dict[str, str] = {}

@app.get("/")
async def root():
    return {"message": "Guide Creator API is running"}

@app.post("/create-guide")
async def create_guide(request: GuideRequest, background_tasks: BackgroundTasks):
    try:
        # Initialize the content crew
        crew = ContentCrew()
        
        # Create initial guide outline
        outline = GuideOutline(
            title=f"Comprehensive Guide to {request.topic}",
            introduction="",  # Will be filled by the crew
            target_audience=request.target_audience,
            sections=[],  # Will be filled by the crew
            conclusion=""  # Will be filled by the crew
        )
        
        # Run the crew task in the background
        background_tasks.add_task(crew.run, outline)
        
        return {
            "status": "Guide creation task started successfully",
            "topic": request.topic,
            "target_audience": request.target_audience
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)