from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import logging
from typing import Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from crewai import Crew, Process
from agents import financial_analyst, verifier, investment_advisor, risk_assessor, coordinator, get_agents_for_query
from task import analyze_financial_document, investment_analysis, risk_assessment, verification
from tools import validate_file_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'.pdf'}
UPLOAD_DIR = "data"
OUTPUT_DIR = "outputs"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("Starting Financial Document Analyzer API...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")
    yield
    # Shutdown
    logger.info("Shutting down Financial Document Analyzer API...")
    # Cleanup old files on shutdown
    try:
        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

app = FastAPI(
    title="Financial Document Analyzer",
    description="""Advanced API for analyzing financial documents using multiple specialized AI agents.
    
    ## Features
    - 📄 **Document Verification**: Automatically verify if uploaded documents are legitimate financial reports
    - 📊 **Financial Analysis**: Extract and interpret key financial metrics and performance indicators
    - 💼 **Investment Recommendations**: Get balanced, data-driven investment advice
    - ⚠️ **Risk Assessment**: Comprehensive risk analysis across multiple dimensions
    - 🔄 **Multi-Agent Collaboration**: Four specialized agents work together for thorough analysis
    
    ## How It Works
    1. Upload a PDF financial document
    2. Optionally specify your query or focus area
    3. Choose between quick or detailed analysis
    4. Receive comprehensive insights from all agents
    
    ## Agent Team
    - **Verifier**: Validates document authenticity
    - **Financial Analyst**: Extracts and interprets financial data
    - **Investment Advisor**: Provides investment recommendations
    - **Risk Assessor**: Evaluates potential risks
    """,
    version="2.0.0",
    contact={
        "name": "VWO Internship Assignment",
        "email": "genai@vwo.com",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_crew(query: str, file_path: str, detailed: bool = True) -> str:
    """Run the crew with appropriate agents based on detail level"""
    
    # Select agents based on query and detail level
    if detailed:
        # Use all agents for comprehensive analysis
        agents = [verifier, financial_analyst, investment_advisor, risk_assessor]
        tasks = [verification, analyze_financial_document, investment_analysis, risk_assessment]
        logger.info("Running detailed analysis with all 4 agents")
    else:
        # Quick analysis with just verifier and financial analyst
        agents = [verifier, financial_analyst]
        tasks = [verification, analyze_financial_document]
        logger.info("Running quick analysis with verifier and financial analyst")
    
    # Check if query needs coordinator (complex queries)
    complex_keywords = ['compare', 'versus', 'vs', 'overview', 'comprehensive', 'synthesize']
    if any(keyword in query.lower() for keyword in complex_keywords):
        agents.insert(0, coordinator)
        logger.info("Added coordinator agent for complex query")
    
    # Create crew with selected agents and tasks
    financial_crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=True,
        cache=True,
        max_rpm=20,
        share_crew=False
    )
    
    # Define inputs for the crew
    inputs = {
        'query': query.strip(),
        'file_path': file_path
    }
    
    try:
        # Kickoff the crew
        logger.info(f"Starting crew analysis with {len(agents)} agents")
        result = financial_crew.kickoff(inputs=inputs)
        logger.info("Crew analysis completed successfully")
        return str(result)
    except Exception as e:
        logger.error(f"Crew execution failed: {str(e)}")
        raise

@app.get("/", 
         response_description="Health check endpoint",
         summary="Root endpoint",
         tags=["Health"])
async def root():
    """Health check endpoint returning API status and version"""
    return {
        "message": "Financial Document Analyzer API is running",
        "status": "operational",
        "version": app.version,
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "agents": ["Verifier", "Financial Analyst", "Investment Advisor", "Risk Assessor"]
    }

@app.get("/health",
         response_description="Detailed health check",
         summary="Health check for monitoring",
         tags=["Health"])
async def health_check():
    """Detailed health check endpoint for monitoring systems"""
    # Check if upload directory is writable
    upload_dir_writable = os.access(UPLOAD_DIR, os.W_OK) if os.path.exists(UPLOAD_DIR) else False
    
    # Check disk space (simple check)
    try:
        stat = os.statvfs(UPLOAD_DIR)
        free_space_mb = (stat.f_frsize * stat.f_bavail) / (1024 * 1024)
        disk_ok = free_space_mb > 100  # At least 100MB free
    except:
        disk_ok = False
        free_space_mb = 0
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "operational",
            "file_system": "operational" if upload_dir_writable else "degraded",
            "disk_space_mb": round(free_space_mb, 2),
            "disk_ok": disk_ok
        },
        "config": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "upload_dir": UPLOAD_DIR,
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        }
    }

@app.post("/analyze",
          response_description="Financial analysis results",
          summary="Analyze a financial document",
          tags=["Analysis"])
async def analyze_document(
    file: UploadFile = File(..., description="Financial document to analyze (PDF format, max 10MB)"),
    query: str = Form(default="Analyze this financial document and provide comprehensive insights including key metrics, investment opportunities, and risk factors"),
    detailed_analysis: bool = Form(default=True, description="If True, runs all 4 agents for comprehensive analysis. If False, runs quick analysis with 2 agents.")
):
    """
    Analyze a financial document and get comprehensive insights from multiple AI agents.
    
    ### Parameters:
    - **file**: PDF file containing the financial document (required, max 10MB)
    - **query**: Specific questions or focus areas for the analysis (optional)
    - **detailed_analysis**: 
      - `True` (default): Runs all 4 agents for comprehensive analysis
      - `False`: Runs quick analysis with verifier and financial analyst only
    
    ### Returns:
    - **status**: Success/failure status
    - **query**: The query that was analyzed
    - **analysis**: Comprehensive analysis from all agents
    - **file_processed**: Name of the processed file
    - **file_id**: Unique identifier for the analysis
    - **analysis_type**: "detailed" or "quick"
    - **agents_used**: List of agents that participated
    - **timestamp**: When the analysis was performed
    
    ### Example Response:
    ```json
    {
        "status": "success",
        "query": "Analyze revenue trends and risks",
        "analysis": "Document verified as Tesla Q2 2025 Update...",
        "file_processed": "TSLA-Q2-2025-Update.pdf",
        "file_id": "550e8400-e29b-41d4-a716-446655440000",
        "analysis_type": "detailed",
        "agents_used": ["Verifier", "Financial Analyst", "Investment Advisor", "Risk Assessor"],
        "timestamp": "2025-02-26T15:30:00Z"
    }
    ```
    """
    
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    
    logger.info(f"Request {request_id}: Received analysis request for file: {file.filename}")
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Request {request_id}: Invalid file type: {file_ext}")
        raise HTTPException(
            status_code=400, 
            detail=f"Only PDF files are supported. Received: {file_ext}"
        )
    
    # Validate file size
    try:
        # Get file size by reading first chunk
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.warning(f"Request {request_id}: File too large: {file_size} bytes")
            raise HTTPException(
                status_code=400, 
                detail=f"File size too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
        logger.info(f"Request {request_id}: File size: {file_size} bytes")
    except Exception as e:
        logger.error(f"Request {request_id}: Error reading file size: {e}")
        raise HTTPException(status_code=400, detail="Could not read file size")
    
    # Create unique file ID and path
    file_id = str(uuid.uuid4())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    
    try:
        # Save uploaded file
        logger.info(f"Request {request_id}: Saving file to {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Set default query if empty
        if not query or query.strip() == "":
            query = "Analyze this financial document and provide comprehensive insights including key metrics, investment opportunities, and risk factors"
            logger.info(f"Request {request_id}: Using default query")
        
        # Log analysis parameters
        logger.info(f"Request {request_id}: Processing with parameters:")
        logger.info(f"  - Query: {query[:100]}...")
        logger.info(f"  - Detailed analysis: {detailed_analysis}")
        
        # Process the financial document
        response = run_crew(
            query=query.strip(), 
            file_path=file_path,
            detailed=detailed_analysis
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Request {request_id}: Analysis completed in {processing_time:.2f} seconds")
        
        # Determine which agents were used
        agents_used = ["Verifier", "Financial Analyst"]
        if detailed_analysis:
            agents_used.extend(["Investment Advisor", "Risk Assessor"])
        
        # Prepare response
        return {
            "status": "success",
            "request_id": request_id,
            "query": query,
            "analysis": str(response),
            "file_processed": file.filename,
            "file_id": file_id,
            "analysis_type": "detailed" if detailed_analysis else "quick",
            "agents_used": agents_used,
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error and raise HTTP exception
        logger.error(f"Request {request_id}: Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing financial document: {str(e)}"
        )
    
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Request {request_id}: Cleaned up file: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Request {request_id}: Error cleaning up file: {cleanup_error}")

@app.get("/agents",
         response_description="List of available agents",
         summary="Get information about AI agents",
         tags=["Information"])
async def list_agents():
    """Get information about all available AI agents and their capabilities"""
    return {
        "agents": [
            {
                "name": "Financial Document Verifier",
                "role": "Verification Expert",
                "capabilities": [
                    "Document type identification",
                    "Authenticity verification",
                    "Section extraction",
                    "Compliance checking"
                ],
                "priority": "Always runs first"
            },
            {
                "name": "Senior Financial Analyst",
                "role": "Financial Analysis",
                "capabilities": [
                    "Financial metric extraction",
                    "Performance analysis",
                    "Trend identification",
                    "Ratio analysis"
                ],
                "priority": "Core analysis"
            },
            {
                "name": "Investment Advisor (CFA)",
                "role": "Investment Recommendations",
                "capabilities": [
                    "Valuation analysis",
                    "Buy/Hold/Sell recommendations",
                    "Portfolio implications",
                    "Growth assessment"
                ],
                "priority": "Detailed analysis only"
            },
            {
                "name": "Chief Risk Officer",
                "role": "Risk Assessment",
                "capabilities": [
                    "Financial risk identification",
                    "Operational risk assessment",
                    "Regulatory compliance",
                    "Risk mitigation strategies"
                ],
                "priority": "Detailed analysis only"
            },
            {
                "name": "Financial Analysis Coordinator",
                "role": "Workflow Coordination",
                "capabilities": [
                    "Complex query handling",
                    "Multi-agent coordination",
                    "Result synthesis"
                ],
                "priority": "Complex queries only"
            }
        ],
        "total_agents": 5,
        "analysis_modes": {
            "quick": ["Verifier", "Financial Analyst"],
            "detailed": ["Verifier", "Financial Analyst", "Investment Advisor", "Risk Assessor"],
            "complex": ["Coordinator", "Verifier", "Financial Analyst", "Investment Advisor", "Risk Assessor"]
        }
    }

@app.get("/stats",
         response_description="API statistics",
         summary="Get API usage statistics",
         tags=["Information"])
async def get_stats():
    """Get basic API statistics (implement with database for production)"""
    # This is a placeholder - implement with actual database in production
    return {
        "status": "operational",
        "total_analyses": "N/A (tracking not implemented)",
        "avg_response_time": "N/A",
        "success_rate": "N/A",
        "uptime": "N/A",
        "note": "Implement database integration for production statistics"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )