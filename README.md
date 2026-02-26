```markdown
# Financial Document Analyzer

An AI-powered financial document analysis system that processes corporate reports, financial statements, and investment documents using multiple specialized agents powered by CrewAI.

## 🚀 Features

- **Multi-Agent Analysis**: Four specialized AI agents work together:
  - **Document Verifier**: Validates if uploaded documents are legitimate financial reports
  - **Financial Analyst**: Extracts and interprets key financial metrics and performance indicators
  - **Investment Advisor**: Provides balanced investment recommendations based on actual data
  - **Risk Assessor**: Evaluates potential risks and provides comprehensive risk profiles

- **PDF Processing**: Extract and analyze text from financial PDF documents
- **REST API**: Easy-to-use FastAPI endpoints with automatic documentation
- **Comprehensive Analysis**: Get complete financial insights in one response from multiple perspectives
- **File Management**: Automatic cleanup of uploaded files after processing

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key (required)
- (Optional) Serper API key for web search capabilities

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd financial-document-analyzer-debug
```

2. **Create and activate virtual environment**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Create `.env` file** in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional: Add Serper API key for web search
# SERPER_API_KEY=your_serper_api_key_here
```

## 🐛 Bugs Fixed

The original codebase had several critical bugs that have been fixed:

| Bug | Issue | Fix |
|-----|-------|-----|
| **PDF Reader** | Missing PdfReader import, incorrect method calls | Added proper pypdf implementation with error handling |
| **Agent Personalities** | Unprofessional, jokey agent descriptions | Redesigned with professional roles and realistic behaviors |
| **Tool Methods** | Async methods without proper implementation | Properly implemented static methods with error handling |
| **Task Definitions** | Vague, unstructured tasks with poor output format | Clear, structured tasks with specific expected outputs |
| **Main Application** | Missing imports, no file validation, poor error handling | Complete FastAPI implementation with proper validation |
| **Requirements** | Missing critical dependencies | Added pypdf, python-multipart, and other required packages |
| **File Cleanup** | Inefficient cleanup with silent failures | Proper cleanup with error logging |

## 🚦 Running the Application

1. **Start the server**:
```bash
python main.py
```
The server will start at `http://localhost:8000`

2. **Access API documentation**:
- Swagger UI: `http://localhost:8000/docs` (interactive testing)
- ReDoc: `http://localhost:8000/redoc` (alternative documentation)

## 📚 API Documentation

### Health Check Endpoint
```http
GET /
```
Returns API status and version information.

**Response**:
```json
{
  "message": "Financial Document Analyzer API is running",
  "status": "healthy",
  "version": "1.0.0"
}
```

### Analyze Document Endpoint
```http
POST /analyze
Content-Type: multipart/form-data
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | PDF file containing financial document |
| `query` | String | No | Specific questions or focus areas for analysis |
| `detailed_analysis` | Boolean | No | If true, runs all agents for comprehensive analysis |

**Example Request using curl**:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -F "file=@TSLA-Q2-2025-Update.pdf" \
  -F "query=What are the key revenue trends and risk factors?" \
  -F "detailed_analysis=true"
```

**Example Response**:
```json
{
  "status": "success",
  "query": "What are the key revenue trends and risk factors?",
  "analysis": "Document Verification: ✓ Valid financial document identified as Tesla Q2 2025 Update\n\nKey Financial Metrics:\n- Total revenue: $22.5B (12% decrease YoY)\n- Operating income: $0.9B (42% decrease YoY)\n- Operating margin: 4.1%\n- Cash and investments: $36.8B\n\nInvestment Analysis: [Detailed investment recommendations...]\n\nRisk Assessment: [Comprehensive risk analysis...]",
  "file_processed": "TSLA-Q2-2025-Update.pdf",
  "file_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Health Check Endpoint (Monitoring)
```http
GET /health
```
For monitoring and health checks.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "api": "operational"
}
```

## 📁 Project Structure

```
financial-document-analyzer-debug/
├── main.py              # FastAPI application with endpoints
├── agents.py            # AI agent definitions and configurations
├── task.py              # Task definitions for each agent
├── tools.py             # Custom tools (PDF reader, investment tools, risk tools)
├── requirements.txt     # Project dependencies
├── .env                 # Environment variables (API keys)
├── README.md            # This documentation
├── data/                # Temporary storage for uploaded files (auto-created)
└── outputs/             # Analysis outputs directory (auto-created)
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for GPT models |
| `SERPER_API_KEY` | No | Serper API key for web search (optional) |

## 🎯 Usage Examples

### Python Client Example
```python
import requests

# API endpoint
url = "http://localhost:8000/analyze"

# Prepare the file and data
files = {"file": open("quarterly_report.pdf", "rb")}
data = {
    "query": "Analyze revenue growth and profit margins for Q2",
    "detailed_analysis": "true"
}

# Send request
response = requests.post(url, files=files, data=data)

# Print results
print(response.json())

# Don't forget to close the file
files["file"].close()
```

### JavaScript Client Example
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('query', 'Analyze this quarterly report');
formData.append('detailed_analysis', 'true');

fetch('http://localhost:8000/analyze', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

## 🧪 Testing the API

### Using Swagger UI
1. Navigate to `http://localhost:8000/docs`
2. Click on the `/analyze` endpoint
3. Click "Try it out"
4. Upload a PDF file and add a query
5. Click "Execute" to see the response

### Using Python requests
```python
import requests

# Test health endpoint
health = requests.get("http://localhost:8000/health")
print("Health:", health.json())

# Test analysis endpoint
with open("test_document.pdf", "rb") as f:
    files = {"file": f}
    data = {"query": "Test analysis"}
    response = requests.post("http://localhost:8000/analyze", files=files, data=data)
    print("Analysis:", response.json())
```

## 🚀 Deployment on Render.com

### Step-by-Step Deployment Guide

1. **Push code to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit with fixed code"
git remote add origin https://github.com/yourusername/financial-document-analyzer.git
git push -u origin main
```

2. **Create a new Web Service on Render**:
   - Go to [Render.com](https://render.com) and sign in
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

3. **Configure the service**:

| Setting | Value |
|---------|-------|
| **Name** | `financial-document-analyzer` |
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Plan** | Free (or choose paid for more resources) |

4. **Add Environment Variables**:
   - Click "Environment" tab
   - Add `OPENAI_API_KEY` with your API key
   - Add any other optional keys

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for the build to complete
   - Your API will be available at `https://your-app-name.onrender.com`

### Important Notes for Render Deployment
- The free tier spins down after 15 minutes of inactivity
- First request after inactivity may take 30-60 seconds to respond
- Files are stored temporarily and cleaned up automatically
- Maximum file size is limited by Render's free tier (512 MB)

## ⚠️ Important Notes

- **API Key Required**: Ensure your OpenAI API key has sufficient credits
- **File Types**: Only PDF files are supported
- **File Cleanup**: Uploaded files are automatically deleted after analysis
- **Rate Limiting**: Consider implementing rate limiting for production use
- **Authentication**: Add API key authentication for production deployment
- **Error Handling**: All errors are logged and returned with appropriate HTTP status codes

## 🔒 Security Considerations

For production deployment, consider:
- Adding API key authentication
- Implementing rate limiting
- Using HTTPS only
- Validating file size and content type
- Scanning uploaded files for malware
- Storing analysis results in a database

## 📝 License

This project is created as part of the VWO (Wingify) Intern - Generative AI assignment. All rights reserved.

## 🤝 Support

For issues or questions:
- Email: genai@vwo.com
- CC: vipul.kumar@vwo.com, gstn.shastri_v@vwo.com

## 🙏 Acknowledgments

- CrewAI for the agent framework
- FastAPI for the web framework
- OpenAI for language models
- VWO for the internship opportunity

---

**Happy Debugging! 🐛✨** 
```

This README is comprehensive and includes:
- All setup instructions
- Fixed bugs documentation
- API documentation
- Usage examples
- Deployment guide for Render.com
- Security considerations
- Contact information

Just update the GitHub repository URL in the clone command and you're ready to go!