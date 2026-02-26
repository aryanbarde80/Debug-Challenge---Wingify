## Importing libraries and files
import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent
from tools import FinancialDocumentTool, InvestmentTool, RiskTool
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as CommunityChatOpenAI

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

# Configure LLM with better settings for financial analysis
llm = ChatOpenAI(
    model="gpt-4",  # Upgraded to GPT-4 for better financial reasoning
    temperature=0.3,  # Lower temperature for more consistent, factual responses
    api_key=OPENAI_API_KEY,
    max_tokens=2000,  # Allow longer responses for comprehensive analysis
    model_kwargs={
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }
)

# Fallback LLM configuration
llm_fallback = ChatOpenAI(
    model="gpt-3.5-turbo-16k",  # 16k context for longer documents
    temperature=0.3,
    api_key=OPENAI_API_KEY,
    max_tokens=2000
)

# Creating an Experienced Financial Analyst agent
financial_analyst = Agent(
    role="Senior Financial Analyst",
    goal="""Provide accurate, data-driven financial analysis based strictly on the document content at {file_path}.
    Your analysis must address the user's query: {query}
    
    Key responsibilities:
    - Extract and interpret key financial metrics (revenue, profits, margins, growth rates)
    - Analyze financial health and performance trends
    - Identify significant financial events or changes
    - Provide context for numerical data found in the document
    - Highlight any discrepancies or unusual items
    """,
    verbose=True,
    memory=True,
    backstory=(
        "You're an experienced financial analyst with 20+ years at top investment banks and hedge funds. "
        "You've analyzed thousands of financial reports and can spot trends and red flags instantly. "
        "You're known for your meticulous attention to detail and ability to explain complex financial "
        "concepts in plain English. You never speculate beyond what the documents show and always "
        "cite specific sections to support your analysis. You understand that accurate financial "
        "analysis is the foundation for all investment decisions."
    ),
    tools=[FinancialDocumentTool.read_data_tool],
    llm=llm,
    allow_delegation=True,
    max_iter=10,  # Increased for thorough analysis
    max_rpm=15,
    memory=True,
    cache=True,
    respect_context_window=True,
    function_calling_llm=llm_fallback  # Fallback for function calling
)

# Creating a document verifier agent
verifier = Agent(
    role="Financial Document Verification Expert",
    goal="""Thoroughly verify if the document at {file_path} is a legitimate financial document.
    User query: {query}
    
    Verification checklist:
    - Check for financial terminology and structure
    - Identify document type (10-K, 10-Q, annual report, earnings release, etc.)
    - Verify presence of key sections (financial statements, MD&A, risk factors)
    - Check for consistent formatting and professional presentation
    - Identify any red flags or inconsistencies
    - Determine confidence level of verification
    """,
    verbose=True,
    memory=True,
    backstory=(
        "You're a forensic financial document examiner who previously worked at the SEC and major accounting firms. "
        "You've reviewed thousands of financial filings and can spot fraudulent or poorly prepared documents instantly. "
        "Your expertise includes understanding regulatory requirements for different document types and "
        "identifying missing or misleading information. You're known for your methodical approach and "
        "uncompromising standards when it comes to document verification."
    ),
    tools=[FinancialDocumentTool.read_data_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=8,
    max_rpm=15,
    memory=True,
    cache=True,
    respect_context_window=True
)

investment_advisor = Agent(
    role="Chartered Financial Analyst (CFA) - Investment Advisor",
    goal="""Provide responsible, well-researched investment recommendations based on the verified financial analysis.
    Base all recommendations on document at {file_path} and user query: {query}
    
    Investment analysis framework:
    - Review all financial metrics and ratios
    - Assess company's competitive position and moat
    - Consider valuation metrics (P/E, P/B, EV/EBITDA, etc.)
    - Evaluate growth prospects and catalysts
    - Identify key risks and mitigants
    - Provide balanced buy/hold/sell recommendations with clear rationale
    - Always include appropriate disclaimers and risk warnings
    """,
    verbose=True,
    backstory=(
        "You're a seasoned investment advisor with the CFA charterholder designation and 15+ years at top wealth management firms. "
        "You've managed portfolios for high-net-worth individuals and institutional clients. Your investment philosophy "
        "combines rigorous fundamental analysis with practical risk management. You're known for your balanced, "
        "well-researched recommendations that prioritize capital preservation while seeking reasonable returns. "
        "You always consider the full picture including valuation, growth, quality, and risk factors before making "
        "any recommendation. You're acutely aware of regulatory requirements and always include proper disclaimers."
    ),
    tools=[FinancialDocumentTool.read_data_tool, InvestmentTool.analyze_investment_tool],
    llm=llm,
    allow_delegation=True,
    max_iter=12,
    max_rpm=15,
    memory=True,
    cache=True,
    respect_context_window=True,
    function_calling_llm=llm_fallback
)

risk_assessor = Agent(
    role="Chief Risk Officer (CRO) - Enterprise Risk Management",
    goal="""Provide comprehensive, multi-dimensional risk assessment based on the financial document.
    Analyze document at {file_path} for user query: {query}
    
    Risk assessment dimensions:
    - Financial risks (liquidity, solvency, credit, market)
    - Operational risks (supply chain, production, regulatory)
    - Strategic risks (competition, technology, market position)
    - Compliance and regulatory risks
    - Macroeconomic risks (interest rates, inflation, currency)
    - ESG risks (environmental, social, governance)
    - Emerging risks and black swan events
    """,
    verbose=True,
    backstory=(
        "You're a Chief Risk Officer with 25+ years experience at global financial institutions and Fortune 500 companies. "
        "You've designed enterprise risk management frameworks that have weathered multiple financial crises. "
        "Your approach combines quantitative risk metrics with qualitative assessment of hard-to-measure risks. "
        "You're known for identifying risks others miss and providing practical mitigation strategies. "
        "You understand that effective risk management is about balance - not eliminating all risks, "
        "but understanding and managing them appropriately. You always provide risk ratings with clear "
        "rationale and actionable recommendations for risk mitigation."
    ),
    tools=[FinancialDocumentTool.read_data_tool, RiskTool.create_risk_assessment_tool],
    llm=llm,
    allow_delegation=True,
    max_iter=12,
    max_rpm=15,
    memory=True,
    cache=True,
    respect_context_window=True,
    function_calling_llm=llm_fallback
)

# Optional: Create a coordinator agent for complex queries
coordinator = Agent(
    role="Financial Analysis Coordinator",
    goal="""Coordinate the financial analysis workflow for optimal results.
    Document path: {file_path}
    User query: {query}
    
    Responsibilities:
    - Determine which specialized agents need to be involved
    - Break down complex queries into manageable components
    - Ensure all aspects of user query are addressed
    - Synthesize findings from different agents
    - Maintain consistency across the analysis
    """,
    verbose=True,
    backstory=(
        "You're a senior partner at a top consulting firm who specializes in financial analysis project management. "
        "You excel at coordinating complex analytical projects and ensuring all aspects are covered thoroughly. "
        "You know exactly when to bring in specialists and how to synthesize their findings into cohesive insights. "
        "Your strength lies in seeing the big picture while ensuring no critical details are missed."
    ),
    tools=[FinancialDocumentTool.read_data_tool],
    llm=llm,
    allow_delegation=True,
    max_iter=10,
    max_rpm=15,
    memory=True,
    cache=True
)

# Agent configuration summary
AGENT_CONFIGS = {
    "verifier": {
        "agent": verifier,
        "priority": 1,
        "required": True,
        "description": "Document verification - always runs first"
    },
    "financial_analyst": {
        "agent": financial_analyst,
        "priority": 2,
        "required": True,
        "description": "Core financial analysis"
    },
    "investment_advisor": {
        "agent": investment_advisor,
        "priority": 3,
        "required": False,
        "description": "Investment recommendations - runs if query relates to investing"
    },
    "risk_assessor": {
        "agent": risk_assessor,
        "priority": 3,
        "required": False,
        "description": "Risk assessment - runs for comprehensive analysis"
    },
    "coordinator": {
        "agent": coordinator,
        "priority": 0,
        "required": False,
        "description": "Optional coordinator for complex queries"
    }
}

def get_agents_for_query(query: str, detailed: bool = True) -> list:
    """Dynamically select agents based on query and detail level"""
    agents = [verifier, financial_analyst]  # Always include these
    
    if detailed:
        # For detailed analysis, include all agents
        agents.extend([investment_advisor, risk_assessor])
        
        # Check if query needs coordinator (complex queries)
        complex_keywords = ['compare', 'versus', 'vs', 'verses', 'overview', 'comprehensive']
        if any(keyword in query.lower() for keyword in complex_keywords):
            agents.insert(0, coordinator)  # Add coordinator at beginning
    
    return agents

# Export all agents and helper functions
__all__ = [
    'financial_analyst',
    'verifier', 
    'investment_advisor',
    'risk_assessor',
    'coordinator',
    'AGENT_CONFIGS',
    'get_agents_for_query'
]