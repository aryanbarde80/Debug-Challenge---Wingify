# Importing libraries and files
from crewai import Task
from agents import financial_analyst, verifier, investment_advisor, risk_assessor
from tools import FinancialDocumentTool, InvestmentTool, RiskTool
from typing import Dict, Any, List
import json

# Helper function to create task context
def create_task_context(query: str, file_path: str) -> Dict[str, Any]:
    """Create context dictionary for tasks"""
    return {
        "query": query,
        "file_path": file_path,
        "document_type": "financial_report",
        "analysis_depth": "comprehensive"
    }

# ----------------------------------------------------------------------------
# Task 1: Document Verification
# ----------------------------------------------------------------------------
verification = Task(
    description="""Verify if the uploaded document is a valid financial document.
    
    Document Location: {file_path}
    User Query: {query}
    
    Verification Process:
    1. Check if the document exists and is accessible
    2. Scan for financial terminology and keywords
    3. Look for key financial sections:
       - Income Statement / Profit & Loss
       - Balance Sheet
       - Cash Flow Statement
       - Management Discussion & Analysis (MD&A)
       - Risk Factors
       - Notes to Financial Statements
    4. Identify document type (10-K, 10-Q, Annual Report, etc.)
    5. Assess confidence level based on findings
    
    Use the PDF reader tool to access document content at {file_path}.
    Document any issues or red flags found.
    """,

    expected_output="""Document Verification Report:
    
    Verification Status: [Valid / Invalid / Partial]
    
    Document Type: [10-K / 10-Q / Annual Report / etc.]
    Company/Ticker: [If identifiable]
    Period Covered: [Fiscal quarter/year]
    Pages: [Number of pages]
    
    Key Sections Found:
    - Section 1: [Present/Missing]
    - Section 2: [Present/Missing]
    - Section 3: [Present/Missing]
    
    Confidence Level: [High / Medium / Low]
    Confidence Factors:
    - [Factor increasing confidence]
    - [Factor decreasing confidence]
    
    Red Flags / Concerns:
    - [Any issues identified]
    
    Verification Summary:
    [Brief summary of verification findings]
    """,

    agent=verifier,
    tools=[FinancialDocumentTool.read_data_tool],
    async_execution=False,
    output_file="outputs/verification_report.json"
)

# ----------------------------------------------------------------------------
# Task 2: Financial Analysis
# ----------------------------------------------------------------------------
analyze_financial_document = Task(
    description="""Analyze the financial document thoroughly based on the user query.
    
    Document Location: {file_path}
    User Query: {query}
    
    Analysis Framework:
    1. Extract key financial metrics:
       - Revenue (total, by segment, growth rates)
       - Profitability metrics (Gross profit, Operating income, Net income)
       - Margins (Gross margin, Operating margin, Net margin)
       - Earnings per share
       - Balance sheet items
       - Cash flow metrics
       - Key financial ratios
    
    2. Performance Analysis:
       - Period-over-period comparisons
       - Trend analysis
       - Segment performance breakdown
    
    3. Financial Health Assessment:
       - Liquidity position
       - Solvency and leverage
       - Operational efficiency
       - Cash generation capability
    
    4. Address specific questions from user query: {query}
    
    Base all analysis on actual document content.
    Cite specific sections and page numbers for key findings.
    """,

    expected_output="""Financial Analysis Report:
    
    Executive Summary:
    [Overview of key findings]
    
    Key Financial Metrics:
    | Metric | Value | Previous Period | Change (%) |
    |--------|-------|-----------------|------------|
    | Revenue | $X | $Y | X% |
    | Net Income | $X | $Y | X% |
    | EPS | $X | $Y | X% |
    
    Financial Ratios:
    - Gross Margin: X%
    - Operating Margin: X%
    - Net Margin: X%
    - Current Ratio: X.X
    - Debt-to-Equity: X.X
    
    Trend Analysis:
    [Analysis of key trends]
    
    Strengths:
    - [Strength 1 with evidence]
    - [Strength 2 with evidence]
    
    Concerns:
    - [Concern 1 with evidence]
    - [Concern 2 with evidence]
    
    Answer to User Query: "{query}"
    [Direct answer based on document content]
    
    Conclusions:
    [Final assessment of financial health]
    """,

    agent=financial_analyst,
    tools=[FinancialDocumentTool.read_data_tool],
    async_execution=False,
    output_file="outputs/financial_analysis.json",
    context=[verification]
)

# ----------------------------------------------------------------------------
# Task 3: Investment Analysis
# ----------------------------------------------------------------------------
investment_analysis = Task(
    description="""Based on the financial document analysis, provide investment insights.
    
    Document Location: {file_path}
    User Query: {query}
    
    Investment Analysis Framework:
    1. Valuation Assessment:
       - P/E ratio
       - P/B ratio
       - EV/EBITDA
       - Comparison to industry averages
    
    2. Growth Analysis:
       - Revenue growth trends
       - Earnings growth trajectory
       - Future growth catalysts mentioned
    
    3. Competitive Position:
       - Market share indicators
       - Competitive advantages
       - Barriers to entry
    
    4. Investment Thesis:
       - Bull case: Reasons to invest
       - Bear case: Reasons to avoid
    
    5. Recommendations with clear rationale
    
    Include appropriate disclaimers and risk warnings.
    """,

    expected_output="""Investment Analysis Report:
    
    Executive Summary:
    [Overview of investment thesis]
    
    Valuation Analysis:
    | Metric | Value | Industry Avg | Assessment |
    |--------|-------|--------------|------------|
    | P/E Ratio | X.X | X.X | [Undervalued/Fair/Overvalued] |
    | P/B Ratio | X.X | X.X | [Undervalued/Fair/Overvalued] |
    
    Growth Assessment:
    - Historical Growth: X% CAGR
    - Growth Drivers: [Key factors]
    - Growth Risks: [Potential impediments]
    
    Investment Thesis:
    
    Bull Case:
    1. [Reason with supporting evidence]
    2. [Reason with supporting evidence]
    
    Bear Case:
    1. [Risk with supporting evidence]
    2. [Risk with supporting evidence]
    
    Recommendations:
    - Short-term (0-12 mo): [Buy/Hold/Sell]
    - Long-term (3+ yr): [Buy/Hold/Sell]
    Rationale: [Explanation]
    
    Disclaimers:
    - This analysis is based solely on the provided document
    - Past performance does not guarantee future results
    - All investments involve risk
    - Consult with a qualified financial advisor
    """,

    agent=investment_advisor,
    tools=[FinancialDocumentTool.read_data_tool, InvestmentTool.analyze_investment_tool],
    async_execution=False,
    output_file="outputs/investment_analysis.json",
    context=[analyze_financial_document]
)

# ----------------------------------------------------------------------------
# Task 4: Risk Assessment
# ----------------------------------------------------------------------------
risk_assessment = Task(
    description="""Assess risks based on the financial document analysis.
    
    Document Location: {file_path}
    User Query: {query}
    
    Risk Assessment Framework:
    1. Financial Risks:
       - Liquidity risk
       - Solvency risk
       - Credit risk
       - Market risk
    
    2. Operational Risks:
       - Supply chain vulnerabilities
       - Production dependencies
       - Technology risks
    
    3. Strategic Risks:
       - Competitive threats
       - Market position erosion
       - Regulatory risks
    
    4. Macroeconomic Risks:
       - Economic cycle sensitivity
       - Interest rate exposure
       - Currency risk
    
    Provide risk ratings with clear rationale.
    Include recommendations for risk mitigation.
    """,

    expected_output="""Risk Assessment Report:
    
    Executive Summary:
    [Overview of key risks]
    
    Risk Matrix:
    | Risk Category | Specific Risk | Severity | Probability | Mitigation |
    |---------------|---------------|----------|-------------|------------|
    | Financial | [Risk] | H/M/L | H/M/L | [Mitigation] |
    | Operational | [Risk] | H/M/L | H/M/L | [Mitigation] |
    
    Detailed Risk Analysis:
    
    Top 3 Critical Risks:
    1. [Risk Name]
       - Description: [Details]
       - Evidence: [Page reference]
       - Impact: [Potential consequences]
       - Mitigation: [What company is doing]
    
    2. [Risk Name]
       - [Similar structure]
    
    Risk Trends Assessment:
    - Risks Increasing: [List of risks]
    - Risks Stable: [List of risks]
    - Risks Decreasing: [List of risks]
    
    Industry-Specific Risks:
    [Analysis of risks particular to this industry]
    
    Risk Management Assessment:
    - Risk Culture: [Assessment of risk awareness]
    - Disclosure Quality: [How well risks are disclosed]
    - Proactive/Reactive: [Company's approach to risk]
    
    Risk-Adjusted Outlook:
    [Overall assessment considering all risks]
    
    Early Warning Indicators:
    | Risk | Warning Sign | Threshold |
    |------|--------------|-----------|
    | [Risk] | [Metric to monitor] | [Level that triggers concern] |
    
    Recommendations for Risk Mitigation:
    1. [Actionable recommendation]
    2. [Actionable recommendation]
    3. [Actionable recommendation]
    
    Answer to User Query: "{query}"
    [Specific risk insights related to user's question]
    """,

    agent=risk_assessor,
    tools=[FinancialDocumentTool.read_data_tool, RiskTool.create_risk_assessment_tool],
    async_execution=False,
    output_file="outputs/risk_assessment.json",
    context=[analyze_financial_document]
)

# ----------------------------------------------------------------------------
# Task Configuration
# ----------------------------------------------------------------------------
TASK_CONFIGS = {
    "verification": {
        "task": verification,
        "priority": 1,
        "required": True,
        "description": "Document verification - always runs first",
        "depends_on": []
    },
    "analyze_financial_document": {
        "task": analyze_financial_document,
        "priority": 2,
        "required": True,
        "description": "Core financial analysis",
        "depends_on": ["verification"]
    },
    "investment_analysis": {
        "task": investment_analysis,
        "priority": 3,
        "required": False,
        "description": "Investment recommendations",
        "depends_on": ["analyze_financial_document"]
    },
    "risk_assessment": {
        "task": risk_assessment,
        "priority": 3,
        "required": False,
        "description": "Risk assessment",
        "depends_on": ["analyze_financial_document"]
    }
}

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def get_tasks_for_analysis(detailed: bool = True, include_investment: bool = True, include_risk: bool = True) -> List[Task]:
    """Get appropriate tasks based on analysis requirements"""
    
    # Always include verification and financial analysis
    tasks = [verification, analyze_financial_document]
    
    if detailed:
        if include_investment:
            tasks.append(investment_analysis)
        if include_risk:
            tasks.append(risk_assessment)
    
    return tasks

# Export all tasks
__all__ = [
    'verification',
    'analyze_financial_document',
    'investment_analysis', 
    'risk_assessment',
    'TASK_CONFIGS',
    'get_tasks_for_analysis'
]