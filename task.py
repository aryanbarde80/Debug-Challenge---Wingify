
### Risk Trends Assessment
- **Risks Increasing**: [List of risks trending upward]
- **Risks Stable**: [List of stable risks]
- **Risks Decreasing**: [List of risks trending downward]

### Industry-Specific Risks
[Analysis of risks particular to this industry]

### Risk Management Assessment
- **Risk Culture**: [Assessment of risk awareness]
- **Disclosure Quality**: [How well risks are disclosed]
- **Proactive/Reactive**: [Company's approach to risk]

### Risk-Adjusted Outlook
[Overall assessment considering all risks]

### Early Warning Indicators
| Risk | Warning Sign | Threshold |
|------|--------------|-----------|
| [Risk] | [Metric to monitor] | [Level that triggers concern] |

### Recommendations for Risk Mitigation
1. [Actionable recommendation]
2. [Actionable recommendation]
3. [Actionable recommendation]

### Answer to User Query: "{query}"
[Specific risk insights related to user's question]

---
*Assessment by: Chief Risk Officer (CRO)*
*Data sourced from: {file_path}*
*Timestamp: Automatically added during execution*""",

    agent=risk_assessor,
    tools=[FinancialDocumentTool.read_data_tool, RiskTool.create_risk_assessment_tool],
    async_execution=False,
    output_file="outputs/risk_assessment.json",
    context=[analyze_financial_document]  # Depends on financial analysis
)

## Task configurations dictionary for easy management
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

def get_tasks_for_analysis(detailed: bool = True, include_investment: bool = True, include_risk: bool = True) -> list:
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