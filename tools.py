## Importing libraries and files
import os
import re
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

from crewai_tools import SerperDevTool, BaseTool
from pypdf import PdfReader
import pandas as pd
import numpy as np

## Configuration
MAX_FILE_SIZE_MB = 10
SUPPORTED_LANGUAGES = ['en']  # Add more if needed

## Creating search tool with configuration
search_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
    n_results=5
)

## Utility functions for text processing
class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # Fix hyphenated line breaks
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text.strip()
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract all numbers from text"""
        # Find numbers that could be financial figures (with commas, decimals, $, etc.)
        pattern = r'\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|M|B)?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        numbers = []
        for match in matches:
            # Clean and convert to float
            clean = match.replace('$', '').replace(',', '').strip()
            if 'million' in clean.lower() or 'M' in clean:
                clean = re.sub(r'[a-zA-Z]', '', clean)
                try:
                    numbers.append(float(clean) * 1_000_000)
                except:
                    pass
            elif 'billion' in clean.lower() or 'B' in clean:
                clean = re.sub(r'[a-zA-Z]', '', clean)
                try:
                    numbers.append(float(clean) * 1_000_000_000)
                except:
                    pass
            else:
                try:
                    numbers.append(float(clean))
                except:
                    pass
        
        return numbers
    
    @staticmethod
    def extract_financial_terms(text: str) -> Dict[str, int]:
        """Extract and count financial terminology"""
        financial_terms = {
            'revenue': 0, 'profit': 0, 'loss': 0, 'asset': 0, 'liability': 0,
            'equity': 0, 'cash': 0, 'debt': 0, 'margin': 0, 'eps': 0,
            'dividend': 0, 'share': 0, 'stock': 0, 'market': 0, 'risk': 0,
            'income': 0, 'balance': 0, 'statement': 0, 'financial': 0,
            'quarter': 0, 'annual': 0, 'fiscal': 0, 'growth': 0
        }
        
        text_lower = text.lower()
        for term in financial_terms:
            financial_terms[term] = text_lower.count(term)
        
        return financial_terms

## Enhanced PDF Reader Tool with CrewAI BaseTool integration
class FinancialDocumentTool(BaseTool):
    """Enhanced tool for reading and analyzing financial PDF documents"""
    
    name: str = "Financial Document Reader"
    description: str = """Reads and extracts text content from financial PDF documents.
    Provides cleaned, normalized text suitable for financial analysis.
    Also returns metadata about the document including page count and basic statistics."""
    
    def _run(self, path: str = 'data/sample.pdf') -> str:
        """Tool to read data from a pdf file from a path
        
        Args:
            path (str): Path of the pdf file. Defaults to 'data/sample.pdf'.
            
        Returns:
            str: Full Financial Document file content with metadata
        """
        try:
            # Validate input
            if not path or not isinstance(path, str):
                return "Error: Invalid file path provided"
            
            # Check if file exists
            if not os.path.exists(path):
                return f"Error: File not found at path: {path}"
            
            # Check file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return f"Error: File size ({file_size_mb:.1f}MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB"
            
            # Create PDF reader object
            reader = PdfReader(path)
            
            # Document metadata
            metadata = {
                "pages": len(reader.pages),
                "file_size_mb": round(file_size_mb, 2),
                "encrypted": reader.is_encrypted
            }
            
            # Handle encrypted PDFs
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except:
                    return "Error: PDF is encrypted and cannot be read"
            
            # Extract text from all pages
            full_report = []
            page_texts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    content = page.extract_text()
                    if content:
                        # Clean and format the content
                        cleaned_content = TextProcessor.clean_text(content)
                        page_texts.append({
                            "page": page_num,
                            "content": cleaned_content,
                            "length": len(cleaned_content)
                        })
                        full_report.append(f"--- Page {page_num} ---\n{cleaned_content}")
                except Exception as e:
                    full_report.append(f"--- Page {page_num} ---\n[Error extracting content: {str(e)}]")
            
            # Check if we got any content
            if not full_report:
                return "Warning: No text could be extracted from the PDF. The file might be scanned or image-based."
            
            # Add document summary
            summary = self._generate_document_summary(page_texts, metadata)
            
            # Combine everything
            result = f"""DOCUMENT ANALYSIS REPORT
========================
File: {os.path.basename(path)}
Pages: {metadata['pages']}
Size: {metadata['file_size_mb']}MB

{document_summary}

CONTENT
=======
{chr(10).join(full_report)}
"""
            
            return result
            
        except Exception as e:
            return f"Error reading PDF file: {str(e)}"
    
    def _generate_document_summary(self, page_texts: List[Dict], metadata: Dict) -> str:
        """Generate a summary of the document"""
        total_chars = sum(p['length'] for p in page_texts)
        
        # Combine all text for analysis
        all_text = ' '.join(p['content'] for p in page_texts)
        
        # Extract financial terms
        financial_terms = TextProcessor.extract_financial_terms(all_text)
        
        # Extract numbers
        numbers = TextProcessor.extract_numbers(all_text)
        
        summary = f"""
DOCUMENT SUMMARY
----------------
Total Content Length: {total_chars} characters
Average Page Length: {total_chars // len(page_texts)} characters

Financial Terminology Detected:
{', '.join([f'{term}: {count}' for term, count in financial_terms.items() if count > 0])}

Numbers Found: {len(numbers)} financial figures detected
"""
        return summary
    
    async def _arun(self, path: str = 'data/sample.pdf') -> str:
        """Async version of the tool"""
        return self._run(path)

## Enhanced Investment Analysis Tool
class InvestmentTool(BaseTool):
    """Advanced tool for analyzing investment opportunities from financial data"""
    
    name: str = "Investment Analysis Tool"
    description: str = """Analyzes financial document data to identify investment opportunities,
    calculate key investment metrics, and provide preliminary investment insights."""
    
    def _run(self, financial_document_data: str) -> str:
        """Analyze financial document for investment opportunities
        
        Args:
            financial_document_data (str): Extracted text from financial document
            
        Returns:
            str: Comprehensive investment analysis
        """
        try:
            # Validate input
            if not financial_document_data:
                return "Error: No financial data provided for analysis"
            
            # Process and analyze the financial document data
            analysis = []
            analysis.append("=" * 60)
            analysis.append("INVESTMENT ANALYSIS REPORT")
            analysis.append("=" * 60)
            
            # Extract key metrics
            metrics = self._extract_investment_metrics(financial_document_data)
            
            # Calculate ratios
            ratios = self._calculate_ratios(metrics)
            
            # Identify opportunities
            opportunities = self._identify_opportunities(metrics, ratios, financial_document_data)
            
            # Identify risks
            risks = self._identify_investment_risks(financial_document_data)
            
            # Format the analysis
            analysis.append("\n📊 KEY METRICS")
            analysis.append("-" * 40)
            for key, value in metrics.items():
                if value:
                    analysis.append(f"{key}: {value}")
            
            analysis.append("\n📈 FINANCIAL RATIOS")
            analysis.append("-" * 40)
            for key, value in ratios.items():
                if value:
                    analysis.append(f"{key}: {value}")
            
            analysis.append("\n💡 INVESTMENT OPPORTUNITIES")
            analysis.append("-" * 40)
            for opp in opportunities:
                analysis.append(f"• {opp}")
            
            analysis.append("\n⚠️ INVESTMENT RISKS")
            analysis.append("-" * 40)
            for risk in risks:
                analysis.append(f"• {risk}")
            
            analysis.append("\n📋 PRELIMINARY ASSESSMENT")
            analysis.append("-" * 40)
            analysis.append(self._generate_investment_assessment(metrics, ratios, opportunities, risks))
            
            analysis.append("\n" + "=" * 60)
            analysis.append("Note: This is an automated preliminary analysis.")
            analysis.append("Consult with a qualified financial advisor before making investment decisions.")
            analysis.append("=" * 60)
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error in investment analysis: {str(e)}"
    
    def _extract_investment_metrics(self, text: str) -> Dict[str, str]:
        """Extract key investment metrics from text"""
        metrics = {
            "Revenue": None,
            "Net Income": None,
            "EPS": None,
            "Total Assets": None,
            "Total Liabilities": None,
            "Shareholders Equity": None,
            "Operating Cash Flow": None,
            "Dividends per Share": None,
            "Shares Outstanding": None,
            "Stock Price": None
        }
        
        text_lower = text.lower()
        
        # Extract revenue
        revenue_patterns = [
            r'revenue[:\s]*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|b)?',
            r'sales[:\s]*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|b)?'
        ]
        for pattern in revenue_patterns:
            match = re.search(pattern, text_lower)
            if match:
                metrics["Revenue"] = self._format_financial_number(match)
                break
        
        # Extract net income
        income_patterns = [
            r'net income[:\s]*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|b)?',
            r'net profit[:\s]*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|b)?',
            r'earnings[:\s]*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|b)?'
        ]
        for pattern in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                metrics["Net Income"] = self._format_financial_number(match)
                break
        
        # Extract EPS
        eps_patterns = [
            r'eps[:\s]*\$?\s*(\d+(?:\.\d+)?)',
            r'earnings per share[:\s]*\$?\s*(\d+(?:\.\d+)?)'
        ]
        for pattern in eps_patterns:
            match = re.search(pattern, text_lower)
            if match:
                metrics["EPS"] = f"${match.group(1)}"
                break
        
        return metrics
    
    def _calculate_ratios(self, metrics: Dict) -> Dict[str, str]:
        """Calculate financial ratios from metrics"""
        ratios = {}
        
        # P/E Ratio (would need stock price)
        if metrics.get("EPS") and metrics.get("Stock Price"):
            try:
                eps = float(metrics["EPS"].replace('$', ''))
                price = float(metrics["Stock Price"].replace('$', ''))
                ratios["P/E Ratio"] = f"{price/eps:.2f}"
            except:
                pass
        
        # Debt-to-Equity
        if metrics.get("Total Liabilities") and metrics.get("Shareholders Equity"):
            try:
                debt = float(metrics["Total Liabilities"].replace('$', '').replace(',', ''))
                equity = float(metrics["Shareholders Equity"].replace('$', '').replace(',', ''))
                ratios["Debt-to-Equity"] = f"{debt/equity:.2f}"
            except:
                pass
        
        return ratios
    
    def _identify_opportunities(self, metrics: Dict, ratios: Dict, text: str) -> List[str]:
        """Identify potential investment opportunities"""
        opportunities = []
        text_lower = text.lower()
        
        # Check for growth indicators
        growth_keywords = ['grew', 'growth', 'increase', 'expansion', 'record']
        if any(keyword in text_lower for keyword in growth_keywords):
            opportunities.append("Company shows signs of growth/expansion")
        
        # Check for profitability
        if metrics.get("Net Income"):
            opportunities.append("Company is profitable with positive net income")
        
        # Check for new products/markets
        if 'new product' in text_lower or 'launch' in text_lower or 'expansion' in text_lower:
            opportunities.append("New products or market expansion mentioned")
        
        # Check for positive guidance
        if 'guidance' in text_lower and ('raise' in text_lower or 'increase' in text_lower):
            opportunities.append("Positive forward guidance provided")
        
        return opportunities if opportunities else ["No clear opportunities identified in automated analysis"]
    
    def _identify_investment_risks(self, text: str) -> List[str]:
        """Identify potential investment risks"""
        risks = []
        text_lower = text.lower()
        
        risk_indicators = {
            'debt': 'High debt levels mentioned',
            'lawsuit': 'Legal/regulatory risks',
            'competition': 'Intense competition noted',
            'regulation': 'Regulatory risks identified',
            'uncertainty': 'Market/economic uncertainty',
            'decline': 'Declining performance indicators'
        }
        
        for keyword, risk_desc in risk_indicators.items():
            if keyword in text_lower:
                risks.append(risk_desc)
        
        return risks if risks else ["No significant risks identified in automated analysis"]
    
    def _format_financial_number(self, match) -> str:
        """Format extracted financial numbers"""
        number = match.group(1).replace(',', '')
        if len(match.groups()) > 1 and match.group(2):
            multiplier = match.group(2).lower()
            if multiplier in ['million', 'm']:
                number = f"${float(number) * 1_000_000:,.0f}"
            elif multiplier in ['billion', 'b']:
                number = f"${float(number) * 1_000_000_000:,.0f}"
            else:
                number = f"${float(number):,.0f}"
        else:
            number = f"${float(number):,.0f}"
        return number
    
    def _generate_investment_assessment(self, metrics: Dict, ratios: Dict, opportunities: List, risks: List) -> str:
        """Generate overall investment assessment"""
        
        score = 0
        max_score = 10
        
        # Positive indicators
        if metrics.get("Revenue"):
            score += 2
        if metrics.get("Net Income"):
            score += 2
        if len(opportunities) > 1 and opportunities[0] != "No clear opportunities identified in automated analysis":
            score += 2
        if len(risks) == 1 and risks[0] == "No significant risks identified in automated analysis":
            score += 2
        
        # Determine rating
        if score >= 6:
            rating = "POSITIVE"
        elif score >= 4:
            rating = "NEUTRAL"
        else:
            rating = "CAUTIOUS"
        
        return f"""
Based on automated analysis:
- Investment Rating: {rating} (Score: {score}/{max_score})
- Key Strengths: {len([o for o in opportunities if 'No clear opportunities' not in o])} opportunities identified
- Key Risks: {len([r for r in risks if 'No significant risks' not in r])} risk factors identified

Recommendation: {'Consider for further research' if rating == 'POSITIVE' else 'Monitor before investing' if rating == 'NEUTRAL' else 'Exercise caution'}
"""
    
    async def _arun(self, financial_document_data: str) -> str:
        """Async version of the tool"""
        return self._run(financial_document_data)

## Enhanced Risk Assessment Tool
class RiskTool(BaseTool):
    """Advanced tool for comprehensive risk assessment"""
    
    name: str = "Risk Assessment Tool"
    description: str = """Performs comprehensive risk analysis on financial documents,
    identifying and categorizing various types of business and investment risks."""
    
    def _run(self, financial_document_data: str) -> str:
        """Create comprehensive risk assessment based on financial document
        
        Args:
            financial_document_data (str): Extracted text from financial document
            
        Returns:
            str: Detailed risk assessment report
        """
        try:
            # Validate input
            if not financial_document_data:
                return "Error: No financial data provided for risk assessment"
            
            # Initialize assessment
            assessment = []
            assessment.append("=" * 60)
            assessment.append("COMPREHENSIVE RISK ASSESSMENT REPORT")
            assessment.append("=" * 60)
            
            # Analyze different risk categories
            risk_categories = {
                "FINANCIAL RISKS": self._assess_financial_risks(financial_document_data),
                "OPERATIONAL RISKS": self._assess_operational_risks(financial_document_data),
                "STRATEGIC RISKS": self._assess_strategic_risks(financial_document_data),
                "COMPLIANCE RISKS": self._assess_compliance_risks(financial_document_data),
                "MARKET RISKS": self._assess_market_risks(financial_document_data)
            }
            
            # Calculate overall risk score
            overall_score, risk_level = self._calculate_risk_score(risk_categories)
            
            # Format the assessment
            assessment.append(f"\n📊 OVERALL RISK RATING: {risk_level} (Score: {overall_score}/100)")
            assessment.append("-" * 60)
            
            for category, risks in risk_categories.items():
                assessment.append(f"\n{category}")
                assessment.append("-" * 40)
                
                if risks:
                    for risk in risks:
                        assessment.append(f"• {risk}")
                else:
                    assessment.append("• No significant risks identified in this category")
            
            # Add risk heat map
            assessment.append("\n🔥 RISK HEAT MAP")
            assessment.append("-" * 40)
            assessment.append(self._generate_risk_heatmap(risk_categories))
            
            # Add mitigation recommendations
            assessment.append("\n🛡️ RECOMMENDED MITIGATIONS")
            assessment.append("-" * 40)
            assessment.extend(self._generate_mitigation_recommendations(risk_categories))
            
            # Add disclaimer
            assessment.append("\n" + "=" * 60)
            assessment.append("DISCLAIMER: This is an automated risk assessment based on")
            assessment.append("document analysis. Professional risk management advice")
            assessment.append("should be sought for critical investment decisions.")
            assessment.append("=" * 60)
            
            return "\n".join(assessment)
            
        except Exception as e:
            return f"Error in risk assessment: {str(e)}"
    
    def _assess_financial_risks(self, text: str) -> List[str]:
        """Assess financial risks from the document"""
        risks = []
        text_lower = text.lower()
        
        # Debt indicators
        debt_indicators = [
            ('debt', 'High debt levels present'),
            ('leverage', 'Significant leverage indicated'),
            ('interest', 'Interest rate exposure'),
            ('covenant', 'Debt covenants mentioned'),
            ('refinance', 'Refinancing risk identified')
        ]
        
        for keyword, risk in debt_indicators:
            if keyword in text_lower:
                risks.append(risk)
        
        # Liquidity indicators
        if 'liquidity' in text_lower or 'cash flow' in text_lower:
            if any(term in text_lower for term in ['declining', 'decrease', 'negative']):
                risks.append("Potential liquidity concerns")
        
        # Profitability indicators
        if 'loss' in text_lower and ('net loss' in text_lower or 'operating loss' in text_lower):
            risks.append("Net losses reported - profitability risk")
        
        return risks
    
    def _assess_operational_risks(self, text: str) -> List[str]:
        """Assess operational risks from the document"""
        risks = []
        text_lower = text.lower()
        
        operational_indicators = [
            ('supply chain', 'Supply chain vulnerabilities'),
            ('manufacturing', 'Manufacturing/operational risks'),
            ('labor', 'Labor/workforce risks'),
            ('employee', 'Workforce-related risks'),
            ('facility', 'Facility/asset risks'),
            ('technology', 'Technology/system risks'),
            ('cyber', 'Cybersecurity risks'),
            ('data', 'Data security concerns'),
            ('business interruption', 'Business interruption risk')
        ]
        
        for keyword, risk in operational_indicators:
            if keyword in text_lower:
                risks.append(risk)
        
        return risks
    
    def _assess_strategic_risks(self, text: str) -> List[str]:
        """Assess strategic risks from the document"""
        risks = []
        text_lower = text.lower()
        
        strategic_indicators = [
            ('competition', 'Intense competition risk'),
            ('market share', 'Market share erosion risk'),
            ('new entrant', 'Threat from new market entrants'),
            ('substitute', 'Substitute products/services risk'),
            ('innovation', 'Innovation/obsolescence risk'),
            ('customer concentration', 'Customer concentration risk'),
            ('supplier concentration', 'Supplier concentration risk')
        ]
        
        for keyword, risk in strategic_indicators:
            if keyword in text_lower:
                risks.append(risk)
        
        return risks
    
    def _assess_compliance_risks(self, text: str) -> List[str]:
        """Assess compliance and regulatory risks"""
        risks = []
        text_lower = text.lower()
        
        compliance_indicators = [
            ('regulatory', 'Regulatory compliance risks'),
            ('lawsuit', 'Legal/litigation risks'),
            ('investigation', 'Under investigation'),
            ('compliance', 'Compliance program risks'),
            ('environmental', 'Environmental compliance risks'),
            ('tax', 'Tax compliance risks'),
            ('export', 'Export/import compliance risks'),
            ('sanction', 'Sanctions/embargo risks')
        ]
        
        for keyword, risk in compliance_indicators:
            if keyword in text_lower:
                risks.append(risk)
        
        return risks
    
    def _assess_market_risks(self, text: str) -> List[str]:
        """Assess market and macroeconomic risks"""
        risks = []
        text_lower = text.lower()
        
        market_indicators = [
            ('economic', 'Economic sensitivity risk'),
            ('market volatility', 'Market volatility risk'),
            ('interest rate', 'Interest rate risk'),
            ('currency', 'Currency/exchange rate risk'),
            ('inflation', 'Inflation risk'),
            ('recession', 'Recession vulnerability'),
            ('geopolitical', 'Geopolitical risk'),
            ('trade', 'Trade policy risk')
        ]
        
        for keyword, risk in market_indicators:
            if keyword in text_lower:
                risks.append(risk)
        
        return risks
    
    def _calculate_risk_score(self, risk_categories: Dict) -> tuple:
        """Calculate overall risk score and level"""
        
        total_risks = sum(len(risks) for risks in risk_categories.values())
        max_possible_risks = len(risk_categories) * 5  # Assume max 5 risks per category
        
        if max_possible_risks > 0:
            risk_percentage = min(100, (total_risks / max_possible_risks) * 100)
        else:
            risk_percentage = 0
        
        # Determine risk level
        if risk_percentage >= 60:
            level = "HIGH"
        elif risk_percentage >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return round(risk_percentage, 1), level
    
    def _generate_risk_heatmap(self, risk_categories: Dict) -> str:
        """Generate a visual risk heatmap"""
        heatmap = []
        
        for category, risks in risk_categories.items():
            risk_count = len(risks)
            if risk_count >= 4:
                level = "🔴 HIGH"
            elif risk_count >= 2:
                level = "🟡 MEDIUM"
            else:
                level = "🟢 LOW"
            
            heatmap.append(f"{category:<20} : {level} ({risk_count} risks)")
        
        return "\n".join(heatmap)
    
    def _generate_mitigation_recommendations(self, risk_categories: Dict) -> List[str]:
        """Generate mitigation recommendations based on identified risks"""
        recommendations = []
        
        for category, risks in risk_categories.items():
            if risks:
                if "FINANCIAL" in category:
                    recommendations.append("• Strengthen balance sheet and improve liquidity")
                    recommendations.append("• Consider debt refinancing options")
                elif "OPERATIONAL" in category:
                    recommendations.append("• Enhance operational resilience and diversify supply chain")
                    recommendations.append("• Invest in technology and cybersecurity")
                elif "STRATEGIC" in category:
                    recommendations.append("• Diversify customer and supplier base")
                    recommendations.append("• Increase R&D investment for innovation")
                elif "COMPLIANCE" in category:
                    recommendations.append("• Strengthen compliance monitoring programs")
                    recommendations.append("• Conduct regular legal/regulatory audits")
                elif "MARKET" in category:
                    recommendations.append("• Implement hedging strategies for market risks")
                    recommendations.append("• Diversify geographic exposure")
        
        if not recommendations:
            recommendations.append("• No significant risks identified - maintain current risk management practices")
        
        return recommendations
    
    async def _arun(self, financial_document_data: str) -> str:
        """Async version of the tool"""
        return self._run(financial_document_data)

## File validation utility
def validate_file_size(file_size: int, max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """Validate file size doesn't exceed limit"""
    return file_size <= max_size_mb * 1024 * 1024

def validate_file_extension(filename: str, allowed_extensions: List[str] = ['.pdf']) -> bool:
    """Validate file extension"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions

## Export all tools
__all__ = [
    'search_tool',
    'FinancialDocumentTool',
    'InvestmentTool', 
    'RiskTool',
    'TextProcessor',
    'validate_file_size',
    'validate_file_extension'
]