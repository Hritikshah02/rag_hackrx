import json
import re
from typing import List, Dict, Any, Optional

import google.generativeai as genai

from config import Config

class LLMReasoner:
    """Handles LLM-based reasoning and response generation using Gemini"""
    
    def __init__(self):
        self.config = Config()
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Gemini LLM"""
        if not self.config.GOOGLE_API_KEY:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.config.LLM_MODEL)
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]], 
                         request_id: str = None, question_index: int = None) -> Dict[str, Any]:
        """Generate structured response from search results with comprehensive logging"""
        # Import here to avoid circular imports
        from query_logger import rag_logger
        
        # Prepare context from search results
        context = self._prepare_context(search_results)
        
        # Create prompt
        prompt = self._create_reasoning_prompt(query, context)
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                )
            )
            
            raw_response = response.text
            
            # Parse and structure the response
            parsing_details = {"parsing_success": False, "parsing_method": "unknown"}
            try:
                structured_response = self._parse_llm_response(raw_response, search_results)
                parsing_details["parsing_success"] = True
                parsing_details["parsing_method"] = "json_parse"
            except Exception as parse_error:
                structured_response = self._create_error_response(f"Parsing failed: {str(parse_error)}", raw_response)
                parsing_details["parsing_error"] = str(parse_error)
                parsing_details["parsing_method"] = "error_fallback"
            
            # Log LLM interaction if request_id provided
            if request_id and question_index:
                rag_logger.log_llm_interaction(
                    request_id, question_index, query, prompt, raw_response,
                    structured_response, parsing_details
                )
            
            return structured_response
            
        except Exception as e:
            error_response = self._create_error_response(f"Error generating response: {str(e)}")
            
            # Log error if request_id provided
            if request_id:
                rag_logger.log_error(request_id, "llm_generation", str(e), {
                    "question": query,
                    "question_index": question_index
                })
            
            print(f"❌ Error generating response: {str(e)}")
            return error_response
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_part = f"""
DOCUMENT CHUNK {i}:
Source: {result.get('source', 'Unknown')}
Relevance Score: {result.get('score', 0):.3f}
Content: {result.get('content', '')}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_reasoning_prompt(self, query: str, context: str) -> str:
        """Create reasoning prompt for the LLM - OPTIMIZED FOR CONCISE ANSWERS"""
        prompt = f"""
You are an expert insurance policy analyst. Provide CONCISE, DIRECT answers only.

USER QUERY: {query}

RELEVANT DOCUMENT CHUNKS:
{context}

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
1. Answer ONLY what is asked - no extra information
2. Keep justification under 50 words maximum
3. Use bullet points for multiple conditions
4. Start with YES/NO for coverage questions
5. Give exact numbers/periods only

JSON RESPONSE FORMAT:
{{
"decision": "APPROVED|REJECTED|PARTIAL|REQUIRES_REVIEW|INSUFFICIENT_INFO|FACTUAL_ANSWER",
"amount": "Exact amount/limit or N/A",
"confidence": "0-100",
"justification": "MAXIMUM 50 WORDS. Direct answer only. No explanations.",
"referenced_clauses": [
    {{
        "clause_number": "Section/Article number",
        "source": "Document name",
        "content": "Key sentence only - max 30 words",
        "relevance_explanation": "Max 15 words"
    }}
],
"key_factors": ["Max 3 items, 10 words each"],
"additional_requirements": ["Max 2 items, 15 words each"]
}}

CONCISE ANSWER EXAMPLES:
- Grace period: "30 days from due date"
- Coverage: "Yes, covered after 2-year waiting period"
- Waiting period: "24 months for pre-existing conditions"
- Amount: "₹5 lakh maximum per policy year"

STRICT RULES:
- NO lengthy explanations
- NO background information
- NO policy context unless directly asked
- ONLY answer the specific question
- Use numbers, dates, amounts from documents exactly

Respond with ONLY the JSON - no other text.
"""
        return prompt
    
    def _parse_llm_response(self, llm_response: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            print(f" LLM Response length: {len(llm_response)} chars")
            print(f" LLM Response preview: {llm_response[:200]}...")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f" Extracted JSON: {json_str[:150]}...")
                try:
                    parsed_response = json.loads(json_str)
                    print(f" JSON parsing successful")
                except json.JSONDecodeError as je:
                    print(f" JSON decode error: {je}")
                    print(f" Attempting fallback parsing...")
                    parsed_response = self._fallback_parse(llm_response)
            else:
                print(f" No JSON found in response, using fallback parsing")
                # Fallback parsing if no JSON found
                parsed_response = self._fallback_parse(llm_response)
            
            # Validate and enhance response
            validated_response = self._validate_response(parsed_response, search_results)
            print(f" Response validation complete")
            
            return validated_response
            
        except json.JSONDecodeError as je:
            print(f" JSON decode error: {je}")
            return self._create_error_response(f"Failed to parse LLM response as JSON: {je}", llm_response)
        except Exception as e:
            print(f" General parsing error: {e}")
            return self._create_error_response(f"Error parsing response: {str(e)}", llm_response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        print(f"🔧 Using fallback parsing for response: {response[:100]}...")
        
        # Extract key information using regex patterns
        decision_match = re.search(r'(?:decision|Decision)[":\s]*(["\']?)([^"\'\
,}]+)\1', response, re.IGNORECASE)
        amount_match = re.search(r'(?:amount|Amount)[":\s]*(["\']?)([^"\'\
,}]+)\1', response, re.IGNORECASE)
        confidence_match = re.search(r'(?:confidence|Confidence)[":\s]*(["\']?)(\d+)\1', response, re.IGNORECASE)
        
        # Try to extract justification from common patterns
        justification = "Response could not be fully parsed. Please review manually."
        justification_patterns = [
            r'(?:justification|Justification)[":\s]*(["\']?)([^"\'\
,}]{10,100})\1',
            r'(?:answer|Answer)[":\s]*(["\']?)([^"\'\
,}]{10,100})\1',
            r'(?:explanation|Explanation)[":\s]*(["\']?)([^"\'\
,}]{10,100})\1'
        ]
        
        for pattern in justification_patterns:
            just_match = re.search(pattern, response, re.IGNORECASE)
            if just_match:
                justification = just_match.group(2)
                break
        
        # Look for yes/no answers in the text
        decision = "REQUIRES_REVIEW"
        if decision_match:
            decision = decision_match.group(2)
        elif re.search(r'\b(yes|covered|approved)\b', response, re.IGNORECASE):
            decision = "APPROVED"
        elif re.search(r'\b(no|not covered|rejected|excluded)\b', response, re.IGNORECASE):
            decision = "REJECTED"
        
        result = {
            "decision": decision,
            "amount": amount_match.group(2) if amount_match else "N/A",
            "confidence": int(confidence_match.group(2)) if confidence_match else 50,
            "justification": justification,
            "referenced_clauses": [],
            "key_factors": [],
            "additional_requirements": []
        }
        
        print(f"🔧 Fallback parsing result: {result['decision']} with confidence {result['confidence']}")
        return result
    
    def _validate_response(self, response: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and enhance the parsed response"""
        # Ensure required fields exist with proper type conversion
        confidence_raw = response.get("confidence", 50)
        try:
            # Handle both string and numeric confidence values
            if isinstance(confidence_raw, str):
                # Extract numeric value from string (e.g., "85%" -> 85)
                confidence_match = re.search(r'(\d+)', confidence_raw)
                confidence_val = int(confidence_match.group(1)) if confidence_match else 50
            else:
                confidence_val = int(confidence_raw)
            confidence_val = min(100, max(0, confidence_val))
        except (ValueError, TypeError):
            confidence_val = 50
        
        validated = {
            "decision": response.get("decision", "REQUIRES_REVIEW"),
            "amount": response.get("amount", "N/A"),
            "confidence": confidence_val,
            "justification": response.get("justification", "No justification provided"),
            "referenced_clauses": response.get("referenced_clauses", []),
            "key_factors": response.get("key_factors", []),
            "additional_requirements": response.get("additional_requirements", [])
        }
        
        # Enhance referenced clauses with search result information
        enhanced_clauses = []
        for clause in validated["referenced_clauses"]:
            if isinstance(clause, dict):
                enhanced_clauses.append(clause)
            else:
                # Convert string clauses to dict format
                enhanced_clauses.append({
                    "clause_number": "N/A",
                    "source": "Unknown",
                    "content": str(clause),
                    "relevance_explanation": "Referenced in analysis"
                })
        
        # Add search results as potential references if no clauses were extracted
        if not enhanced_clauses and search_results:
            for i, result in enumerate(search_results[:3], 1):
                enhanced_clauses.append({
                    "clause_number": f"Chunk {i}",
                    "source": result.get("source", "Unknown"),
                    "content": result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),
                    "relevance_explanation": f"High relevance score: {result.get('score', 0):.3f}",
                    "relevance_score": result.get('score', 0)
                })
        
        validated["referenced_clauses"] = enhanced_clauses
        
        return validated
    
    def _create_error_response(self, error_message: str, raw_response: str = "") -> Dict[str, Any]:
        """Create error response structure"""
        return {
            "decision": "ERROR",
            "amount": "N/A",
            "confidence": 0,
            "justification": error_message,
            "referenced_clauses": [],
            "key_factors": ["Error in processing"],
            "additional_requirements": [],
            "raw_response": raw_response
        }
    
    def extract_key_entities(self, query: str) -> Dict[str, Any]:
        """Extract key entities from user query for better processing"""
        entities = {
            "age": None,
            "gender": None,
            "procedure": None,
            "location": None,
            "policy_duration": None,
            "amount": None
        }
        
        # Age extraction
        age_match = re.search(r'(\d+)[-\s]*(?:year|yr|y)[-\s]*old|(\d+)M|(\d+)F', query, re.IGNORECASE)
        if age_match:
            entities["age"] = age_match.group(1) or age_match.group(2) or age_match.group(3)
        
        # Gender extraction
        if re.search(r'\bmale\b|M\b', query, re.IGNORECASE):
            entities["gender"] = "male"
        elif re.search(r'\bfemale\b|F\b', query, re.IGNORECASE):
            entities["gender"] = "female"
        
        # Common procedures
        procedures = ["surgery", "operation", "procedure", "treatment", "therapy", "consultation"]
        for proc in procedures:
            if proc.lower() in query.lower():
                entities["procedure"] = proc
                break
        
        # Policy duration
        duration_match = re.search(r'(\d+)[-\s]*(?:month|year|day)[-\s]*(?:old\s+)?policy', query, re.IGNORECASE)
        if duration_match:
            entities["policy_duration"] = duration_match.group(0)
        
        return entities
