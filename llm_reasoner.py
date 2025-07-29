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
        """Create reasoning prompt for the LLM"""
        prompt = f"""
You are an expert insurance policy analyst. Your task is to provide precise, direct answers to user queries based on policy documents.

USER QUERY: {query}

RELEVANT DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
1. Read the user query carefully and identify what specific information is being requested.
2. Search through the document chunks for the exact information that answers the query.
3. Provide a direct, concise answer based on the policy text.
4. If the query is about claims/coverage decisions, make a clear decision. If it's a factual question about policy terms, provide the factual answer.

Provide a structured response in the following JSON format:

{{
    "decision": "APPROVED" | "REJECTED" | "PARTIAL" | "REQUIRES_REVIEW" | "INSUFFICIENT_INFO" | "FACTUAL_ANSWER",
    "amount": "Specific amount/percentage/limit if applicable, or 'N/A'",
    "confidence": "Confidence score from 0-100",
    "justification": "Direct, precise answer to the question. Be concise and factual. Quote specific policy terms, periods, percentages, or conditions exactly as stated in the documents.",
    "referenced_clauses": [
        {{
            "clause_number": "Clause identifier if available",
            "source": "Source document name",
            "content": "Exact relevant clause text from the policy",
            "relevance_explanation": "Why this clause directly answers the query"
        }}
    ],
    "key_factors": [
        "Key policy terms, conditions, or requirements that directly relate to the answer"
    ],
    "additional_requirements": [
        "Any conditions, exceptions, or additional details mentioned in the policy"
    ]
}}

IMPORTANT GUIDELINES:
- Answer directly and precisely - avoid lengthy explanations unless necessary
- Extract exact numbers, periods, percentages, and conditions from the policy text
- Use "FACTUAL_ANSWER" as decision type for informational queries (not claims decisions)
- Quote policy language exactly when providing definitions or specific terms
- If asking about waiting periods, coverage limits, or conditions - provide the exact timeframes and amounts
- For yes/no questions, start with "Yes" or "No" followed by the specific conditions
- Be concise but complete - include all relevant conditions and exceptions
- If information is not found in the documents, state "INSUFFICIENT_INFO"

EXAMPLE RESPONSE STYLES:
- For "What is the grace period?": "A grace period of thirty days is provided for premium payment after the due date..."
- For "Does policy cover X?": "Yes, the policy covers X under the following conditions: [specific conditions]"
- For "What is the waiting period?": "The waiting period is [specific time] for [specific condition/treatment]"

Respond with ONLY the JSON structure, no additional text.
"""
        return prompt
    
    def _parse_llm_response(self, llm_response: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_response = json.loads(json_str)
            else:
                # Fallback parsing
                parsed_response = self._fallback_parse(llm_response)
            
            # Validate and enhance response
            validated_response = self._validate_response(parsed_response, search_results)
            
            return validated_response
            
        except json.JSONDecodeError:
            return self._create_error_response("Failed to parse LLM response as JSON", llm_response)
        except Exception as e:
            return self._create_error_response(f"Error parsing response: {str(e)}", llm_response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        # Extract key information using regex patterns
        decision_match = re.search(r'(?:decision|Decision)[":\s]*(["\']?)([^"\'\\n,}]+)\1', response, re.IGNORECASE)
        amount_match = re.search(r'(?:amount|Amount)[":\s]*(["\']?)([^"\'\\n,}]+)\1', response, re.IGNORECASE)
        confidence_match = re.search(r'(?:confidence|Confidence)[":\s]*(["\']?)(\d+)\1', response, re.IGNORECASE)
        
        return {
            "decision": decision_match.group(2) if decision_match else "REQUIRES_REVIEW",
            "amount": amount_match.group(2) if amount_match else "N/A",
            "confidence": int(confidence_match.group(2)) if confidence_match else 50,
            "justification": "Response could not be fully parsed. Please review manually.",
            "referenced_clauses": [],
            "key_factors": [],
            "additional_requirements": []
        }
    
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
