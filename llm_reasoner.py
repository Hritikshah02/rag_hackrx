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
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate structured response based on query and search results
        
        Args:
            query: User query
            search_results: Relevant document chunks from semantic search
            
        Returns:
            Structured response with decision, amount, justification, and referenced clauses
        """
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
            
            # Parse and structure the response
            structured_response = self._parse_llm_response(response.text, search_results)
            
            return structured_response
            
        except Exception as e:
            return {
                "decision": "Error",
                "amount": "N/A",
                "justification": f"Error generating response: {str(e)}",
                "confidence": 0,
                "referenced_clauses": [],
                "raw_response": ""
            }
    
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
You are an expert AI assistant specializing in document analysis and decision-making for insurance, legal, and policy documents. Your task is to analyze the provided query against the relevant document chunks and provide a structured, accurate response.

USER QUERY: {query}

RELEVANT DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
1. Carefully analyze the user query to identify key information such as:
   - Age, gender, medical conditions
   - Procedures, treatments, or services
   - Location, timing, policy details
   - Any other relevant criteria

2. Review the provided document chunks to find relevant clauses, rules, or policies that apply to the query.

3. Make a clear decision based on the available information.

4. Provide a structured response in the following JSON format:

{{
    "decision": "APPROVED" | "REJECTED" | "PARTIAL" | "REQUIRES_REVIEW" | "INSUFFICIENT_INFO",
    "amount": "Specific amount if applicable, or 'N/A'",
    "confidence": "Confidence score from 0-100",
    "justification": "Clear explanation of the decision with reasoning",
    "referenced_clauses": [
        {{
            "clause_number": "Clause identifier if available",
            "source": "Source document name",
            "content": "Relevant clause text",
            "relevance_explanation": "Why this clause is relevant"
        }}
    ],
    "key_factors": [
        "List of key factors that influenced the decision"
    ],
    "additional_requirements": [
        "Any additional requirements or conditions if applicable"
    ]
}}

IMPORTANT GUIDELINES:
- Base your decision ONLY on the information provided in the document chunks
- If information is insufficient, state "INSUFFICIENT_INFO" as the decision
- Be specific about which clauses support your decision
- Provide clear, professional justification
- If amounts are mentioned in the documents, extract them accurately
- Consider edge cases and exceptions mentioned in the documents
- Maintain consistency with the policy language and terminology

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
