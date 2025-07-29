import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

class RAGLogger:
    """Comprehensive logger for RAG system to track document processing, queries, and responses"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create different log files for different aspects
        self.request_log_file = self.log_dir / "requests.jsonl"
        self.chunk_log_file = self.log_dir / "chunk_retrievals.jsonl"
        self.llm_log_file = self.log_dir / "llm_interactions.jsonl"
        self.error_log_file = self.log_dir / "errors.jsonl"
        
    def start_request(self, document_url: str, questions: List[str]) -> str:
        """Start logging a new request and return request_id"""
        request_id = str(uuid.uuid4())
        
        request_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_url": document_url,
            "questions": questions,
            "num_questions": len(questions),
            "status": "started"
        }
        
        self._write_to_log(self.request_log_file, request_entry)
        return request_id
    
    def log_document_processing(self, request_id: str, document_info: Dict[str, Any]):
        """Log document processing details"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "document_processing",
            "document_path": document_info.get("path"),
            "num_chunks": document_info.get("num_chunks"),
            "total_tokens": document_info.get("total_tokens"),
            "processing_time_seconds": document_info.get("processing_time")
        }
        
        self._write_to_log(self.request_log_file, log_entry)
    
    def log_chunk_retrieval(self, request_id: str, question_index: int, question: str, 
                           search_results: List[Dict[str, Any]], search_params: Dict[str, Any]):
        """Log chunk retrieval for each question"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_index": question_index,
            "question": question,
            "search_params": search_params,
            "num_results": len(search_results),
            "search_results": [
                {
                    "chunk_id": result.get("chunk_id"),
                    "source": result.get("source"),
                    "score": result.get("score"),
                    "content_preview": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                    "full_content": result.get("content", ""),  # Full content for analysis
                    "token_count": result.get("token_count"),
                    "keywords": result.get("keywords")
                }
                for result in search_results
            ]
        }
        
        self._write_to_log(self.chunk_log_file, log_entry)
    
    def log_llm_interaction(self, request_id: str, question_index: int, question: str,
                           prompt: str, raw_response: str, parsed_response: Dict[str, Any],
                           processing_details: Dict[str, Any]):
        """Log LLM prompt, response, and processing details"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_index": question_index,
            "question": question,
            "prompt": prompt,
            "raw_llm_response": raw_response,
            "parsed_response": parsed_response,
            "processing_details": processing_details,
            "response_length": len(raw_response),
            "parsing_success": processing_details.get("parsing_success", False),
            "confidence": parsed_response.get("confidence", 0),
            "decision": parsed_response.get("decision", "UNKNOWN")
        }
        
        self._write_to_log(self.llm_log_file, log_entry)
    
    def log_final_answer(self, request_id: str, question_index: int, question: str, 
                        final_answer: str, response_metadata: Dict[str, Any]):
        """Log the final answer returned to user"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "final_answer",
            "question_index": question_index,
            "question": question,
            "final_answer": final_answer,
            "answer_length": len(final_answer),
            "response_metadata": response_metadata
        }
        
        self._write_to_log(self.request_log_file, log_entry)
    
    def complete_request(self, request_id: str, success: bool, total_time: float, 
                        answers: List[str], error_message: str = None):
        """Mark request as completed with summary"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "request_completed",
            "success": success,
            "total_processing_time_seconds": total_time,
            "num_answers": len(answers),
            "answers": answers,
            "error_message": error_message
        }
        
        self._write_to_log(self.request_log_file, log_entry)
    
    def log_error(self, request_id: str, error_type: str, error_message: str, 
                 context: Dict[str, Any] = None):
        """Log errors with context"""
        error_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self._write_to_log(self.error_log_file, error_entry)
    
    def _write_to_log(self, log_file: Path, entry: Dict[str, Any]):
        """Write entry to JSONL log file"""
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Failed to write to log {log_file}: {e}")
    
    def get_request_summary(self, request_id: str) -> Dict[str, Any]:
        """Get summary of a specific request"""
        summary = {
            "request_id": request_id,
            "request_events": [],
            "chunk_retrievals": [],
            "llm_interactions": [],
            "errors": []
        }
        
        # Read relevant entries from all log files
        for log_file, key in [(self.request_log_file, "request_events"),
                              (self.chunk_log_file, "chunk_retrievals"),
                              (self.llm_log_file, "llm_interactions"),
                              (self.error_log_file, "errors")]:
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            if entry.get("request_id") == request_id:
                                summary[key].append(entry)
                except Exception as e:
                    print(f"⚠️ Error reading {log_file}: {e}")
        
        return summary

# Global logger instance
rag_logger = RAGLogger()
