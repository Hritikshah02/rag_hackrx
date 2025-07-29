#!/usr/bin/env python3
"""
Log Analysis Tool for RAG System
Analyzes comprehensive logs to identify accuracy issues and provide insights
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict, Counter

class RAGLogAnalyzer:
    """Analyzer for RAG system logs"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        
    def analyze_request(self, request_id: str) -> Dict[str, Any]:
        """Analyze a specific request in detail"""
        analysis = {
            "request_id": request_id,
            "request_info": {},
            "document_processing": {},
            "questions": [],
            "overall_stats": {},
            "issues_found": []
        }
        
        # Load all log entries for this request
        entries = self._load_request_entries(request_id)
        
        if not entries:
            analysis["issues_found"].append(f"No log entries found for request {request_id}")
            return analysis
        
        # Analyze request info
        request_entries = entries.get("requests", [])
        start_entry = next((e for e in request_entries if e.get("status") == "started"), None)
        end_entry = next((e for e in request_entries if e.get("event_type") == "request_completed"), None)
        
        if start_entry:
            analysis["request_info"] = {
                "timestamp": start_entry["timestamp"],
                "document_url": start_entry["document_url"],
                "num_questions": start_entry["num_questions"],
                "questions": start_entry["questions"]
            }
        
        if end_entry:
            analysis["overall_stats"] = {
                "success": end_entry["success"],
                "total_time": end_entry["total_processing_time_seconds"],
                "num_answers": end_entry["num_answers"],
                "error_message": end_entry.get("error_message")
            }
        
        # Analyze document processing
        doc_entry = next((e for e in request_entries if e.get("event_type") == "document_processing"), None)
        if doc_entry:
            analysis["document_processing"] = {
                "num_chunks": doc_entry["num_chunks"],
                "total_tokens": doc_entry["total_tokens"],
                "processing_time": doc_entry["processing_time_seconds"]
            }
        
        # Analyze each question
        chunk_entries = entries.get("chunks", [])
        llm_entries = entries.get("llm", [])
        answer_entries = [e for e in request_entries if e.get("event_type") == "final_answer"]
        
        questions_data = defaultdict(dict)
        
        # Group by question index
        for entry in chunk_entries:
            q_idx = entry["question_index"]
            questions_data[q_idx]["chunk_retrieval"] = {
                "question": entry["question"],
                "num_results": entry["num_results"],
                "search_time": entry["search_params"]["search_time_seconds"],
                "top_chunks": entry["search_results"][:3]  # Top 3 chunks
            }
        
        for entry in llm_entries:
            q_idx = entry["question_index"]
            questions_data[q_idx]["llm_interaction"] = {
                "prompt_length": len(entry["prompt"]),
                "response_length": entry["response_length"],
                "parsing_success": entry["parsing_success"],
                "confidence": entry["confidence"],
                "decision": entry["decision"],
                "raw_response": entry["raw_llm_response"][:500] + "..." if len(entry["raw_llm_response"]) > 500 else entry["raw_llm_response"]
            }
        
        for entry in answer_entries:
            q_idx = entry["question_index"]
            questions_data[q_idx]["final_answer"] = {
                "answer": entry["final_answer"],
                "answer_length": entry["answer_length"],
                "llm_processing_time": entry["response_metadata"]["llm_processing_time"]
            }
        
        # Convert to list and add to analysis
        for q_idx in sorted(questions_data.keys()):
            analysis["questions"].append({
                "question_index": q_idx,
                **questions_data[q_idx]
            })
        
        # Find issues
        analysis["issues_found"] = self._find_issues(analysis)
        
        return analysis
    
    def analyze_all_requests(self, limit: int = 10) -> Dict[str, Any]:
        """Analyze multiple recent requests"""
        all_requests = self._get_all_requests()
        
        if not all_requests:
            return {"error": "No requests found in logs"}
        
        # Sort by timestamp and take most recent
        recent_requests = sorted(all_requests, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        summary = {
            "total_requests": len(all_requests),
            "analyzed_requests": len(recent_requests),
            "success_rate": sum(1 for r in recent_requests if r.get("success", False)) / len(recent_requests),
            "average_processing_time": sum(r.get("total_processing_time_seconds", 0) for r in recent_requests) / len(recent_requests),
            "common_issues": self._find_common_issues(recent_requests),
            "parsing_success_rate": self._calculate_parsing_success_rate(),
            "requests": []
        }
        
        for req in recent_requests:
            req_analysis = self.analyze_request(req["request_id"])
            summary["requests"].append({
                "request_id": req["request_id"],
                "timestamp": req["timestamp"],
                "success": req.get("success", False),
                "num_questions": req.get("num_questions", 0),
                "processing_time": req.get("total_processing_time_seconds", 0),
                "major_issues": len(req_analysis["issues_found"])
            })
        
        return summary
    
    def _load_request_entries(self, request_id: str) -> Dict[str, List[Dict]]:
        """Load all log entries for a specific request"""
        entries = {"requests": [], "chunks": [], "llm": [], "errors": []}
        
        # Load from each log file
        log_files = {
            "requests": self.log_dir / "requests.jsonl",
            "chunks": self.log_dir / "chunk_retrievals.jsonl", 
            "llm": self.log_dir / "llm_interactions.jsonl",
            "errors": self.log_dir / "errors.jsonl"
        }
        
        for key, log_file in log_files.items():
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            if entry.get("request_id") == request_id:
                                entries[key].append(entry)
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")
        
        return entries
    
    def _get_all_requests(self) -> List[Dict]:
        """Get all request entries"""
        requests = []
        request_file = self.log_dir / "requests.jsonl"
        
        if request_file.exists():
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get("event_type") == "request_completed":
                            requests.append(entry)
            except Exception as e:
                print(f"Error reading requests: {e}")
        
        return requests
    
    def _find_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Find potential issues in a request analysis"""
        issues = []
        
        # Check overall success
        if not analysis["overall_stats"].get("success", True):
            issues.append(f"Request failed: {analysis['overall_stats'].get('error_message', 'Unknown error')}")
        
        # Check document processing
        doc_processing = analysis.get("document_processing", {})
        if doc_processing.get("num_chunks", 0) == 0:
            issues.append("No chunks were created from document")
        elif doc_processing.get("num_chunks", 0) < 5:
            issues.append(f"Very few chunks created: {doc_processing.get('num_chunks')} (may indicate parsing issues)")
        
        # Check questions
        for q in analysis.get("questions", []):
            q_idx = q["question_index"]
            
            # Check chunk retrieval
            if "chunk_retrieval" in q:
                chunk_data = q["chunk_retrieval"]
                if chunk_data["num_results"] == 0:
                    issues.append(f"Question {q_idx}: No relevant chunks found")
                elif chunk_data["num_results"] < 3:
                    issues.append(f"Question {q_idx}: Few relevant chunks ({chunk_data['num_results']}) - possible relevance issue")
            
            # Check LLM interaction
            if "llm_interaction" in q:
                llm_data = q["llm_interaction"]
                if not llm_data["parsing_success"]:
                    issues.append(f"Question {q_idx}: LLM response parsing failed")
                if llm_data["confidence"] < 0.5:
                    issues.append(f"Question {q_idx}: Low confidence ({llm_data['confidence']:.2f})")
                if llm_data["decision"] in ["UNKNOWN", "Error"]:
                    issues.append(f"Question {q_idx}: LLM decision is {llm_data['decision']}")
            
            # Check final answer
            if "final_answer" in q:
                answer_data = q["final_answer"]
                if len(answer_data["answer"]) < 10:
                    issues.append(f"Question {q_idx}: Very short answer (possible error)")
                if "No answer found" in answer_data["answer"]:
                    issues.append(f"Question {q_idx}: No answer found")
        
        return issues
    
    def _find_common_issues(self, requests: List[Dict]) -> List[str]:
        """Find common issues across multiple requests"""
        issue_counts = Counter()
        
        for req in requests:
            analysis = self.analyze_request(req["request_id"])
            for issue in analysis["issues_found"]:
                # Generalize issue patterns
                if "parsing failed" in issue:
                    issue_counts["LLM response parsing failures"] += 1
                elif "No relevant chunks" in issue:
                    issue_counts["No relevant chunks found"] += 1
                elif "Low confidence" in issue:
                    issue_counts["Low confidence responses"] += 1
                elif "Very short answer" in issue:
                    issue_counts["Very short answers"] += 1
                elif "No answer found" in issue:
                    issue_counts["No answer found responses"] += 1
        
        return [f"{issue}: {count} times" for issue, count in issue_counts.most_common(5)]
    
    def _calculate_parsing_success_rate(self) -> float:
        """Calculate LLM response parsing success rate"""
        llm_file = self.log_dir / "llm_interactions.jsonl"
        if not llm_file.exists():
            return 0.0
        
        total = 0
        successful = 0
        
        try:
            with open(llm_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    total += 1
                    if entry.get("parsing_success", False):
                        successful += 1
        except Exception as e:
            print(f"Error calculating parsing success rate: {e}")
            return 0.0
        
        return successful / total if total > 0 else 0.0

def main():
    """Main function for command line usage"""
    analyzer = RAGLogAnalyzer()
    
    if len(sys.argv) > 1:
        # Analyze specific request
        request_id = sys.argv[1]
        print(f"Analyzing request: {request_id}")
        analysis = analyzer.analyze_request(request_id)
        print(json.dumps(analysis, indent=2))
    else:
        # Analyze recent requests
        print("Analyzing recent requests...")
        summary = analyzer.analyze_all_requests()
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
