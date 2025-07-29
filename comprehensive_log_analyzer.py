#!/usr/bin/env python3
"""
Comprehensive RAG System Log Analyzer
Creates a single consolidated document with all request details for easy problem identification
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict, Counter

class ComprehensiveRAGAnalyzer:
    """Creates comprehensive single-document analysis of RAG system performance"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        
    def create_comprehensive_report(self, request_id: str = None, output_file: str = None) -> str:
        """Create a comprehensive analysis report in a single document"""
        
        if request_id:
            # Analyze specific request
            report = self._analyze_single_request(request_id)
        else:
            # Analyze most recent request
            recent_requests = self._get_recent_requests(1)
            if not recent_requests:
                return "❌ No requests found in logs"
            request_id = recent_requests[0]["request_id"]
            report = self._analyze_single_request(request_id)
        
        # Format as markdown for easy reading
        markdown_report = self._format_as_markdown(report)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            print(f"📝 Report saved to: {output_path}")
        
        return markdown_report
    
    def _analyze_single_request(self, request_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of a single request"""
        
        # Load all data for this request
        entries = self._load_all_entries(request_id)
        
        analysis = {
            "request_id": request_id,
            "timestamp": None,
            "document_info": {},
            "questions_analysis": [],
            "performance_metrics": {},
            "problems_identified": [],
            "recommendations": []
        }
        
        # Extract request info
        request_entries = entries.get("requests", [])
        start_entry = next((e for e in request_entries if e.get("status") == "started"), None)
        complete_entry = next((e for e in request_entries if e.get("event_type") == "request_completed"), None)
        doc_entry = next((e for e in request_entries if e.get("event_type") == "document_processing"), None)
        
        if start_entry:
            analysis["timestamp"] = start_entry["timestamp"]
            analysis["document_info"] = {
                "url": start_entry["document_url"],
                "questions": start_entry["questions"],
                "num_questions": start_entry["num_questions"]
            }
        
        if doc_entry:
            analysis["document_info"].update({
                "num_chunks": doc_entry["num_chunks"],
                "total_tokens": doc_entry["total_tokens"],
                "processing_time": doc_entry["processing_time_seconds"]
            })
        
        if complete_entry:
            analysis["performance_metrics"] = {
                "success": complete_entry["success"],
                "total_time_seconds": complete_entry["total_processing_time_seconds"],
                "error_message": complete_entry.get("error_message")
            }
        
        # Analyze each question in detail
        chunk_entries = entries.get("chunks", [])
        llm_entries = entries.get("llm", [])
        answer_entries = [e for e in request_entries if e.get("event_type") == "final_answer"]
        
        # Group by question index
        questions_data = defaultdict(dict)
        
        for entry in chunk_entries:
            q_idx = entry["question_index"]
            questions_data[q_idx]["retrieval"] = {
                "question": entry["question"],
                "search_time": entry["search_params"]["search_time_seconds"],
                "num_chunks_found": entry["num_results"],
                "chunks": entry["search_results"]
            }
        
        for entry in llm_entries:
            q_idx = entry["question_index"]
            questions_data[q_idx]["llm"] = {
                "prompt": entry["prompt"],
                "raw_response": entry["raw_llm_response"],
                "parsed_response": entry["parsed_response"],
                "parsing_success": entry["parsing_success"],
                "processing_details": entry["processing_details"],
                "confidence": entry.get("confidence", 0),
                "decision": entry.get("decision", "UNKNOWN")
            }
        
        for entry in answer_entries:
            q_idx = entry["question_index"]
            questions_data[q_idx]["final"] = {
                "answer": entry["final_answer"],
                "answer_length": entry["answer_length"],
                "processing_time": entry["response_metadata"]["llm_processing_time"]
            }
        
        # Convert to list and analyze each question
        for q_idx in sorted(questions_data.keys()):
            question_analysis = self._analyze_question(q_idx, questions_data[q_idx])
            analysis["questions_analysis"].append(question_analysis)
        
        # Identify overall problems
        analysis["problems_identified"] = self._identify_problems(analysis)
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_question(self, q_idx: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed analysis of a single question"""
        
        question_analysis = {
            "question_index": q_idx,
            "question_text": data.get("retrieval", {}).get("question", "Unknown"),
            "retrieval_analysis": {},
            "llm_analysis": {},
            "final_analysis": {},
            "issues": [],
            "quality_score": 0
        }
        
        # Analyze retrieval
        if "retrieval" in data:
            retrieval = data["retrieval"]
            question_analysis["retrieval_analysis"] = {
                "chunks_found": retrieval["num_chunks_found"],
                "search_time": retrieval["search_time"],
                "chunk_quality": self._assess_chunk_quality(retrieval["chunks"]),
                "top_chunk_scores": [c.get("score", 0) for c in retrieval["chunks"][:3]]
            }
            
            # Check retrieval issues
            if retrieval["num_chunks_found"] == 0:
                question_analysis["issues"].append("❌ No relevant chunks found")
            elif max(question_analysis["retrieval_analysis"]["top_chunk_scores"]) < 0.5:
                question_analysis["issues"].append("⚠️ Low relevance scores for retrieved chunks")
        
        # Analyze LLM processing
        if "llm" in data:
            llm = data["llm"]
            question_analysis["llm_analysis"] = {
                "prompt_length": len(llm["prompt"]),
                "response_length": len(llm["raw_response"]),
                "parsing_successful": llm["parsing_success"],
                "confidence": llm["confidence"],
                "decision": llm["decision"],
                "context_chunks_used": llm["prompt"].count("Chunk"),
                "prompt_preview": llm["prompt"],
                "raw_response_preview": llm["raw_response"]
            }
            
            # Check LLM issues
            if not llm["parsing_success"]:
                question_analysis["issues"].append("❌ LLM response parsing failed")
            if llm["confidence"] < 0.6:
                question_analysis["issues"].append(f"⚠️ Low confidence: {llm['confidence']:.2f}")
            if llm["decision"] in ["UNKNOWN", "Error"]:
                question_analysis["issues"].append(f"❌ LLM decision: {llm['decision']}")
        
        # Analyze final answer
        if "final" in data:
            final = data["final"]
            question_analysis["final_analysis"] = {
                "answer": final["answer"],
                "answer_length": final["answer_length"],
                "processing_time": final["processing_time"]
            }
            
            # Check answer quality
            if final["answer_length"] < 20:
                question_analysis["issues"].append("⚠️ Very short answer (likely incomplete)")
            if "No answer found" in final["answer"]:
                question_analysis["issues"].append("❌ No answer found")
            if "error" in final["answer"].lower():
                question_analysis["issues"].append("❌ Error in answer")
        
        # Calculate quality score
        question_analysis["quality_score"] = self._calculate_question_quality_score(question_analysis)
        
        return question_analysis
    
    def _assess_chunk_quality(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of retrieved chunks"""
        if not chunks:
            return {"average_score": 0, "score_distribution": [], "content_analysis": "No chunks"}
        
        scores = [c.get("score", 0) for c in chunks]
        content_lengths = [len(c.get("full_content", "")) for c in chunks]
        
        return {
            "average_score": sum(scores) / len(scores),
            "score_distribution": scores,
            "average_content_length": sum(content_lengths) / len(content_lengths),
            "num_chunks": len(chunks),
            "top_chunk_preview": chunks[0].get("content_preview", "No content") if chunks else "No chunks"
        }
    
    def _calculate_question_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate a quality score for a question (0-100)"""
        score = 100
        
        # Deduct for each issue
        score -= len(analysis["issues"]) * 20
        
        # Bonus for good retrieval
        if analysis.get("retrieval_analysis", {}).get("chunks_found", 0) >= 3:
            score += 10
        
        # Bonus for successful parsing
        if analysis.get("llm_analysis", {}).get("parsing_successful", False):
            score += 15
        
        # Bonus for high confidence
        confidence = analysis.get("llm_analysis", {}).get("confidence", 0)
        score += confidence * 20
        
        # Bonus for reasonable answer length
        answer_length = analysis.get("final_analysis", {}).get("answer_length", 0)
        if 50 <= answer_length <= 500:
            score += 10
        
        return max(0, min(100, score))
    
    def _identify_problems(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify major problems in the request"""
        problems = []
        
        # Overall problems
        if not analysis["performance_metrics"].get("success", True):
            problems.append(f"🚨 REQUEST FAILED: {analysis['performance_metrics'].get('error_message', 'Unknown error')}")
        
        # Document processing problems
        doc_info = analysis.get("document_info", {})
        if doc_info.get("num_chunks", 0) < 5:
            problems.append(f"🚨 DOCUMENT PROCESSING: Only {doc_info.get('num_chunks', 0)} chunks created (document may not be properly parsed)")
        
        # Question-specific problems
        low_quality_questions = [q for q in analysis["questions_analysis"] if q["quality_score"] < 50]
        if len(low_quality_questions) > len(analysis["questions_analysis"]) / 2:
            problems.append(f"🚨 ACCURACY ISSUE: {len(low_quality_questions)}/{len(analysis['questions_analysis'])} questions have low quality scores")
        
        parsing_failures = [q for q in analysis["questions_analysis"] if not q.get("llm_analysis", {}).get("parsing_successful", True)]
        if parsing_failures:
            problems.append(f"🚨 LLM PARSING: {len(parsing_failures)} questions failed to parse LLM responses")
        
        return problems
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []
        
        # Check common patterns
        questions = analysis["questions_analysis"]
        
        # Parsing issues
        parsing_failures = [q for q in questions if not q.get("llm_analysis", {}).get("parsing_successful", True)]
        if parsing_failures:
            recommendations.append("🔧 FIX LLM PROMPT: Simplify JSON structure and add more explicit parsing instructions")
        
        # Low relevance
        low_relevance = [q for q in questions if max(q.get("retrieval_analysis", {}).get("top_chunk_scores", [0])) < 0.5]
        if low_relevance:
            recommendations.append("🔧 IMPROVE RETRIEVAL: Consider better embedding model or query preprocessing")
        
        # Short answers
        short_answers = [q for q in questions if q.get("final_analysis", {}).get("answer_length", 0) < 30]
        if short_answers:
            recommendations.append("🔧 ENHANCE CONTEXT: Provide more detailed context in prompts or increase chunk overlap")
        
        # Low confidence
        low_confidence = [q for q in questions if q.get("llm_analysis", {}).get("confidence", 0) < 0.6]
        if low_confidence:
            recommendations.append("🔧 BOOST CONFIDENCE: Add relevance validation and improve prompt clarity")
        
        return recommendations
    
    def _format_as_markdown(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as readable markdown"""
        
        md = f"""# 📊 RAG System Analysis Report

## 🆔 Request Information
- **Request ID**: {analysis['request_id']}
- **Timestamp**: {analysis['timestamp']}
- **Success**: {'✅ Yes' if analysis['performance_metrics'].get('success', False) else '❌ No'}
- **Total Processing Time**: {analysis['performance_metrics'].get('total_time_seconds', 0):.2f} seconds

## 📄 Document Information
- **URL**: {analysis['document_info'].get('url', 'Unknown')}
- **Chunks Created**: {analysis['document_info'].get('num_chunks', 0)}
- **Total Tokens**: {analysis['document_info'].get('total_tokens', 0)}
- **Processing Time**: {analysis['document_info'].get('processing_time', 0):.2f} seconds

## 🚨 Major Problems Identified
"""
        
        if analysis['problems_identified']:
            for problem in analysis['problems_identified']:
                md += f"- {problem}\n"
        else:
            md += "- ✅ No major problems detected\n"
        
        md += f"""
## 💡 Recommendations
"""
        
        if analysis['recommendations']:
            for rec in analysis['recommendations']:
                md += f"- {rec}\n"
        else:
            md += "- ✅ No specific recommendations\n"
        
        md += f"""
## 📝 Question-by-Question Analysis

"""
        
        for q_analysis in analysis['questions_analysis']:
            md += f"""### Question {q_analysis['question_index']}: Quality Score {q_analysis['quality_score']:.0f}/100

**Question**: {q_analysis['question_text']}

#### 🔍 Retrieval Analysis
- **Chunks Found**: {q_analysis['retrieval_analysis'].get('chunks_found', 0)}
- **Search Time**: {q_analysis['retrieval_analysis'].get('search_time', 0):.3f}s
- **Top Scores**: {q_analysis['retrieval_analysis'].get('top_chunk_scores', [])}
- **Average Score**: {q_analysis['retrieval_analysis'].get('chunk_quality', {}).get('average_score', 0):.3f}

**Top Chunk Content**:
```
{q_analysis['retrieval_analysis'].get('chunk_quality', {}).get('top_chunk_preview', 'No content')}
```

#### 🤖 LLM Analysis
- **Parsing Success**: {'✅' if q_analysis['llm_analysis'].get('parsing_successful', False) else '❌'}
- **Confidence**: {q_analysis['llm_analysis'].get('confidence', 0):.2f}
- **Decision**: {q_analysis['llm_analysis'].get('decision', 'Unknown')}
- **Prompt Length**: {q_analysis['llm_analysis'].get('prompt_length', 0)} characters
- **Response Length**: {q_analysis['llm_analysis'].get('response_length', 0)} characters

**LLM Prompt (Preview)**:
```
{q_analysis['llm_analysis'].get('prompt_preview', 'No prompt')}
```

**LLM Response (Preview)**:
```
{q_analysis['llm_analysis'].get('raw_response_preview', 'No response')}
```

#### 📤 Final Answer
- **Answer Length**: {q_analysis['final_analysis'].get('answer_length', 0)} characters
- **Processing Time**: {q_analysis['final_analysis'].get('processing_time', 0):.3f}s

**Final Answer**:
```
{q_analysis['final_analysis'].get('answer', 'No answer')}
```

#### ⚠️ Issues Found
"""
            
            if q_analysis['issues']:
                for issue in q_analysis['issues']:
                    md += f"- {issue}\n"
            else:
                md += "- ✅ No issues detected\n"
            
            md += "\n---\n\n"
        
        return md
    
    def _load_all_entries(self, request_id: str) -> Dict[str, List[Dict]]:
        """Load all entries for a request from all log files"""
        entries = {"requests": [], "chunks": [], "llm": [], "errors": []}
        
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
    
    def _get_recent_requests(self, limit: int = 5) -> List[Dict]:
        """Get most recent requests"""
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
        
        return sorted(requests, key=lambda x: x["timestamp"], reverse=True)[:limit]

def main():
    """Command line interface"""
    analyzer = ComprehensiveRAGAnalyzer()
    
    request_id = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else "rag_analysis_report.md"
    
    print("🔍 Creating comprehensive RAG analysis report...")
    
    try:
        report = analyzer.create_comprehensive_report(request_id, output_file)
        print("✅ Analysis complete!")
        print(f"📄 Report saved to: {output_file}")
        
        # Also print to console
        print("\n" + "="*50)
        print(report)
        
    except Exception as e:
        print(f"❌ Error creating report: {e}")

if __name__ == "__main__":
    main()
