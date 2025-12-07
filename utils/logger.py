# Adithya Vardhan 32956089
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class SystemLogger:
    """
    Comprehensive logging system for tracking status changes and LLM API calls.
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files (default: ./logs)
        """
        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"system_log_{timestamp}.json")
        
        # Initialize log data structure
        self.logs = {
            "session_id": timestamp,
            "start_time": datetime.now().isoformat(),
            "status_changes": [],
            "llm_calls": [],
            "errors": [],
            "summary": {}
        }
        
        # Setup Python logging
        self.logger = logging.getLogger("MultiAgentSystem")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, f"system_{timestamp}.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_status_change(self, component: str, old_status: str, new_status: str, 
                         task_id: Optional[str] = None, details: Optional[Dict] = None):
        """
        Log a status change event.
        
        Args:
            component: Component name (e.g., "main_agent", "task_1")
            old_status: Previous status
            new_status: New status
            task_id: Optional task ID
            details: Optional additional details
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "task_id": task_id,
            "old_status": old_status,
            "new_status": new_status,
            "details": details or {}
        }
        
        self.logs["status_changes"].append(event)
        
        log_msg = f"Status Change: {component}"
        if task_id:
            log_msg += f" (Task: {task_id})"
        log_msg += f" {old_status} -> {new_status}"
        
        self.logger.info(log_msg)
        self._save_logs()
    
    def log_llm_call(self, agent: str, purpose: str, model: str, 
                     prompt_tokens: int, completion_tokens: int, 
                     total_tokens: int, response_time: float,
                     cost_estimate: Optional[float] = None,
                     task_id: Optional[str] = None):
        """
        Log an LLM API call with token usage.
        
        Args:
            agent: Agent making the call (e.g., "coding_agent")
            purpose: Purpose of the call (e.g., "code_generation", "task_breakdown")
            model: Model name (e.g., "gemini-pro")
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens used
            response_time: Response time in seconds
            cost_estimate: Optional cost estimate in USD
            task_id: Optional task ID
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "task_id": task_id,
            "purpose": purpose,
            "model": model,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens
            },
            "response_time_seconds": response_time,
            "cost_estimate_usd": cost_estimate
        }
        
        self.logs["llm_calls"].append(event)
        
        log_msg = (f"LLM Call: {agent} - {purpose} | "
                  f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}) | "
                  f"Time: {response_time:.2f}s")
        if cost_estimate:
            log_msg += f" | Cost: ${cost_estimate:.4f}"
        
        self.logger.info(log_msg)
        self._save_logs()
    
    def log_error(self, component: str, error_type: str, error_message: str, 
                  task_id: Optional[str] = None, stack_trace: Optional[str] = None):
        """
        Log an error event.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            task_id: Optional task ID
            stack_trace: Optional stack trace
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "task_id": task_id,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace
        }
        
        self.logs["errors"].append(event)
        
        log_msg = f"Error in {component}: {error_type} - {error_message}"
        self.logger.error(log_msg)
        self._save_logs()
    
    def generate_summary(self) -> Dict:
        """
        Generate a summary of all logged events.
        
        Returns:
            Dict containing summary statistics
        """
        # Calculate LLM statistics
        total_llm_calls = len(self.logs["llm_calls"])
        total_tokens = sum(call["tokens"]["total"] for call in self.logs["llm_calls"])
        total_prompt_tokens = sum(call["tokens"]["prompt"] for call in self.logs["llm_calls"])
        total_completion_tokens = sum(call["tokens"]["completion"] for call in self.logs["llm_calls"])
        total_cost = sum(call.get("cost_estimate_usd", 0) for call in self.logs["llm_calls"])
        avg_response_time = (sum(call["response_time_seconds"] for call in self.logs["llm_calls"]) / total_llm_calls) if total_llm_calls > 0 else 0
        
        # Group by agent
        calls_by_agent = {}
        tokens_by_agent = {}
        for call in self.logs["llm_calls"]:
            agent = call["agent"]
            calls_by_agent[agent] = calls_by_agent.get(agent, 0) + 1
            tokens_by_agent[agent] = tokens_by_agent.get(agent, 0) + call["tokens"]["total"]
        
        # Group by purpose
        calls_by_purpose = {}
        tokens_by_purpose = {}
        for call in self.logs["llm_calls"]:
            purpose = call["purpose"]
            calls_by_purpose[purpose] = calls_by_purpose.get(purpose, 0) + 1
            tokens_by_purpose[purpose] = tokens_by_purpose.get(purpose, 0) + call["tokens"]["total"]
        
        # Status change statistics
        total_status_changes = len(self.logs["status_changes"])
        status_changes_by_component = {}
        for change in self.logs["status_changes"]:
            component = change["component"]
            status_changes_by_component[component] = status_changes_by_component.get(component, 0) + 1
        
        # Error statistics
        total_errors = len(self.logs["errors"])
        errors_by_component = {}
        for error in self.logs["errors"]:
            component = error["component"]
            errors_by_component[component] = errors_by_component.get(component, 0) + 1
        
        summary = {
            "session_info": {
                "session_id": self.logs["session_id"],
                "start_time": self.logs["start_time"],
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - datetime.fromisoformat(self.logs["start_time"])).total_seconds()
            },
            "llm_statistics": {
                "total_calls": total_llm_calls,
                "total_tokens": total_tokens,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_cost_usd": total_cost,
                "average_response_time_seconds": avg_response_time,
                "calls_by_agent": calls_by_agent,
                "tokens_by_agent": tokens_by_agent,
                "calls_by_purpose": calls_by_purpose,
                "tokens_by_purpose": tokens_by_purpose
            },
            "status_changes": {
                "total_changes": total_status_changes,
                "changes_by_component": status_changes_by_component
            },
            "errors": {
                "total_errors": total_errors,
                "errors_by_component": errors_by_component
            }
        }
        
        self.logs["summary"] = summary
        self._save_logs()
        
        return summary
    
    def get_formatted_summary(self) -> str:
        """
        Get a human-readable formatted summary.
        
        Returns:
            Formatted summary string
        """
        summary = self.generate_summary()
        
        output = []
        output.append("=" * 80)
        output.append("MULTI-AGENT SYSTEM - SESSION SUMMARY")
        output.append("=" * 80)
        output.append("")
        
        # Session Info
        output.append("SESSION INFORMATION")
        output.append("-" * 80)
        output.append(f"Session ID: {summary['session_info']['session_id']}")
        output.append(f"Start Time: {summary['session_info']['start_time']}")
        output.append(f"End Time: {summary['session_info']['end_time']}")
        output.append(f"Duration: {summary['session_info']['duration_seconds']:.2f} seconds")
        output.append("")
        
        # LLM Statistics
        output.append("LLM API USAGE")
        output.append("-" * 80)
        llm = summary['llm_statistics']
        output.append(f"Total API Calls: {llm['total_calls']}")
        output.append(f"Total Tokens: {llm['total_tokens']:,}")
        output.append(f"  - Prompt Tokens: {llm['prompt_tokens']:,}")
        output.append(f"  - Completion Tokens: {llm['completion_tokens']:,}")
        output.append(f"Average Response Time: {llm['average_response_time_seconds']:.2f}s")
        output.append(f"Estimated Total Cost: ${llm['total_cost_usd']:.4f}")
        output.append("")
        
        output.append("Calls by Agent:")
        for agent, count in llm['calls_by_agent'].items():
            tokens = llm['tokens_by_agent'][agent]
            output.append(f"  - {agent}: {count} calls, {tokens:,} tokens")
        output.append("")
        
        output.append("Calls by Purpose:")
        for purpose, count in llm['calls_by_purpose'].items():
            tokens = llm['tokens_by_purpose'][purpose]
            output.append(f"  - {purpose}: {count} calls, {tokens:,} tokens")
        output.append("")
        
        # Status Changes
        output.append("STATUS CHANGES")
        output.append("-" * 80)
        output.append(f"Total Status Changes: {summary['status_changes']['total_changes']}")
        for component, count in summary['status_changes']['changes_by_component'].items():
            output.append(f"  - {component}: {count} changes")
        output.append("")
        
        # Errors
        if summary['errors']['total_errors'] > 0:
            output.append("ERRORS")
            output.append("-" * 80)
            output.append(f"Total Errors: {summary['errors']['total_errors']}")
            for component, count in summary['errors']['errors_by_component'].items():
                output.append(f"  - {component}: {count} errors")
            output.append("")
        
        output.append("=" * 80)
        
        return "\n".join(output)
    
    def _save_logs(self):
        """Save logs to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_log_file_path(self) -> str:
        """Get the path to the current log file."""
        return self.log_file


# Global logger instance
_global_logger: Optional[SystemLogger] = None

def get_logger() -> SystemLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SystemLogger()
    return _global_logger

def reset_logger():
    """Reset the global logger (useful for new sessions)."""
    global _global_logger
    _global_logger = None
