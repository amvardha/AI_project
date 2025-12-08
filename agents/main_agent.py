# Adithya Vardhan 32956089
import os
import json
import asyncio
import time
import traceback
from typing import List, Dict, Callable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import google.generativeai as genai
from dotenv import load_dotenv
from utils.logger import get_logger
from agents.coding_agent import CodingAgent

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class MainAgent:
    """
    Main orchestrator agent that coordinates the entire application generation process.
    """
    
    def __init__(self, status_callback: Callable[[str], None] = None):
        """
        Initialize the main agent.
        
        Args:
            status_callback: Optional callback function to report status updates to UI
        """
        self.status_callback = status_callback
        self.orchestrator_session = None
        self.file_session = None
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.logger = get_logger()
        self.current_status = "initialized"
        self.coding_agent = CodingAgent()
        
    def report_status(self, status: str):
        """Report status to UI if callback is provided."""
        old_status = self.current_status
        self.current_status = status
        
        # Log status change
        self.logger.log_status_change(
            component="main_agent",
            old_status=old_status,
            new_status=status
        )
        
        if self.status_callback:
            self.status_callback(status)
    
    def _get_orchestrator_params(self):
        """Get orchestrator MCP server parameters."""
        return StdioServerParameters(
            command="python",
            args=[os.getenv("ORCHESTRATOR_MCP_PATH", "../MCPs/orchestrator_mcp.py")],
            env=None
        )
    
    def _get_file_params(self):
        """Get file MCP server parameters."""
        return StdioServerParameters(
            command="python",
            args=[os.getenv("FILE_MCP_PATH", "../MCPs/files_mcp.py")],
            env=None
        )
    
    async def parse_requirements(self, requirements_file: str) -> str:
        """
        Parse requirements file using orchestrator MCP.
        
        Args:
            requirements_file: Path to requirements file
            
        Returns:
            str: Requirements content
        """
        result = await self.orchestrator_session.call_tool(
            "read_requirements",
            arguments={"file_path": requirements_file}
        )
        return result.content[0].text
    
    async def break_down_tasks(self, requirements: str) -> List[Dict]:
        """
        Use LLM to break down requirements into actionable tasks.
        
        Args:
            requirements: Requirements text
            
        Returns:
            List of task dictionaries
        """
        prompt = f"""
        You are a software architect. Given the following requirements, break them down into specific, actionable coding tasks.
        
        Requirements:
        {requirements}
        
        Create a list of tasks in JSON format. Each task should have:
        - id: A unique identifier (task_1, task_2, etc.)
        - description: Clear description of what needs to be coded
        - status: Set to "pending"
        - assigned_to: Set to "coding_agent"
        
        Return ONLY a valid JSON array of tasks, nothing else.
        """
        
        # Track LLM call
        start_time = time.time()
        response = self.model.generate_content(prompt)
        response_time = time.time() - start_time
        
        # Estimate token counts (Gemini doesn't provide exact counts in response)
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        completion_tokens = len(response.text.split()) * 1.3
        total_tokens = int(prompt_tokens + completion_tokens)
        
        # Log LLM call (Gemini Pro pricing: ~$0.00025 per 1K tokens for input, $0.0005 per 1K tokens for output)
        cost_estimate = (prompt_tokens / 1000 * 0.00025) + (completion_tokens / 1000 * 0.0005)
        self.logger.log_llm_call(
            agent="main_agent",
            purpose="task_breakdown",
            model=GEMINI_MODEL,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total_tokens,
            response_time=response_time,
            cost_estimate=cost_estimate
        )
        
        tasks_json = response.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in tasks_json:
            tasks_json = tasks_json.split("```json")[1].split("```")[0].strip()
        elif "```" in tasks_json:
            tasks_json = tasks_json.split("```")[1].split("```")[0].strip()
        
        tasks = json.loads(tasks_json)

        self.logger.info(f"Tasks: {tasks}")
        
        # Ensure all tasks have required fields
        for task in tasks:
            if "result" not in task:
                task["result"] = ""
            if "errors" not in task:
                task["errors"] = []
        
        return tasks
    
    async def set_tasks_in_orchestrator(self, tasks: List[Dict]) -> str:
        """
        Initialize tasks in orchestrator MCP.
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            str: Confirmation message
        """
        result = await self.orchestrator_session.call_tool(
            "set_tasks",
            arguments={"tasks": tasks}
        )
        self.logger.info(f"Tasks set in orchestrator: {len(tasks)}")
        return result.content[0].text
    
    async def get_tasks_from_orchestrator(self) -> List[Dict]:
        """Get all tasks from orchestrator."""
        result = await self.orchestrator_session.call_tool("get_tasks", arguments={})
        tasks_json = result.content[0].text
        return json.loads(tasks_json)
    
    async def execute_task_with_coding_agent(self, task: Dict, project_path: str) -> bool:
        """
        Execute a single task using the coding agent.
        
        Args:
            task: Task dictionary
            project_path: Path to the project directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        
        # Update task status to "coding"
        self.logger.log_status_change(
            component="task",
            old_status=task.get("status", "pending"),
            new_status="coding",
            task_id=task["id"]
        )
        
        await self.orchestrator_session.call_tool(
            "update_task_status",
            arguments={
                "task_id": task["id"],
                "status": "coding",
                "result": "",
                "errors": None
            }
        )
        
        self.logger.info(f"Calling coding agent for task {task['description']}")
        # Execute the task
        success, result, errors = await self.coding_agent.execute_task(task, project_path)
        
        # Update task status based on result
        if success:
            self.logger.log_status_change(
                component="task",
                old_status="coding",
                new_status="testing required",
                task_id=task["id"]
            )
            
            await self.orchestrator_session.call_tool(
                "update_task_status",
                arguments={
                    "task_id": task["id"],
                    "status": "testing required",
                    "result": result,
                    "errors": None
                }
            )
        else:
            self.logger.log_status_change(
                component="task",
                old_status="coding",
                new_status="error",
                task_id=task["id"],
                details={"errors": errors}
            )
            
            self.logger.log_error(
                component="coding_agent",
                error_type="task_execution_failed",
                error_message=result,
                task_id=task["id"]
            )
            
            await self.orchestrator_session.call_tool(
                "update_task_status",
                arguments={
                    "task_id": task["id"],
                    "status": "error",
                    "result": result,
                    "errors": errors
                }
            )
        
        return success
    
    async def run_testing_agent(self, project_path: str, requirements: str = "") -> Dict:
        """
        Run testing agent on all completed tasks.
        
        Args:
            project_path: Path to the project directory
            requirements: Original requirements text for validation (optional)
            
        Returns:
            Dict with testing results
        """
        from agents.testing_agent import TestingAgent
        
        testing_agent = TestingAgent()
        
        # Get all tasks
        tasks = await self.get_tasks_from_orchestrator()
        
        # Run tests with requirements validation
        results = await testing_agent.run_tests(tasks, project_path, requirements)
        
        # Update task statuses based on test results
        for task_id, test_result in results.items():
            if test_result["passed"]:
                self.logger.log_status_change(
                    component="task",
                    old_status="testing required",
                    new_status="completed",
                    task_id=task_id
                )
                
                await self.orchestrator_session.call_tool(
                    "update_task_status",
                    arguments={
                        "task_id": task_id,
                        "status": "completed",
                        "result": test_result["message"],
                        "errors": None
                    }
                )
            else:
                self.logger.log_status_change(
                    component="task",
                    old_status="testing required",
                    new_status="error",
                    task_id=task_id,
                    details={"errors": test_result.get("errors", [])}
                )
                
                await self.orchestrator_session.call_tool(
                    "update_task_status",
                    arguments={
                        "task_id": task_id,
                        "status": "error",
                        "result": test_result["message"],
                        "errors": test_result.get("errors", [])
                    }
                )
        
        return results
    
    async def run(self, requirements_file: str, project_path: str) -> Dict:
        """
        Main execution flow.
        
        Args:
            requirements_file: Path to requirements file
            project_path: Path where project should be generated
            
        Returns:
            Dict with execution results
        """
        # Use async with to properly manage MCP connections
        async with stdio_client(self._get_orchestrator_params()) as (orchestrator_read, orchestrator_write):
            async with stdio_client(self._get_file_params()) as (file_read, file_write):
                async with ClientSession(orchestrator_read, orchestrator_write) as orchestrator_session:
                    async with ClientSession(file_read, file_write) as file_session:
                        # Store sessions for use in other methods
                        self.orchestrator_session = orchestrator_session
                        self.file_session = file_session
                        
                        # Initialize the sessions to ensure they're ready
                        await orchestrator_session.initialize()
                        await file_session.initialize()
                        
                        try:
                            # Step 1: Parse requirements
                            self.report_status("Planning")
                            requirements = await self.parse_requirements(requirements_file)
                            
                            # Step 2: Break down into tasks
                            tasks = await self.break_down_tasks(requirements)
                            
                            # Step 3: Initialize tasks in orchestrator
                            await self.set_tasks_in_orchestrator(tasks)
                            
                            # Step 4: Execute tasks sequentially
                            self.report_status("Coding")
                            max_retries = 1
                            
                            for task in tasks:
                                retries = 0
                                success = False
                                
                                while retries < max_retries and not success:
                                    success = await self.execute_task_with_coding_agent(task, project_path)
                                    
                                    if not success:
                                        retries += 1
                                        if retries < max_retries:
                                            # Retry the task
                                            continue
                                        else:
                                            # Max retries reached, move to next task
                                            break
                            
                            # Step 5: Run testing with requirements validation
                            self.report_status("Testing")
                            test_results = await self.run_testing_agent(project_path, requirements)
                            
                            # Step 6: Final status
                            self.report_status("Completed")
                            
                            # Get final task states
                            final_tasks = await self.get_tasks_from_orchestrator()
                            
                            # Generate summary
                            summary = self.logger.generate_summary()
                            
                            return {
                                "success": True,
                                "tasks": final_tasks,
                                "test_results": test_results,
                                "log_summary": summary,
                                "log_file": self.logger.get_log_file_path()
                            }
                        
                        except Exception as e:
                            self.report_status("Error")
                            self.logger.log_error(
                                component="main_agent",
                                error_type="execution_error",
                                error_message=str(e),
                                stack_trace=traceback.format_exc()
                            )
                            
                            summary = self.logger.generate_summary()
                            
                            return {
                                "success": False,
                                "error": str(e),
                                "log_summary": summary,
                                "log_file": self.logger.get_log_file_path()
                            }


async def main():
    """Test the main agent."""
    agent = MainAgent(status_callback=lambda s: print(f"Status: {s}"))
    
    # Create a test requirements file
    test_req_file = "/tmp/test_requirements.txt"
    with open(test_req_file, 'w') as f:
        f.write("Create a simple HTML application with a home page that displays 'Hello World'")
    
    result = await agent.run(test_req_file, "/tmp/test_project")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
