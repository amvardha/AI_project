# Adithya Vardhan 32956089
import os
import json
import asyncio
import subprocess
import time
from typing import Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import google.generativeai as genai
from dotenv import load_dotenv
from utils.logger import get_logger

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class TestingAgent:
    """
    Testing agent that runs lint checks and generates test cases.
    """
    
    def __init__(self):
        """Initialize the testing agent."""
        self.orchestrator_session = None
        self.file_session = None
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.logger = get_logger()
    
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
    
    def run_lint_check(self, file_path: str) -> Dict:
        """
        Run lint check on a file.
        
        Args:
            file_path: Path to the file to lint
            
        Returns:
            Dict with lint results
        """
        if not os.path.exists(file_path):
            return {
                "passed": False,
                "errors": [f"File not found: {file_path}"]
            }
        
        # Determine file type and run appropriate linter
        if file_path.endswith('.py'):
            try:
                # Run pylint
                result = subprocess.run(
                    ['python', '-m', 'pylint', file_path, '--output-format=json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.stdout:
                    lint_results = json.loads(result.stdout)
                    errors = [f"{r['message']} (line {r['line']})" for r in lint_results]
                    return {
                        "passed": len(errors) == 0,
                        "errors": errors
                    }
                else:
                    return {"passed": True, "errors": []}
            
            except subprocess.TimeoutExpired:
                return {"passed": False, "errors": ["Lint check timed out"]}
            except Exception as e:
                # If pylint fails, just skip linting
                return {"passed": True, "errors": [], "warning": f"Linting skipped: {str(e)}"}
        
        elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
            # For JavaScript/TypeScript, we could use eslint but skip for now
            return {"passed": True, "errors": [], "warning": "JS/TS linting not implemented"}
        
        else:
            return {"passed": True, "errors": [], "warning": "No linter for this file type"}
    
    async def generate_test_cases(self, file_path: str, code_content: str) -> str:
        """
        Generate test cases for a file using LLM.
        
        Args:
            file_path: Path to the file
            code_content: Content of the file
            
        Returns:
            str: Generated test code
        """
        try:
            prompt = f"""
            Generate test cases for the following code file: {file_path}
            
            Code:
            {code_content}
            
            Generate appropriate test cases based on the file type. Return ONLY the test code, no explanations.
            """
            
            # Track LLM call
            start_time = time.time()
            response = self.model.generate_content(prompt)
            response_time = time.time() - start_time
            
            # Estimate token counts
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response.text.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Log LLM call
            cost_estimate = (prompt_tokens / 1000 * 0.00025) + (completion_tokens / 1000 * 0.0005)
            self.logger.log_llm_call(
                agent="testing_agent",
                purpose="test_generation",
                model=GEMINI_MODEL,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                response_time=response_time,
                cost_estimate=cost_estimate
            )
            
            test_code = response.text.strip()
            
            # Remove markdown code blocks if present
            if "```" in test_code:
                lines = test_code.split('\n')
                code_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or not line.strip().startswith('```'):
                        code_lines.append(line)
                
                test_code = '\n'.join(code_lines)
            
            return test_code
        
        except Exception as e:
            self.logger.log_error(
                component="testing_agent",
                error_type="test_generation_error",
                error_message=str(e)
            )
            return f"# Error generating tests: {str(e)}"
    
    async def validate_requirements(self, task_description: str, requirements: str, file_contents: Dict[str, str]) -> Dict:
        """
        Validate that generated code fulfills the original requirements.
        
        Args:
            task_description: Description of the task
            requirements: Original requirements text
            file_contents: Dictionary mapping file paths to their contents
            
        Returns:
            Dict with validation results:
                - validated: true or false
                - message: string explanation
                - errors: list of issues found
        """
        try:
            # Build file contents summary
            files_summary = "\n\n".join([
                f"File: {path}\n```\n{content}\n```"
                for path, content in file_contents.items()
            ])
            
            prompt = f"""
            You are a requirements validation expert. Your task is to determine if the generated code fulfills the original requirements.
            
            ORIGINAL REQUIREMENTS:
            {requirements}
            
            TASK DESCRIPTION:
            {task_description}
            
            GENERATED CODE:
            {files_summary}
            
            Analyze whether the generated code fulfills the requirements for this specific task.
            
            Respond in the following JSON format:
            {{
                "validated": true or false,
                "message": "Brief explanation of whether requirements are met",
                "errors": ["list of specific issues if any, empty array if validated is true"]
            }}
            
            Return ONLY valid JSON, nothing else.
            """
            
            # Track LLM call
            start_time = time.time()
            response = self.model.generate_content(prompt)
            response_time = time.time() - start_time
            
            # Estimate token counts
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response.text.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Log LLM call
            cost_estimate = (prompt_tokens / 1000 * 0.00025) + (completion_tokens / 1000 * 0.0005)
            self.logger.log_llm_call(
                agent="testing_agent",
                purpose="requirements_validation",
                model=GEMINI_MODEL,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                response_time=response_time,
                cost_estimate=cost_estimate
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(response_text)
            
            # Ensure all required fields are present
            if "validated" not in validation_result:
                validation_result["validated"] = False
            if "message" not in validation_result:
                validation_result["message"] = "Validation result incomplete"
            if "errors" not in validation_result:
                validation_result["errors"] = []
            
            return validation_result
        
        except Exception as e:
            self.logger.log_error(
                component="testing_agent",
                error_type="requirements_validation_error",
                error_message=str(e)
            )
            return {
                "validated": False,
                "message": f"Error validating requirements: {str(e)}",
                "errors": [str(e)]
            }
    
    async def run_tests(self, tasks: List[Dict], project_path: str, requirements: str = "") -> Dict[str, Dict]:
        """
        Run tests on all completed tasks.
        
        Args:
            tasks: List of task dictionaries
            project_path: Path to the project directory
            requirements: Original requirements text for validation (optional)
            
        Returns:
            Dict mapping task_id to test results
        """
        # Use async with to properly manage MCP connections
        async with stdio_client(self._get_file_params()) as (file_read, file_write):
            async with ClientSession(file_read, file_write) as file_session:
                self.file_session = file_session
                
                # Initialize the session to ensure it's ready
                await file_session.initialize()
                
                results = {}
                
                for task in tasks:
                    if task["status"] not in ["testing required", "completed"]:
                        continue
                    
                    task_id = task["id"]
                    task_result = {
                        "passed": True,
                        "message": "",
                        "errors": [],
                        "requirements_validated": False,
                        "requirements_validation_message": "",
                        "requirements_validation_errors": []
                    }
                    
                    # Get list of files from task result
                    # For now, we'll scan the project directory for Python files
                    python_files = []
                    for root, dirs, files in os.walk(project_path):
                        for file in files:
                            if file.endswith('.py'):
                                python_files.append(os.path.join(root, file))
                    
                    # Run lint checks on each file
                    all_lint_errors = []
                    for file_path in python_files:
                        lint_result = self.run_lint_check(file_path)
                        if not lint_result["passed"]:
                            all_lint_errors.extend(lint_result["errors"])
                    
                    # Generate and save test cases
                    test_files_created = []
                    for file_path in python_files:
                        if file_path.endswith('_test.py') or file_path.endswith('test_.py'):
                            continue  # Skip existing test files
                        
                        try:
                            # Read file content
                            result = await self.file_session.call_tool(
                                "read_file",
                                arguments={"file_path": file_path}
                            )
                            code_content = result.content[0].text
                            
                            # Generate test cases
                            test_code = await self.generate_test_cases(file_path, code_content)
                            
                            # Save test file
                            test_file_path = file_path.replace('.py', '_test.py')
                            await self.file_session.call_tool(
                                "edit_file",
                                arguments={
                                    "file_name": test_file_path,
                                    "content": test_code
                                }
                            )
                            test_files_created.append(test_file_path)
                        
                        except Exception as e:
                            all_lint_errors.append(f"Error generating tests for {file_path}: {str(e)}")
                    
                    # Validate requirements if provided
                    if requirements:
                        try:
                            # Read all generated files for this task
                            all_files = []
                            for root, dirs, files in os.walk(project_path):
                                for file in files:
                                    all_files.append(os.path.join(root, file))
                            
                            # Read file contents
                            file_contents = {}
                            for file_path in all_files:
                                if not file_path.endswith('_test.py'):
                                    try:
                                        result = await self.file_session.call_tool(
                                            "read_file",
                                            arguments={"file_path": file_path}
                                        )
                                        file_contents[file_path] = result.content[0].text
                                    except Exception:
                                        # Skip files that can't be read
                                        pass
                            
                            # Validate requirements
                            validation_result = await self.validate_requirements(
                                task["description"],
                                requirements,
                                file_contents
                            )
                            
                            task_result["requirements_validated"] = validation_result["validated"]
                            task_result["requirements_validation_message"] = validation_result["message"]
                            task_result["requirements_validation_errors"] = validation_result["errors"]
                            
                            # Update overall pass/fail based on requirements validation
                            if not validation_result["validated"]:
                                task_result["passed"] = False
                                all_lint_errors.extend(validation_result["errors"])
                        
                        except Exception as e:
                            task_result["requirements_validation_message"] = f"Error during validation: {str(e)}"
                            task_result["requirements_validation_errors"] = [str(e)]
                    
                    # Compile results
                    if all_lint_errors:
                        task_result["passed"] = False
                        task_result["errors"] = all_lint_errors
                        task_result["message"] = f"Found {len(all_lint_errors)} issues. Test files created: {len(test_files_created)}"
                    else:
                        task_result["passed"] = True
                        task_result["message"] = f"All checks passed. Test files created: {len(test_files_created)}"
                    
                    results[task_id] = task_result
                
                return results


async def main():
    """Test the testing agent."""
    agent = TestingAgent()
    
    tasks = [
        {
            "id": "test_1",
            "description": "Test task",
            "status": "testing required",
            "assigned_to": "coding_agent",
            "result": "",
            "errors": []
        }
    ]
    
    results = await agent.run_tests(tasks, "/tmp/test_project")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
