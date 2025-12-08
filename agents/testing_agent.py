# Adithya Vardhan 32956089
# Daniel Rivas 55919944
import os
import json
import asyncio
import subprocess
import time
from typing import Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from utils.logger import get_logger

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

class TestingAgent:
    """
    Testing agent that runs lint checks and generates test cases.
    Uses LangChain ReAct framework with MCP tools.
    """
    
    def __init__(self):
        """Initialize the testing agent."""
        self.file_session = None
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        self.logger = get_logger()
        self.agent_executor = None
    
    def _get_file_params(self):
        """Get file MCP server parameters."""
        return StdioServerParameters(
            command="python",
            args=[os.getenv("FILE_MCP_PATH", "run_mcp_servers.py"), "file"],
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
            response = self.llm.invoke(prompt)
            response_time = time.time() - start_time
            
            # Get response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Estimate token counts
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response_text.split()) * 1.3
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
            
            test_code = response_text.strip()
            
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
    
    async def validate_requirements_optimized(self, task_description: str, requirements: str, 
                                             file_list: List[str], file_contents: Dict[str, str]) -> Dict:
        """
        Optimized requirements validation using directory structure and selective file reading.
        
        Args:
            task_description: Description of the task
            requirements: Original requirements text
            file_list: List of all files in the project directory
            file_contents: Dictionary mapping selected file names to their contents
            
        Returns:
            Dict with validation results
        """
        try:
            # Build a concise file contents summary (only key files)
            files_summary = "\n\n".join([
                f"File: {name}\n```\n{content[:1000]}{'...(truncated)' if len(content) > 1000 else ''}\n```"
                for name, content in file_contents.items()
            ])
            
            prompt = f"""
            You are a requirements validation expert. Determine if the generated code fulfills the requirements.
            
            ORIGINAL REQUIREMENTS:
            {requirements}
            
            TASK DESCRIPTION:
            {task_description}
            
            PROJECT STRUCTURE (all files):
            {', '.join(file_list)}
            
            KEY FILE CONTENTS (sample):
            {files_summary}
            
            Based on the project structure and key file contents, analyze whether the requirements are met.
            
            Respond in JSON format:
            {{
                "validated": true or false,
                "message": "Brief explanation",
                "errors": ["list of issues if any, empty if validated"]
            }}
            
            Return ONLY valid JSON.
            """
            
            # Track LLM call
            start_time = time.time()
            response = self.llm.invoke(prompt)
            response_time = time.time() - start_time
            
            # Get response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Estimate token counts
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response_text.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Log LLM call
            cost_estimate = (prompt_tokens / 1000 * 0.00025) + (completion_tokens / 1000 * 0.0005)
            self.logger.log_llm_call(
                agent="testing_agent",
                purpose="requirements_validation_optimized",
                model=GEMINI_MODEL,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                response_time=response_time,
                cost_estimate=cost_estimate
            )
            
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
            response = self.llm.invoke(prompt)
            response_time = time.time() - start_time
            
            # Get response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Estimate token counts
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response_text.split()) * 1.3
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
                
                try:
                    # 1. Get project structure
                    dir_result = await self.file_session.call_tool(
                        "list_directory",
                        arguments={"folder_path": project_path}
                    )
                    dir_structure = json.loads(dir_result.content[0].text)
                    
                    # 2. Identify backend and frontend files
                    backend_files = []
                    frontend_files = []
                    
                    def categorize_files(items):
                        for item in items:
                            if item["type"] == "directory":
                                categorize_files(item.get("children", []))
                            elif item["type"] == "file":
                                path = item["path"]
                                name = item["name"]
                                # Skip existing tests and non-code files
                                if name.endswith('_test.py') or name.endswith('.pyc') or name.startswith('.'):
                                    continue
                                    
                                if name.endswith('.py'):
                                    backend_files.append(item)
                                elif name.endswith('.html') or name.endswith('.js') or name.endswith('.css'):
                                    frontend_files.append(item)
                    
                    categorize_files(dir_structure.get("children", []))
                    
                    # 3. Read key files for context (limit to save tokens)
                    backend_context = ""
                    for file_info in backend_files[:5]: # Top 5 Python files
                        try:
                            res = await self.file_session.call_tool("read_file", arguments={"file_path": file_info["path"]})
                            content = res.content[0].text[:2000] # Truncate
                            backend_context += f"\nFile: {file_info['path']}\n```python\n{content}\n```\n"
                        except: pass
                        
                    frontend_context = ""
                    for file_info in frontend_files[:5]: # Top 5 Frontend files
                        try:
                            res = await self.file_session.call_tool("read_file", arguments={"file_path": file_info["path"]})
                            content = res.content[0].text[:2000] # Truncate
                            frontend_context += f"\nFile: {file_info['path']}\n```\n{content}\n```\n"
                        except: pass
                    
                    # 4. Generate ONE Backend Test File
                    if backend_files:
                        self.logger.info("Testing_agent: Generating single backend test suite...")
                        backend_test_code = await self.generate_test_suite("backend", backend_context)
                        
                        backend_test_path = os.path.join(project_path, "backend_tests.py")
                        await self.file_session.call_tool(
                            "edit_file",
                            arguments={"file_name": backend_test_path, "content": backend_test_code}
                        )
                        results["backend_tests"] = {"passed": True, "message": "Backend tests generated", "file": backend_test_path}

                    # 5. Generate ONE Frontend Test File
                    if frontend_files:
                        self.logger.info("Testing_agent: Generating single frontend test suite...")
                        frontend_test_code = await self.generate_test_suite("frontend", frontend_context)
                        
                        frontend_test_path = os.path.join(project_path, "frontend_tests.py")
                        await self.file_session.call_tool(
                            "edit_file",
                            arguments={"file_name": frontend_test_path, "content": frontend_test_code}
                        )
                        results["frontend_tests"] = {"passed": True, "message": "Frontend tests generated", "file": frontend_test_path}

                    # 6. Validate Requirements (Single Call)
                    if requirements:
                        self.logger.info("Testing_agent: Validating requirements...")
                        file_list = [f["name"] for f in backend_files + frontend_files]
                        # Re-use context we already read
                        validation_result = await self.validate_requirements_optimized(
                            "Check if the project meets requirements based on generated files.",
                            requirements,
                            file_list,
                            {"backend_context": backend_context, "frontend_context": frontend_context}
                        )
                        results["requirements_validation"] = validation_result
                        
                except Exception as e:
                    self.logger.log_error("testing_agent", "run_tests_error", str(e))
                    results["error"] = str(e)
                
                return results

    async def generate_test_suite(self, suite_type: str, context: str) -> str:
        """Generate a comprehensive test suite for backend or frontend."""
        prompt = f"""
        Generate a comprehensive {suite_type} test suite (Python pytest) for the following code context.
        Just 5 test cases should be generated.
        Create a SINGLE generic test file that covers the main functionality found in the context.
        Mock external dependencies where possible.
        
        CONTEXT:
        {context}
        
        Return ONLY the Python code for the test file.
        """
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content



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
