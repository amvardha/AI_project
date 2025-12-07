# Adithya Vardhan 32956089
import os
import json
import asyncio
import time
from typing import Tuple, List
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

class CodingAgent:
    """
    Coding agent that generates code for assigned tasks.
    """
    
    def __init__(self):
        """Initialize the coding agent."""
        self.orchestrator_session = None
        self.file_session = None
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.logger = get_logger()
    
    async def connect_to_mcps(self):
        """Connect to orchestrator and file MCP servers."""
        # Connect to orchestrator MCP
        orchestrator_params = StdioServerParameters(
            command="python",
            args=[os.getenv("ORCHESTRATOR_MCP_PATH", "../MCPs/orchestrator_mcp.py")],
            env=None
        )
        
        # Connect to file MCP
        file_params = StdioServerParameters(
            command="python",
            args=[os.getenv("FILE_MCP_PATH", "../MCPs/files_mcp.py")],
            env=None
        )
        
        # Create sessions
        self.orchestrator_read, self.orchestrator_write = await stdio_client(orchestrator_params)
        self.file_read, self.file_write = await stdio_client(file_params)
        
        self.orchestrator_session = ClientSession(self.orchestrator_read, self.orchestrator_write)
        self.file_session = ClientSession(self.file_read, self.file_write)
        
        await self.orchestrator_session.__aenter__()
        await self.file_session.__aenter__()
    
    async def disconnect_from_mcps(self):
        """Disconnect from MCP servers."""
        if self.orchestrator_session:
            await self.orchestrator_session.__aexit__(None, None, None)
        if self.file_session:
            await self.file_session.__aexit__(None, None, None)
    
    async def generate_code(self, task_description: str, project_path: str, errors: List[str] = []) -> Tuple[bool, str, List[str]]:
        """
        Generate code for the given task.
        
        Args:
            task_description: Description of the task
            project_path: Path to the project directory
            
        Returns:
            Tuple of (success, result_message, errors)
        """
        try:
            prompt = f"""
            You are an expert software developer. Generate code for the following task:
            
            Task: {task_description}
            Project Path: {project_path}
            
            Provide your response in the following JSON format:
            {{
                "files": [
                    {{
                        "path": "relative/path/to/file.ext",
                        "content": "file content here"
                    }}
                ],
                "description": "Brief description of what was implemented"
            }}
            
            Return ONLY valid JSON, nothing else.
            """

            if errors:
                prompt += f"\nErrors faced after previous attempts: {errors}"
            
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
                agent="coding_agent",
                purpose="code_generation",
                model="gemini-pro",
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
            
            code_plan = json.loads(response_text)
            
            # Create files using file MCP
            created_files = []
            for file_info in code_plan.get("files", []):
                file_path = os.path.join(project_path, file_info["path"])
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Create/write file using file MCP
                await self.file_session.call_tool(
                    "edit_file",
                    arguments={
                        "file_name": file_path,
                        "content": file_info["content"]
                    }
                )
                created_files.append(file_info["path"])
            
            result_message = f"{code_plan.get('description', 'Code generated')}. Files created: {', '.join(created_files)}"
            return True, result_message, []
        
        except Exception as e:
            error_message = f"Error generating code: {str(e)}"
            self.logger.log_error(
                component="coding_agent",
                error_type="code_generation_error",
                error_message=error_message
            )
            return False, error_message, [str(e)]
    
    async def execute_task(self, task: dict, project_path: str) -> Tuple[bool, str, List[str]]:
        """
        Execute a coding task.
        
        Args:
            task: Task dictionary with description
            project_path: Path to the project directory
            
        Returns:
            Tuple of (success, result, errors)
        """
        return await self.generate_code(task["description"], project_path, task["errors"])


async def main():
    """Test the coding agent."""
    agent = CodingAgent()
    await agent.connect_to_mcps()
    
    try:
        task = {
            "id": "test_1",
            "description": "Create a simple Python hello world script",
            "status": "pending",
            "assigned_to": "coding_agent"
        }
        
        success, result, errors = await agent.execute_task(task, "/tmp/test_coding")
        print(f"Success: {success}")
        print(f"Result: {result}")
        print(f"Errors: {errors}")
    
    finally:
        await agent.disconnect_from_mcps()


if __name__ == "__main__":
    asyncio.run(main())
