# Adithya Vardhan 32956089
import os
import json
from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("OrchestratorMCP")

# In-memory task storage
tasks_db: Dict[str, Dict] = {}

@mcp.tool()
def read_requirements(file_path: str) -> str:
    """
    Parse and return requirements from uploaded file.
    
    Args:
        file_path: Path to the requirements file
        
    Returns:
        str: Contents of the requirements file
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    return content

@mcp.tool()
def set_tasks(tasks: List[Dict]) -> str:
    """
    Initialize task list with task objects.
    
    Args:
        tasks: List of task dictionaries containing:
            - id: Unique task identifier
            - description: Task description
            - status: Task status ("pending", "coding", "testing required", "completed", "error")
            - assigned_to: Agent assigned to the task
            - result: Task result (empty initially)
            - errors: List of error messages
            
    Returns:
        str: Confirmation message with number of tasks created
    """
    global tasks_db
    
    for task in tasks:
        # Validate required fields
        required_fields = ["id", "description", "status", "assigned_to"]
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Missing required field: {field}")
        
        # Set defaults for optional fields
        if "result" not in task:
            task["result"] = ""
        if "errors" not in task:
            task["errors"] = []
        
        # Validate status
        valid_statuses = ["pending", "coding", "testing required", "completed", "error"]
        if task["status"] not in valid_statuses:
            raise ValueError(f"Invalid status: {task['status']}. Must be one of {valid_statuses}")
        
        tasks_db[task["id"]] = task
    
    return f"Successfully created {len(tasks)} tasks"

@mcp.tool()
def get_tasks() -> str:
    """
    Retrieve all tasks with their current status.
    
    Returns:
        str: JSON string of all tasks
    """
    return json.dumps(list(tasks_db.values()), indent=2)

@mcp.tool()
def update_task_status(task_id: str, status: str, result: str = "", errors: Optional[List[str]] = None) -> str:
    """
    Update task status, result, and errors.
    
    Args:
        task_id: ID of the task to update
        status: New status ("pending", "coding", "testing required", "completed", "error")
        result: Task result or output
        errors: List of error messages (optional)
        
    Returns:
        str: Confirmation message
    """
    if task_id not in tasks_db:
        raise ValueError(f"Task not found: {task_id}")
    
    # Validate status
    valid_statuses = ["pending", "coding", "testing required", "completed", "error"]
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
    
    tasks_db[task_id]["status"] = status
    tasks_db[task_id]["result"] = result
    
    if errors is not None:
        tasks_db[task_id]["errors"] = errors
    
    return f"Task {task_id} updated to status: {status}"

@mcp.tool()
def delete_task(task_id: str) -> str:
    """
    Remove a task from the list.
    
    Args:
        task_id: ID of the task to delete
        
    Returns:
        str: Confirmation message
    """
    if task_id not in tasks_db:
        raise ValueError(f"Task not found: {task_id}")
    
    del tasks_db[task_id]
    return f"Task {task_id} deleted successfully"

@mcp.tool()
def get_task_by_id(task_id: str) -> str:
    """
    Get a specific task by ID.
    
    Args:
        task_id: ID of the task to retrieve
        
    Returns:
        str: JSON string of the task
    """
    if task_id not in tasks_db:
        raise ValueError(f"Task not found: {task_id}")
    
    return json.dumps(tasks_db[task_id], indent=2)

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
