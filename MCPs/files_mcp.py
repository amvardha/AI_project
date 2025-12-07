# Daniel Rivas 55919944
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FileProtocol")

# MCP Server will have tooling for create file, edit file, overwrite file
@mcp.tool()
def create_file(file_name: str) -> str:
    """Create a file"""
    
    # Create a file
    with open(file_name, "w") as f:
        f.write("")
    
    # Return the file name
    return file_name

@mcp.tool()
def edit_file(file_name: str, content: str) -> str:
    """Edit a file"""
    
    # Edit a file
    with open(file_name, "w") as f:
        f.write(content)
    
    # Return the file name
    return file_name

@mcp.tool()
def overwrite_file(file_name: str, content: str) -> str:
    """Overwrite a file"""
    
    # Overwrite a file
    with open(file_name, "w") as f:
        f.write(content)
    
    # Return the file name
    return file_name

@mcp.tool()
def delete_file(file_name: str) -> str:
    """Delete a file"""
    
    # Delete a file
    os.remove(file_name)
    
    # Return the file name
    return file_name

# Linting tool that edits specific lines of a file. Params: file_name, start, end, content
@mcp.tool()
def lint_file(file_name: str, start: int, end: int, content: str) -> str:
    """Lint a file"""
    ## -- Param Edge Cases -- ##
    if start < 0 or end < 0:
        raise ValueError("Start and end must be greater than 0")
    if start > end:
        raise ValueError("Start must be less than end")
    
    ## -- File Edge Cases -- ##
    # Check if file exists
    if not os.path.exists(file_name):
        raise ValueError("File does not exist")
    
    # Check if file is a file
    if not os.path.isfile(file_name):
        raise ValueError("File is not a file")
    
    # Check if file is empty
    if os.path.getsize(file_name) == 0:
        raise ValueError("File is empty")
    
    ## -- File Operations -- ##
    # Lint a file
    with open(file_name, "r") as f:
        lines = f.readlines()

    if start >= len(lines) or end > len(lines):
        raise ValueError(f"Start and end must be within file line count ({len(lines)})")
    
    # Edit specific lines
    for i in range(start, end):
        lines[i] = content
    
    # Write the file
    with open(file_name, "w") as f:
        f.writelines(lines)
    
    # Return the file name
    return file_name

if __name__ == "__main__":
    # Test the linting functionality
    test_file = "lint_test.txt"
    create_file(test_file)
    # File must have content to be linted (lint_file checks for empty file and valid line indices)
    overwrite_file(test_file, "Line 1\nLine 2\n")
    lint_file(test_file, 0, 1, "Modified Line 1\n")

    # Run the server
    mcp.run(transport="stdio")
