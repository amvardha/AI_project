# Daniel Rivas 55919944
import os
import sys
import subprocess
from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FileProtocol")

# MCP Server will have tooling for create file, edit file, overwrite file
@mcp.tool()
def create_file(file_name: str) -> str:
    """Create a file"""
    
    print(f"[FilesMCP] create_file called: {file_name}", flush=True, file=sys.stderr)
    
    # Create a file
    with open(file_name, "w") as f:
        f.write("")
    
    print(f"[FilesMCP] File created successfully: {file_name}", flush=True, file=sys.stderr)
    # Return the file name
    return file_name

@mcp.tool()
def read_file(file_path: str) -> str:
    """
    Read and return file contents.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        str: Contents of the file
    """
    print(f"[FilesMCP] read_file called: {file_path}", flush=True, file=sys.stderr)
    
    if not os.path.exists(file_path):
        print(f"[FilesMCP] ERROR: File not found: {file_path}", flush=True, file=sys.stderr)
        raise ValueError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"[FilesMCP] Read {len(content)} characters from {file_path}", flush=True, file=sys.stderr)
    return content

@mcp.tool()
def edit_file(file_name: str, content: str) -> str:
    """Edit a file"""
    
    print(f"[FilesMCP] edit_file called: {file_name} ({len(content)} chars)", flush=True, file=sys.stderr)
    
    # Edit a file
    with open(file_name, "w") as f:
        f.write(content)
    
    print(f"[FilesMCP] File edited successfully: {file_name}", flush=True, file=sys.stderr)
    # Return the file name
    return file_name

@mcp.tool()
def overwrite_file(file_name: str, content: str) -> str:
    """Overwrite a file"""
    
    print(f"[FilesMCP] overwrite_file called: {file_name} ({len(content)} chars)", flush=True, file=sys.stderr)
    
    # Overwrite a file
    with open(file_name, "w") as f:
        f.write(content)
    
    print(f"[FilesMCP] File overwritten successfully: {file_name}", flush=True, file=sys.stderr)
    # Return the file name
    return file_name

@mcp.tool()
def delete_file(file_name: str) -> str:
    """Delete a file"""
    
    print(f"[FilesMCP] delete_file called: {file_name}", flush=True, file=sys.stderr)
    
    # Delete a file
    os.remove(file_name)
    
    print(f"[FilesMCP] File deleted successfully: {file_name}", flush=True, file=sys.stderr)
    # Return the file name
    return file_name

# Linting tool that edits specific lines of a file. Params: file_name, start, end, content
@mcp.tool()
def replace_lines(file_name: str, start: int, end: int, content: str) -> str:
    """Replace lines in a file"""
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

@mcp.tool()
def list_directory(folder_path: str) -> str:
    """
    List the folder structure and subdirectories of a given folder.
    
    Args:
        folder_path: Path to the folder to list
        
    Returns:
        str: JSON string containing folder structure with files and subdirectories
    """
    import json
    
    print(f"[FilesMCP] list_directory called: {folder_path}", flush=True, file=sys.stderr)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"[FilesMCP] ERROR: Folder not found: {folder_path}", flush=True, file=sys.stderr)
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Check if it's a directory
    if not os.path.isdir(folder_path):
        print(f"[FilesMCP] ERROR: Path is not a directory: {folder_path}", flush=True, file=sys.stderr)
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Build folder structure
    structure = {
        "path": folder_path,
        "name": os.path.basename(folder_path) or folder_path,
        "type": "directory",
        "children": []
    }
    
    try:
        # List all items in the directory
        items = os.listdir(folder_path)
        items.sort()  # Sort alphabetically
        
        for item in items:
            item_path = os.path.join(folder_path, item)
            
            if os.path.isdir(item_path):
                # It's a subdirectory
                structure["children"].append({
                    "name": item,
                    "type": "directory",
                    "path": item_path
                })
            else:
                # It's a file
                file_size = os.path.getsize(item_path)
                structure["children"].append({
                    "name": item,
                    "type": "file",
                    "path": item_path,
                    "size": file_size
                })
        
        print(f"[FilesMCP] Listed {len(structure['children'])} items in {folder_path}", flush=True, file=sys.stderr)
        return json.dumps(structure, indent=2)
    
    except PermissionError:
        print(f"[FilesMCP] ERROR: Permission denied: {folder_path}", flush=True, file=sys.stderr)
        raise ValueError(f"Permission denied to access: {folder_path}")

if __name__ == "__main__":
    # Run the server
    print("Starting Files MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")
