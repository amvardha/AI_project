# Daniel Rivas 
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentCommunicationProtocol") # ACP

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

if __name__ == "__main__":
    mcp.run(transport="stdio")