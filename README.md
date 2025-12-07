#Adithya Vardhan 32956089 
# Multi-Agent AI Code Generator

A sophisticated multi-agent system that automatically generates, tests, and validates code based on requirements documents.

## Architecture

This system consists of:

### MCP Servers
- **Orchestrator MCP**: Manages tasks, tracks status, and coordinates agent communication
- **File MCP**: Handles file operations, Next.js app creation, and npm package installation

### Agents
- **Main Agent**: Orchestrates the entire workflow, parses requirements, breaks down tasks
- **Coding Agent**: Generates code using LLM based on task descriptions
- **Testing Agent**: Runs lint checks and generates test cases

### UI
- **Gradio Interface**: User-friendly web interface for uploading requirements and monitoring progress

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Update MCP Paths (if needed)

If your MCP server paths are different, update them in `.env`:

```
ORCHESTRATOR_MCP_PATH=/path/to/orchestrator_mcp.py
FILE_MCP_PATH=/path/to/files_mcp.py
```

## Usage

### Running the UI

```bash
python ui.py
```

This will launch a Gradio interface at `http://localhost:7860`

### Using the System

1. **Enter Requirements**: Either type your software description and requirements in the text boxes, OR upload a requirements document
2. **Click Generate**: Press the "Generate Application" button
3. **Monitor Progress**: Watch the status updates as the system:
   - Plans the implementation
   - Breaks down tasks
   - Generates code
   - Runs tests
   - Creates test cases
4. **Download Results**: Once complete, download the generated project as a ZIP file

## Task Statuses

The system uses the following task statuses:

- **pending**: Task is waiting to be executed
- **coding**: Task is currently being coded
- **testing required**: Code is complete, waiting for testing
- **completed**: Task passed all tests
- **error**: Task encountered an error

## Architecture Details

### Task Flow

1. User uploads requirements → Main Agent
2. Main Agent parses requirements → Orchestrator MCP
3. Main Agent breaks down into tasks → LLM (Gemini)
4. Tasks stored in Orchestrator MCP
5. For each task:
   - Main Agent assigns to Coding Agent
   - Coding Agent generates code → File MCP
   - Status updated to "testing required"
6. Testing Agent:
   - Runs lint checks on all files
   - Generates test cases using LLM
   - Updates task status to "completed" or "error"

### Error Handling

- Tasks that fail are retried up to 3 times
- Errors are tracked in the task object's `errors` field
- Failed tasks are marked with "error" status

## File Structure

```
AI_project/
├── agents/
│   ├── __init__.py
│   ├── main_agent.py       # Orchestrator agent
│   ├── coding_agent.py     # Code generation agent
│   └── testing_agent.py    # Testing and linting agent
├── MCPs/
│   ├── orchestrator_mcp.py # Task management MCP
│   └── files_mcp.py        # File operations MCP
├── ui.py                   # Gradio interface
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Testing Individual Components

### Test Orchestrator MCP

```bash
cd MCPs
python orchestrator_mcp.py
```

### Test File MCP

```bash
cd MCPs
python files_mcp.py
```

### Test Main Agent

```bash
cd agents
python main_agent.py
```

## Troubleshooting

### "Module not found" errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### "API key not found" errors
Ensure your `.env` file has the correct `GEMINI_API_KEY`

### MCP connection errors
Check that the MCP server paths in `.env` are correct

## Future Enhancements

- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Persistent task storage (database)
- Real-time collaboration features
- Advanced test execution and reporting
- Support for more programming languages

## License

MIT License - See LICENSE file for details
