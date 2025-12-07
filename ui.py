# Adithya Vardhan 32956089
import os
import time
import asyncio
import zipfile
import tempfile
import gradio as gr
from pathlib import Path

from agents.main_agent import MainAgent
from utils.logger import reset_logger

# Color mapping for different statuses
STATUS_COLORS = {
    "Planning": "#4A90E2",  # Blue
    "Analyzing Requirements": "#50C9CE",  # Cyan
    "Coding": "#F5A623",  # Orange/Yellow
    "Testing": "#9013FE",  # Purple
    "Linting": "#FF6B6B",  # Red/Orange
    "Completed": "#7ED321",  # Green
    "Ready to generate...": "#CCCCCC"  # Gray
}

def get_status_html(status):
    """
    Generate HTML for a colored status box.
    
    Args:
        status: The current status message
    
    Returns:
        str: HTML string with a colored status box
    """
    color = STATUS_COLORS.get(status, "#CCCCCC")
    html = f"""
    <div style="
        width: 400px;
        height: 20px;
        background-color: {color};
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px auto;
    ">
        {status}
    </div>
    """
    return html

# Placeholder function that will be called when the "Generate Application" button is pressed
def generate_app(description, requirements, uploaded_file):
    """
    Generate application code and test cases using the multi-agent system.
    
    Args:
        description: The software description text
        requirements: Additional requirements from the user
        uploaded_file: Uploaded requirements file (optional)
    
    Yields:
        tuple: (status_message, folder_path, zip_file_path, log_summary) - status, project folder path, zip file, and log summary
    """
    
    # Reset logger for new session
    reset_logger()
    
    # Create a temporary directory for the generated project
    project_name = "generated_project"
    output_dir = os.path.join(tempfile.gettempdir(), project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create requirements file from inputs
    requirements_file = os.path.join(tempfile.gettempdir(), "requirements.txt")
    
    if uploaded_file is not None:
        # Use uploaded file
        requirements_file = uploaded_file.name
    else:
        # Create requirements file from text inputs
        with open(requirements_file, 'w') as f:
            f.write(f"# Software Description\n{description}\n\n")
            if requirements:
                f.write(f"# Additional Requirements\n{requirements}\n")
    
    # Status callback to update UI
    current_status = {"value": "Planning"}
    
    def status_callback(status):
        current_status["value"] = status
    
    # Create and run main agent
    agent = MainAgent(status_callback=status_callback)
    
    # Run agent in async context
    async def run_agent():
        return await agent.run(requirements_file, output_dir)
    
    # Execute agent and yield status updates
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start agent in background
    task = loop.create_task(run_agent())
    
    # Poll for status updates
    while not task.done():
        status_html = get_status_html(current_status["value"])
        yield status_html, output_dir, None, ""
        time.sleep(1)
    
    # Get final result
    try:
        result = task.result()
        
        if result.get("success"):
            # Create a zip file of the project
            zip_path = os.path.join(tempfile.gettempdir(), f"{project_name}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
            
            # Final status update with zip file
            final_status_html = get_status_html("Completed")
            
            # Generate log summary
            log_summary = ""
            if "log_summary" in result:
                from utils.logger import get_logger
                logger = get_logger()
                log_summary = logger.get_formatted_summary()
            
            yield final_status_html, output_dir, zip_path, log_summary
        else:
            # Error occurred
            error_status_html = get_status_html("Error")
            
            # Generate log summary even on error
            log_summary = ""
            if "log_summary" in result:
                from utils.logger import get_logger
                logger = get_logger()
                log_summary = logger.get_formatted_summary()
            
            yield error_status_html, output_dir, None, log_summary
    
    except Exception as e:
        error_status_html = get_status_html("Error")
        yield error_status_html, f"Error: {str(e)}", None, ""
    
    finally:
        loop.close()


# Pre-filled software description for QuickDelivery
default_description = """QuickDelivery is a Restaurants&Delivery software application that focuses on providing fast and efficient delivery services for users. It allows users to easily browse through a curated selection of popular dishes from local restaurants and place orders for delivery. Users can track the progress of their orders in real-time and benefit from seamless payment options."""

# Create the Gradio Blocks interface
with gr.Blocks(title="AI Coder - Software Generator") as demo:
    # Title and description at the top
    gr.Markdown("# AI Coder — Software Generator")
    gr.Markdown("Enter your software description and requirements to generate application code and test cases.")
    
    # Two-column layout: inputs on left, outputs on right
    with gr.Row():
        # Left column: Inputs
        with gr.Column(scale=1):
            gr.Markdown("## Project Description")
            # Multiline textbox for software description (pre-filled)
            description_input = gr.Textbox(
                label="Software Description",
                value="",
                lines=8,
                max_lines=15,
                placeholder="Enter your software description here..."
            )
            
            gr.Markdown("## Additional Requirements")
            # Multiline textbox for additional requirements
            requirements_input = gr.Textbox(
                label="Additional Requirements",
                lines=5,
                max_lines=10,
                placeholder="Enter any additional requirements or specifications here..."
            )
            
            gr.Markdown("## Upload Requirements File (Optional)")
            # File upload for requirements document
            file_upload = gr.File(
                label="Upload Requirements Document",
                file_types=[".txt", ".md", ".pdf", ".doc", ".docx"],
                file_count="single"
            )
            
            # Generate button
            generate_btn = gr.Button("Generate Application", variant="primary", size="lg")
        
        # Right column: Outputs
        with gr.Column(scale=1):
            gr.Markdown("## Generation Status")
            # Status display showing current generation phase as a colored box
            status_output = gr.HTML(
                value=get_status_html("Ready to generate..."),
                label="Status"
            )
            
            gr.Markdown("## Project Location")
            # Display the folder path where the project is generated
            folder_output = gr.Textbox(
                label="Project Folder",
                value="",
                lines=2,
                interactive=False,
                placeholder="Project folder path will appear here..."
            )
            
            gr.Markdown("## Download Project")
            # File download component for the zip file
            zip_output = gr.File(
                label="Download Project as ZIP",
                file_count="single",
                visible=True
            )
            
            gr.Markdown("## Session Summary")
            # Log summary display
            log_summary_output = gr.Textbox(
                label="Token Usage & API Call Summary",
                value="",
                lines=15,
                max_lines=30,
                interactive=False,
                placeholder="Summary will appear here after generation..."
            )
    
    # Connect the button click to the generate_app function
    # The function takes description, requirements, and file upload as inputs
    # and outputs to status_output, folder_output, zip_output, and log_summary_output
    generate_btn.click(
        fn=generate_app,
        inputs=[description_input, requirements_input, file_upload],
        outputs=[status_output, folder_output, zip_output, log_summary_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()

