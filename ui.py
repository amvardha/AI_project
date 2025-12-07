import gradio as gr
import os
import zipfile
import tempfile
import time
from pathlib import Path

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
def generate_app(description, requirements):
    """
    Placeholder function for generating application code and test cases.
    This function will be updated with actual generation logic later.
    
    Args:
        description: The software description text
        requirements: Additional requirements from the user
    
    Yields:
        tuple: (status_message, folder_path, zip_file_path) - status, project folder path, and zip file path
    """
    # Create a temporary directory for the generated project
    project_name = "generated_project"
    output_dir = os.path.join(tempfile.gettempdir(), project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a placeholder file to ensure the directory has content
    placeholder_file = os.path.join(output_dir, "README.md")
    with open(placeholder_file, 'w') as f:
        f.write("# Generated Project\n\nThis is a placeholder project.")
    
    # Status updates during generation
    status_messages = [
        "Planning",
        "Analyzing Requirements",
        "Coding",
        "Testing",
        "Linting",
        "Completed"
    ]
    
    # Simulate progress updates (replace with actual generation logic)
    for i, status in enumerate(status_messages):
        # Small delay to show status progression
        time.sleep(0.5)
        # Generate HTML for the colored status box
        status_html = get_status_html(status)
        # Yield status update (zip will be None until completion)
        yield status_html, output_dir, None
    
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
    yield final_status_html, output_dir, zip_path


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
                value=default_description,
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
    
    # Connect the button click to the generate_app function
    # The function takes description and requirements as inputs
    # and outputs to status_output, folder_output, and zip_output
    generate_btn.click(
        fn=generate_app,
        inputs=[description_input, requirements_input],
        outputs=[status_output, folder_output, zip_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()

