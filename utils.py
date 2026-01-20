import pdfplumber
import base64
from groq import Groq
import os
from dotenv import load_dotenv
import markdown
from xhtml2pdf import pisa

load_dotenv()

# Initialize Groq for Vision tasks
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_image(file_path: str) -> str:
    # Encode image
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Send to Llama Vision
    completion = groq_client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this resume image strictly. No summary."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                ]
            }
        ],
        max_tokens=1024
    )
    return completion.choices[0].message.content

def save_resume_as_pdf(markdown_content: str, output_filename: str):
    """
    Converts Markdown resume text into a formatted PDF file.
    """
    # Simple & Professional CSS for Resume
    css = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 10pt; line-height: 1.4; color: #333; }
        h1 { color: #000; border-bottom: 2px solid #000; padding-bottom: 5px; margin-top: 20px; font-size: 18pt; text-transform: uppercase; }
        h2 { color: #2c3e50; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 15px; font-size: 14pt; }
        h3 { color: #444; font-size: 12pt; margin-top: 10px; margin-bottom: 2px; font-weight: bold; }
        ul { margin-top: 5px; padding-left: 20px; }
        li { margin-bottom: 3px; text-align: justify; }
        p { margin-bottom: 5px; }
        strong { color: #000; }
    </style>
    """

    # Convert Markdown to HTML
    html_text = markdown.markdown(markdown_content)
    full_html = f"<html><head>{css}</head><body>{html_text}</body></html>"

    # Create folder if not exists
    os.makedirs("generated_resumes", exist_ok=True)
    file_path = f"generated_resumes/{output_filename}"
    
    # Write PDF
    with open(file_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
    
    if pisa_status.err:
        print("PDF generation error")
        return None
    return file_path