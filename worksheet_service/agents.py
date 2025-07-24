from pydantic_ai import Agent
from worksheet_service.schemas import WorksheetOutput
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
import logging
import pytesseract
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from deep_translator import GoogleTranslator
from langchain.agents import initialize_agent, Tool
from langchain_google_vertexai import VertexAI
import mimetypes
import PyPDF2
import docx
import pdfplumber

model = GeminiModel(
    model_name="gemini-2.5-flash",
    provider=GoogleVertexProvider(region="us-central1")
)
worksheet_agent = Agent(
    model=model,
    output_type=WorksheetOutput,
    deps_type=None,
    system_prompt=(
        "You are a helpful AI assistant for teachers in rural Indian schools. "
        "You are given an image of a textbook page. Generate simple, age-appropriate worksheets "
        "for the requested grade levels and return them as a JSON dictionary."
    )
)

def ocr_tool(image_url: str) -> str:
    try:
        if image_url.startswith('http://') or image_url.startswith('https://'):
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_url)
        text = pytesseract.image_to_string(img)
        return text if text.strip() else "[No text extracted from image]"
    except (UnidentifiedImageError, Exception) as e:
        return f"[OCR extraction failed: {str(e)}]"

def extract_text_from_pdf(pdf_url: str) -> str:
    try:
        if pdf_url.startswith('http://') or pdf_url.startswith('https://'):
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = BytesIO(response.content)
        else:
            pdf_bytes = open(pdf_url, 'rb')
        with pdf_bytes:
            reader = PyPDF2.PdfReader(pdf_bytes)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
            if text.strip():
                return text
            # Fallback to OCR if no text extracted
            logging.info("[Worksheet Agent] No text found in PDF, using OCR fallback.")
            pdf_bytes.seek(0)
            ocr_text = ""
            with pdfplumber.open(pdf_bytes) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        img = page.to_image(resolution=300).original
                        ocr_text += pytesseract.image_to_string(img) + "\n"
                    except Exception as e:
                        logging.warning(f"[Worksheet Agent] OCR failed for PDF page {i}: {e}")
            return ocr_text if ocr_text.strip() else "[No text extracted from PDF, even with OCR]"
    except Exception as e:
        return f"[PDF extraction failed: {str(e)}]"

def extract_text_from_docx(docx_url: str) -> str:
    try:
        if docx_url.startswith('http://') or docx_url.startswith('https://'):
            response = requests.get(docx_url)
            response.raise_for_status()
            docx_bytes = BytesIO(response.content)
        else:
            docx_bytes = open(docx_url, 'rb')
        with docx_bytes:
            doc = docx.Document(docx_bytes)
            text = " ".join([para.text for para in doc.paragraphs])
            if text.strip():
                return text
            # Fallback to OCR for images in DOCX
            logging.info("[Worksheet Agent] No text found in DOCX, using OCR fallback.")
            ocr_text = ""
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        img_bytes = rel.target_part.blob
                        img = Image.open(BytesIO(img_bytes))
                        ocr_text += pytesseract.image_to_string(img) + "\n"
                    except Exception as e:
                        logging.warning(f"[Worksheet Agent] OCR failed for DOCX image: {e}")
            return ocr_text if ocr_text.strip() else "[No text extracted from DOCX, even with OCR]"
    except Exception as e:
        return f"[DOCX extraction failed: {str(e)}]"

def translate_tool(text: str, target_language: str) -> str:
    lang_map = {
        "english": "en",
        "hindi": "hi",
        "kannada": "kn",
        "german": "de"
    }
    to_code = lang_map.get(target_language.lower(), "en")
    if to_code == "en":
        return text
    try:
        translated = GoogleTranslator(source="en", target=to_code).translate(text)
        return translated
    except Exception:
        return text

def get_lang_code(language: str) -> str:
    lang_map = {
        "english": "en",
        "hindi": "hi",
        "kannada": "kn",
        "german": "de"
    }
    return lang_map.get(language.lower(), "en")

ocr_tool_lc = Tool(
    name="ocr",
    func=ocr_tool,
    description="Extracts text from image files (JPG, PNG, etc.) using OCR."
)
pdf_tool_lc = Tool(
    name="extract_pdf",
    func=extract_text_from_pdf,
    description="Extracts text from PDF files."
)
docx_tool_lc = Tool(
    name="extract_docx",
    func=extract_text_from_docx,
    description="Extracts text from Word (DOCX) files."
)
translate_tool_lc = Tool(
    name="translate",
    func=translate_tool,
    description="Translates text to the target language."
)

llm_lc = VertexAI(model_name="gemini-2.5-flash")

async def generate_worksheets(file_url: str, grade_input: str, language: str = "English") -> dict:
    tools = [ocr_tool_lc, pdf_tool_lc, docx_tool_lc, translate_tool_lc]
    agent = initialize_agent(
        tools=tools,
        llm=llm_lc,
        agent="zero-shot-react-description",
        verbose=False
    )
    extraction_task = f"Extract all text from this file: {file_url}"
    extracted_text = agent.run(extraction_task)
    logging.info(f"[Worksheet Agent] Extracted text: {extracted_text}")
    if extracted_text.startswith("[") and "failed" in extracted_text:
        raise ValueError(f"Text extraction failed: {extracted_text}")
    lang_code = get_lang_code(language)
    if lang_code != "en":
        extracted_text = agent.run(f"Translate the following text to {language}: {extracted_text}")
    prompt = f"Generate simple, age-appropriate worksheets for the following grades: {grade_input}. The content should be in {language}. Use this extracted textbook text: {extracted_text}. Return a JSON object with keys grade_1 to grade_6, each containing the worksheet for that grade (if requested)."
    result = await worksheet_agent.run([prompt])
    return result.output.model_dump()