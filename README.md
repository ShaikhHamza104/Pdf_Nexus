# Advanced PDF Processing App

## Overview
A Streamlit-based application for comprehensive PDF processing, offering multiple features including text extraction, image and table extraction, keyword analysis, semantic analysis, PDF merging, and rotation.

## Features
- PDF Reading and Text Extraction
- PDF Image and Table Extraction
- Keyword Extraction using spaCy
- PDF Summarization with Hugging Face
- Semantic Analysis
- PDF Merging
- PDF Page Rotation

## Prerequisites
- Python 3.8+
- Tesseract OCR
- System dependencies listed in requirements.txt

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-processing-app.git
cd pdf-processing-app
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

4. Install Tesseract OCR
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On macOS: `brew install tesseract`
- On Windows: Download from official Tesseract GitHub repository

## Running the Application
```bash
streamlit run app.py
```

## Requirements
- streamlit
- pdfplumber
- Pillow
- pytesseract
- PyMuPDF
- PyPDF2
- pandas
- spacy==3.5.2
- transformers
- torch

## Supported PDF Operations
- Text Extraction
- Image Extraction
- Table Extraction
- Keyword Extraction
- Semantic Analysis
- Summarization
- PDF Merging
- PDF Rotation

## Limitations
- Large PDFs may require more memory
- OCR accuracy depends on Tesseract configuration
- Semantic analysis uses basic spaCy model

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

## License
[MIT](https://choosealicense.com/licenses/mit/)
