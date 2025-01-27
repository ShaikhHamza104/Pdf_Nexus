import textwrap
from PyPDF2 import PdfWriter, PdfMerger, PdfReader
import os
import streamlit as st
import pdfplumber
import io
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for handling PDFs
import pandas as pd
import spacy
import torch
from transformers import pipeline
import numpy as np
from torch import Tensor
def pdf_reader(pdf_file):
    """
    Read PDF and extract text and metadata.
    
    Args:
        pdf_file: Uploaded PDF file
    
    Returns:
        Tuple of extracted text and PDF metadata
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            # Extract text with robust error handling
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            
            # Get file metadata
            pdf_info = {
                "total_pages": len(pdf.pages),
                "file_name": pdf_file.name,
                "file_size": pdf_file.size,
                "text_length": len(text)
            }
            return text.strip(), pdf_info
    except Exception as e:
        st.error(f"PDF reading error: {e}")
        return "", {}

def rotate_pdf_pages(pdf_file, rotation_angle=90):
    """
    Rotate pages of a PDF.
    
    Args:
        pdf_file: PDF file to rotate
        rotation_angle: Angle of rotation (default 90 degrees)
    
    Returns:
        Rotated PDF as BytesIO object
    """
    try:
        reader = PdfReader(pdf_file)
        writer = PdfWriter()
        
        for page in reader.pages:
            page.rotate(rotation_angle)
            writer.add_page(page)

        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"PDF rotation error: {e}")
        return None

def merge_pdfs(pdf_files):
    """
    Merge multiple PDF files.
    
    Args:
        pdf_files: List of PDF files to merge
    
    Returns:
        Merged PDF as BytesIO object
    """
    try:
        merger = PdfMerger()
        for pdf_file in pdf_files:
            merger.append(pdf_file)
        
        output = io.BytesIO()
        merger.write(output)
        merger.close()  # Properly close the merger
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"PDF merge error: {e}")
        return None


def extract_pdf_images_and_tables(pdf_file):
    """
    Extract images and tables from a PDF.
    
    Args:
        pdf_file: PDF file to extract from
    
    Returns:
        Tuple of lists containing images and tables
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            tables = []
            images = []
            
            for page_num, page in enumerate(pdf.pages):
                # Extract tables with error handling
                page_tables = page.extract_tables() or []
                for table_idx, table in enumerate(page_tables):
                    try:
                        # Ensure table has at least a header row
                        if table and len(table) > 1:
                            # Handle duplicate columns by renaming them
                            columns = table[0]
                            columns = [f"{col}_{i}" if columns.count(col) > 1 else col for i, col in enumerate(columns)]
                            
                            # Create DataFrame
                            df = pd.DataFrame(table[1:], columns=columns)
                            tables.append((f"Page {page_num + 1} Table {table_idx + 1}", df))
                    except Exception as table_error:
                        st.warning(f"Could not process table on page {page_num + 1}: {table_error}")

                # Extract images with error handling
                page_images = page.images or []
                for img_idx, img in enumerate(page_images):
                    try:
                        img_bytes = io.BytesIO(img["stream"].get_data())
                        pil_image = Image.open(img_bytes)
                        images.append((f"Page {page_num + 1} Image {img_idx + 1}", pil_image))
                    except Exception as img_error:
                        st.warning(f"Could not process image on page {page_num + 1}: {img_error}")

            return images, tables
    except Exception as e:
        st.error(f"PDF image and table extraction error: {e}")
        return [], []
    
def extract_keywords(pdf_file):
    """
    Extract keywords using spaCy NLP.
    
    Args:
        pdf_file: PDF file to extract keywords from
    
    Returns:
        List of unique keywords
    """
    try:
        # Load spaCy model with error handling
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
            return []

        # Extract text and process with NLP
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            
            # Process text with spaCy
            doc = nlp(text)
            
            # Extract and deduplicate keywords
            keywords = list(dict.fromkeys([
                token.lemma_.lower() 
                for token in doc 
                if token.pos_ in ["NOUN", "PROPN"] and len(token.lemma_) > 2
            ]))
            
            return keywords
    except Exception as e:
        st.error(f"Keyword extraction error: {e}")
        return []


def pdf_summarizer(pdf_file):
    """
    Generate a summary of the PDF content.
    
    Args:
        pdf_file: PDF file to summarize. Should specify the expected type (e.g., Path, file object, or bytes)
    
    Returns:
        str: Summarized text or error message
    """
    try:
        # Extract text from PDF
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        if not text.strip():
            return "The PDF appears to be empty or contains no extractable text."
            
        # Use smaller chunks to stay within BART's limits (1024 tokens ≈ 750-1000 words)
        chunk_size = 1000  # Reduced chunk size
        chunks = textwrap.wrap(text, chunk_size, break_long_words=False, break_on_hyphens=False)
        
        # Initialize summarizer with explicit model max length
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=torch.cuda.is_available() and "cuda" or "cpu",
            min_length=30,
            truncation=True
        )
        
        summaries = []
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            try:
                # Remove max_length from here since it's set in pipeline initialization
                chunk_summary = summarizer(
                    chunk,
                    do_sample=False,
                    num_beams=4
                )
                if chunk_summary and len(chunk_summary) > 0:
                    summaries.append(chunk_summary[0]['summary_text'])
            except Exception as e:
                print(f"Chunk processing error: {str(e)}")
                continue
        
        if not summaries:
            return "Could not generate a summary from the PDF content."
            
        final_summary = " ".join(summaries)
        
        # Second summarization if needed
        if len(final_summary) > chunk_size:
            try:
                final_summary = summarizer(
                    final_summary,
                    do_sample=False,
                    num_beams=4
                )[0]['summary_text']
            except Exception as e:
                print(f"Final summarization error: {str(e)}")
        
        return final_summary.strip()

    except Exception as e:
        error_msg = f"PDF Summarization error: {str(e)}"
        print(error_msg)
        return error_msg
    
def pdf_semantic_analysis(pdf_file):
    """
    Perform basic semantic analysis on PDF content.
    
    Args:
        pdf_file: PDF file to analyze
    
    Returns:
        Dictionary of semantic analysis results
    """
    try:
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Extract text from PDF
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        # Process text
        doc = nlp(text)
        
        # Semantic analysis components
        analysis = {
            "named_entities": [
                (ent.text, ent.label_) 
                for ent in doc.ents
            ],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
            "token_count": len(doc),
            "sentiment": None  # Placeholder for more advanced sentiment analysis
        }
        
        return analysis
    except Exception as e:
        st.error(f"Semantic Analysis error: {e}")
        return {}

def main():
    """
    Main Streamlit application entry point
    Provides multiple PDF processing features
    """
    st.title("Advanced PDF Processing App")

    # Sidebar feature selection
    app_mode = st.sidebar.selectbox("Choose Feature", [
        "PDF Reader",
        "PDF Merger",
        "PDF Image & Table Extractor",
        "PDF Rotator",
        "PDF Keyword Extractor",
        "PDF Semantic Analysis",
        "PDF Summarizer"
    ])

    # File uploader with multiple file support
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'], accept_multiple_files=True)

    if not uploaded_file:
        st.warning("Please upload a PDF file.")
        return

    try:
        if app_mode == "PDF Reader":
            for file in uploaded_file:
                text, pdf_info = pdf_reader(file)
                st.subheader(f"PDF: {pdf_info.get('file_name', 'Unknown')}")
                
                # Display extracted text
                st.text_area("Extracted Text", text, height=400)

                # Display PDF information
                st.subheader("PDF Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Total Pages:** {pdf_info.get('total_pages', 'N/A')}")
                with col2:
                    st.write(f"**File Size:** {pdf_info.get('file_size', 'N/A')} bytes")
                    
        elif app_mode == "PDF Image & Table Extractor":
            images, tables = extract_pdf_images_and_tables(uploaded_file[0])
            
            st.subheader("Extracted Images")
            for title, image in images:
                st.write(f"**{title}**")
                st.image(image, use_container_width=True)

            st.subheader("Extracted Tables")
            for title, table in tables:
                st.write(f"**{title}**")
                st.dataframe(table)

        elif app_mode == "PDF Keyword Extractor":
            keywords = extract_keywords(uploaded_file[0])
            st.write("Extracted Keywords:")
            st.write(", ".join(keywords))

        elif app_mode == "PDF Summarizer":
            summary = pdf_summarizer(uploaded_file[0])
            st.write("Document Summary:")
            st.write(summary)

        elif app_mode == "PDF Semantic Analysis":
            analysis = pdf_semantic_analysis(uploaded_file[0])
            st.subheader("Semantic Analysis Results")
            st.write("Named Entities:", analysis.get('named_entities', []))
            st.write("Noun Chunks:", analysis.get('noun_chunks', []))
            st.write("Total Tokens:", analysis.get('token_count', 0))

        elif app_mode == "PDF Merger":
            if len(uploaded_file) > 1:
                merged_pdf = merge_pdfs(uploaded_file)
                if merged_pdf:
                    st.download_button("Download Merged PDF", merged_pdf, file_name="merged_pdf.pdf")
            else:
                st.warning("Please upload multiple PDFs to merge")
                
        elif app_mode == "PDF Rotator":
            for file in uploaded_file:
                rotated_pdf = rotate_pdf_pages(file)
                if rotated_pdf:
                    st.download_button(f"Download Rotated {file.name}", rotated_pdf, file_name=f"rotated_{file.name}")
                    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main()