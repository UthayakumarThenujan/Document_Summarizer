import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers import T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os


def save_summary_as_pdf(summary, filename):
    # Get current date and time
    print("save is start")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory if it doesn't exist
    directory = "summary"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct filename with timestamp
    pdf_filename = os.path.join(directory, f"X{timestamp}.pdf")
    print(pdf_filename)
    # Create PDF
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, "Summary")
    c.drawString(100, 730, summary)
    c.save()
    print("saved")
    return pdf_filename


# model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map="auto", torch_dtype=torch.float32
)


# file loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts


# LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    summary = result[0]["summary_text"]
    return summary


# function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# function to save the summary as PDF
def save_summary_as_pdf(summary, filename):
    c = canvas.Canvas(f"summary/{filename}.pdf", pagesize=letter)
    c.drawString(100, 750, "Summary")
    c.drawString(100, 730, summary)
    c.save()


# streamlit code
st.set_page_config(layout="wide")


def main():
    st.title("Document Summarization App")

    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            print(uploaded_file.name)
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_display = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.subheader("Summary")
                editable_summary = st.text_area(
                    "Editable Summary", value=summary, height=200
                )
                # if st.button("Save Summary"):
                #     print("fun start")
                #     pdf_filename = save_summary_as_pdf(summary, uploaded_file.name)
                #     print("fun end")
                #     st.success(f"Summary saved as {pdf_filename}")


if __name__ == "__main__":
    main()
