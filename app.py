import streamlit as st
import json
from llamaapi import LlamaAPI
import PyPDF2
import google.generativeai as genai

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_response_llama(pdf_text, user_input, api_key):
    """Generate a chat response using Llama API based on the extracted PDF text and user input."""
    llama = LlamaAPI(api_key)
    prompt_template = (
        "You are a chatbot that assists users by providing information from a given document. "
        "The document content is as follows:\n\n'{pdf_text}'\n\n"
        "User's question: '{user_input}'\n"
        "Answer the question based on the document content."
    )
   
    prompt = prompt_template.format(pdf_text=pdf_text, user_input=user_input)
   
    api_request_json = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "model": "llama3.1-405b",
        "stream": False
    }
   
    try:
        response = llama.run(api_request_json)
        response_json = response.json()
        if response_json and 'choices' in response_json:
            return response_json['choices'][0]['message']['content'].strip()
        else:
            return "No valid response received from the API."
   
    except Exception as e:
        st.error(f"Error during Llama API call: {str(e)}")
        return "Error generating response."

def generate_response_gemini(pdf_text, user_input, api_key):
    """Generate a chat response using Google's Gemini model based on the extracted PDF text and user input."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt_template = (
        "You are a chatbot that assists users by providing information from a given document. "
        "The document content is as follows:\n\n'{pdf_text}'\n\n"
        "User's question: '{user_input}'\n"
        "Answer the question based on the document content."
    )
   
    prompt = prompt_template.format(pdf_text=pdf_text, user_input=user_input)
   
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
   
    except Exception as e:
        st.error(f"Error during Gemini API call: {str(e)}")
        return "Error generating response."

def main():
    st.title("PDF Chatbot with Llama API and Google's Gemini")
    st.write("Upload a PDF file, choose a model, and ask questions about its content.")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
   
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("PDF content loaded. You can now ask questions about it.")
       
        # Model selection
        model_choice = st.selectbox("Choose a model:", ("Llama API", "Google's Gemini"))
       
        # API Key input
        api_key = st.text_input("Enter your API key:")
       
        user_input = st.text_input("You: ", "")
       
        if user_input and api_key:
            if model_choice == "Llama API":
                response = generate_response_llama(pdf_text, user_input, api_key)
            else:
                response = generate_response_gemini(pdf_text, user_input, api_key)
           
            st.write("Chatbot: ", response)
   
    st.write("Type 'exit' or 'quit' to end the chat.")

if __name__ == "__main__":
    main()