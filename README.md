# Vietnamese Healthcare Chatbot
 
Project for starting LLM on medical data.

## 1. Introduction

This project aims to build a Vietnamese healthcare chatbot leveraging advanced Retrieve-Augment-Generate (RAG) capabilities, such as Hybrid Search, Density Chain, and Re-ranking. The chatbot is designed to provide accurate and helpful health-related information and support.
<p align="center">
  <img width="800" alt="Gradio Interface" src="https://github.com/luongphambao/medical_llm/assets/127675330/c04580c5-d63e-4076-930d-350f0d520083">
</p>
<p align="center">
  <em>Gradio Interface</em>
</p>

### Key features:

**🌟 Advanced RAG Capabilities:**  Utilizing Hybrid Search, Density Chain, and Re-ranking to enhance information retrieval and generation.

**🌟 Model Experimentation:**  Testing various LLM models, including GPT, Gemini, Vistral,Vinallama etc., to find the best fit for different use case

**🌟 Model and Embedding Selection** : Allowing users to choose the most appropriate model and embeddings to create a personalized assistant.

**🌟 User-Friendly Interface:**  Using Gradio to create an intuitive and easy-to-use interface for beginners.

## 2. How to use?
### Getting Started
```
git clone https://github.com/luongphambao/medical_llm.git
cd medical_llm
```
```
conda create -n medical_llm python=3.10
conda activate medical_llm
```
```
pip install -r requirements.txt
```
### Using Gradio
Add the OpenAI API KEY in the **.env** file.
```python
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI-...
```
Then, you can run the **app.py** file and a Gradio interface will appear.
```
python -m  run app.py
```
