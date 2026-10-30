# 📝 Transformer-Based Text Summarizer

**Transforming lengthy text into concise, meaningful summaries using Natural Language Processing and Transformer models.**

A machine learning project that uses the **FLAN-T5 Transformer architecture** to summarize long paragraphs and extract their essential information. The project combines model fine-tuning, text preprocessing, and a web application to make automatic text summarization accessible through a simple user interface.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-yellow?style=for-the-badge" alt="Transformers">
  <img src="https://img.shields.io/badge/NLP-Text%20Summarization-blueviolet?style=for-the-badge" alt="NLP">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/HTML-CSS-JavaScript-orange?style=for-the-badge" alt="Frontend">
</p>

---

## 🚀 Project Overview

Reading lengthy articles, documents, and paragraphs can be time-consuming. This project addresses that problem using **abstractive text summarization**, where a Transformer model generates a shorter version of the input while attempting to preserve its central meaning.

Users can paste text into a web interface and generate a concise summary. The interface also displays the original word count, summary word count, and percentage reduction in length.

### ✨ Key Features

- **Transformer-based summarization:** Uses the FLAN-T5 sequence-to-sequence model.
- **Fine-tuned model:** Trained on the SAMSum dataset to adapt the pretrained model to summarization.
- **Text preprocessing:** Handles missing values, newline characters, excessive whitespace, and HTML tags.
- **Interactive web interface:** Provides a text input area and a dedicated summary display.
- **Summary statistics:** Displays word counts and the percentage reduction in text length.
- **Copy summary:** Allows users to copy generated summaries.
- **Recent summaries:** Makes previous summaries available for quick reuse during the session.
- **Dark/light theme:** Provides a theme toggle for the interface.
- **API integration:** Uses FastAPI to connect the web interface with the summarization model.

---

## 📑 Table of Contents

1. [How It Works](#-how-it-works)
2. [Technology Stack](#-technology-stack)
3. [Dataset](#-dataset)
4. [Model and Training](#-model-and-training)
5. [Application Workflow](#-application-workflow)
6. [Project Structure](#-project-structure)
7. [Installation and Setup](#-installation-and-setup)
8. [Running the Application](#-running-the-application)
9. [Example](#-example)
10. [Challenges and Learning Outcomes](#-challenges-and-learning-outcomes)
11. [Future Improvements](#-future-improvements)
12. [Author](#-author)

---

## 🧠 How It Works

The application follows a straightforward machine learning pipeline:

**Input Text → Preprocessing → Tokenization → Transformer Model → Generated Summary → Web Interface**

1. **Input:** The user enters a paragraph or other text.
2. **Preprocessing:** The input is cleaned to remove unnecessary whitespace and formatting artifacts.
3. **Tokenization:** The tokenizer converts the text into numerical token IDs that the model can process.
4. **Summarization:** The fine-tuned FLAN-T5 model generates a shorter sequence based on the input.
5. **Decoding:** The generated token IDs are converted back into readable text.
6. **Display:** The application displays the summary and calculates text-reduction statistics.

The model uses a maximum input length of 512 tokens and a maximum generated length of 150 tokens in the summarization configuration. Longer inputs may be truncated.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning model execution and training |
| Hugging Face Transformers | Loading, fine-tuning, and generating text with FLAN-T5 |
| FLAN-T5 Base | Pretrained sequence-to-sequence model used for summarization |
| SAMSum | Dataset used for training and validation |
| Pandas | Dataset loading and preprocessing |
| FastAPI | Backend API and web application |
| Uvicorn | Local ASGI server |
| HTML | Web page structure |
| CSS | Layout, styling, and themes |
| JavaScript | User interactions and frontend behavior |
| Google Colab | Model training and experimentation |
| GitHub | Source code and project version control |
| Hugging Face Hub | Hosting the trained model files |

---

## 📚 Dataset

The project uses the **SAMSum dataset**, which contains dialogue conversations paired with human-written summaries.

The dataset is used to train the model to generate concise summaries from conversational text.

### Data preparation

The preprocessing pipeline includes:

- Handling missing text values.
- Replacing newline characters with spaces.
- Normalizing repeated whitespace.
- Removing HTML tags where present.
- Tokenizing dialogue text and reference summaries.
- Truncating inputs and targets to their configured maximum lengths.
- Masking padding tokens in training labels so they do not contribute to the loss.

The notebook uses sampled subsets of the dataset for experimentation:

- Training: 10,000 examples.
- Validation: 700 examples.

These are the configured sample sizes in the training notebook, not the full size of the SAMSum dataset.

---

## 🤖 Model and Training

### Model: FLAN-T5 Base

The project uses `google/flan-t5-base`, a pretrained encoder-decoder Transformer model.

The model is suitable for sequence-to-sequence tasks, including text summarization, because it encodes the input text and generates an output sequence.

### Training configuration

| Parameter | Configuration |
|---|---|
| Base model | `google/flan-t5-base` |
| Task | Abstractive text summarization |
| Training examples | 10,000 |
| Validation examples | 700 |
| Maximum input length | 512 tokens |
| Maximum target length | 150 tokens |
| Training epochs | Up to 8 |
| Training batch size | 4 |
| Evaluation batch size | 4 |
| Learning rate | `5e-5` |
| Weight decay | `0.01` |
| Optimizer and scheduler | Hugging Face Trainer defaults |
| Checkpoint strategy | Save at the end of each epoch |
| Evaluation strategy | Evaluate at the end of each epoch |

The final model is saved locally and loaded by the application for inference.

**Important:** The eight epochs are the configured maximum. Actual training completion and the final model's evaluation results should be verified from the training logs before reporting them as completed results.

### Text generation

During inference, the model uses beam search with four beams and early stopping to generate a summary.

The objective is to produce a concise, readable output while retaining the main ideas of the input text.

---

## 🔄 Application Workflow

The application has two main components.

### 1. Machine Learning Component

The notebook handles:

- Dataset loading and preprocessing.
- Tokenization.
- Loading the pretrained FLAN-T5 model.
- Fine-tuning and validation.
- Saving the trained model and tokenizer.
- Testing summary generation.

### 2. Web Application

The web application uses FastAPI to serve the interface and connect requests to the trained model.

Users can:

1. Open the application in their browser.
2. Enter or paste text.
3. Select **Summarize**.
4. View the generated summary.
5. Copy the summary.
6. Review the word count and percentage reduction.
7. Reuse a recent summary or switch the interface theme.

---

## 📂 Project Structure

The repository should follow a structure similar to this:

```text
Text-Summarizer-Transformer/
│
├── app.py
├── index.html
├── requirements.txt
├── .gitignore
├── README.md
│
├── notebooks/
│   └── Text_Summarizer_Transformer_Project.ipynb
│
└── saved_summary_model/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin
```

The exact paths may differ depending on how the files are organized in the repository.

**Model hosting:** The trained model weights are large, so they may be hosted separately on the Hugging Face Hub rather than committed directly to GitHub. If you use that approach, update `app.py` to load the model from the Hub and configure the deployment to download it at startup.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system.
- Git, if you want to clone the repository.
- Internet access for downloading dependencies and model files.
- Sufficient RAM and storage to load the Transformer model.

### Step 1: Clone the repository

Replace the repository URL below if you rename your GitHub repository.

```bash
git clone https://github.com/Krish07chawla/-Text-Summarizer-Transformer.git
cd -Text-Summarizer-Transformer
```

### Step 2: Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` contains the libraries required by the actual application, including FastAPI, Uvicorn, PyTorch, Transformers, and Jinja2 if your backend uses Jinja2 templates.

### Step 4: Configure the model path

The application must be able to locate the trained model and tokenizer.

If the model is hosted on Hugging Face, configure the application to load the correct model repository ID.

If the model is stored locally, ensure that the path in `app.py` points to the directory containing `config.json`, the model weights, and the tokenizer files.

---

## ▶️ Running the Application

Start the FastAPI server with:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open the following address in your browser:

```text
http://127.0.0.1:8001
```

The API documentation is available at:

```text
http://127.0.0.1:8001/docs
```

These commands run the application locally. A public website requires a separate deployment configuration and a publicly accessible hosting service.

---

## 🧪 Example

### Input

> Online education has become increasingly popular because it allows students to learn from anywhere and at their own pace. However, it also presents several challenges. Teachers can improve online education by making lessons interactive, providing regular feedback, and encouraging students to participate in discussions.

### Expected summary

> Online education offers flexible learning but requires interactive lessons, regular feedback, and student participation to be effective.

*Example output is illustrative. Actual summaries may differ because they are generated by the model.*

### Summary statistics

The interface calculates:

- Original word count.
- Summary word count.
- Percentage reduction in word count.

The percentage reduction is calculated as:

`((Original Word Count - Summary Word Count) / Original Word Count) × 100`

A higher reduction percentage does not necessarily mean a better summary. The summary must also preserve the important information in the original text.

---

## 🧩 Challenges and Learning Outcomes

This project provides practical experience with several important machine learning and software engineering concepts.

### Technical challenges

- Preparing raw text for a Transformer model.
- Tokenizing inputs and target summaries correctly.
- Masking padding tokens in sequence-to-sequence training labels.
- Fine-tuning a pretrained model within available computing resources.
- Saving checkpoints and loading the trained model for inference.
- Integrating a deep learning model with a FastAPI backend.
- Connecting the backend to an interactive frontend.
- Handling large model files when preparing a project for GitHub and deployment.

### Key learning outcomes

- Understanding encoder-decoder Transformer models.
- Applying transfer learning to an NLP task.
- Working with the Hugging Face Transformers library.
- Building a text preprocessing and tokenization pipeline.
- Generating text using beam search.
- Serving model predictions through a web API.
- Developing an interactive interface for a machine learning application.
- Preparing an ML project for source control and deployment.

---

## 🔮 Future Improvements

Potential improvements include:

- **Improved evaluation:** Report ROUGE scores and other appropriate summarization metrics.
- **Better summary control:** Allow users to select short, medium, and detailed summaries.
- **Long-document support:** Implement chunking and hierarchical summarization for text exceeding the model's input limit.
- **Performance optimization:** Investigate quantization or other inference optimizations to reduce latency and memory usage.
- **Robust input handling:** Add input validation, clearer error messages, and limits for very large submissions.
- **Deployment:** Host the frontend and backend so users can access the application through a public URL.
- **Automated testing:** Add tests for the API, input validation, and summarization pipeline.
- **Model comparison:** Compare the fine-tuned model with the original pretrained model and alternative summarization architectures.

---

## 👨‍💻 Author

**Krish Chawla**

Computer Science Engineering Student | Interested in Machine Learning, Natural Language Processing, and Generative AI.

- **GitHub:** [Krish07chawla](https://github.com/Krish07chawla)
- **Project Repository:** [Text Summarizer Transformer](https://github.com/Krish07chawla/-Text-Summarizer-Transformer)
- **Live Demo:** Add your deployed application URL here once deployment is complete.
- **Trained Model:** [FLAN-T5 Text Summarizer on Hugging Face](https://huggingface.co/coherentcoder/text-summarizer-flan-t5)

---

## 📄 License

This repository uses the license selected in the GitHub repository. Review the repository's `LICENSE` file before redistributing the code or model.

---

**If you find this project interesting, feel free to explore the repository and try the application once the live deployment is available.**
