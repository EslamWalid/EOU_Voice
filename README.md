# Arabic End-of-Utterance (EOU) Detection Model

## 1. Project Overview
This project focuses on building an Arabic End-of-Utterance (EOU) detection model
designed for real-time voice agents. The model predicts the probability that a speaker
has finished their turn based on partial or complete transcriptions.

The system is optimized for real-time usage and integrated with LiveKit agents to
improve conversational naturalness, especially for Arabic dialogues with emphasis on
the Saudi dialect.

---

## Project Structure

    .
    ├── notebook/
    │   └── finetuning_arabic_eou.ipynb
    │
    ├── main.py
    ├── README.md
    ├── .gitignore

### Directory Description

#### `notebook/`

Contains the **Google Colab notebook** used to fine-tune the **Arabic End-Of-Utterance (EOU)** model.

This notebook includes:
- Data loading and preprocessing
- Labeling utterances (EOU / non-EOU)
- Model training and fine-tuning
- Evaluation and metrics analysis

The notebook is intended to be run on **Google Colab** for GPU acceleration.

---

#### `main.py`

Implements a **simple LiveKit voice agent** responsible for real-time voice interaction.

It integrates the following components:

- **Speech-to-Text (STT):**
  - ElevenLabs

- **Large Language Model (LLM):**
  - Gemini (`gemini-2.5-flash`)

- **Text-to-Speech (TTS):**
  - Gemini (`gemini-2.5-flash-preview-tts`)

- **End-Of-Utterance Detection (EOU):**
  - Uses the **Arabic EOU SDK** to determine when the speaker has finished their utterance and trigger the LLM response accordingly.


## 2. Problem Definition
End-of-Utterance detection is a critical component in real-time voice systems.
Incorrect detection can lead to:
- Interrupting users mid-sentence
- Long and unnatural response delays

Arabic presents additional challenges due to:
- Dialectal variations
- Frequent fillers and discourse markers
- Flexible sentence endings

This project addresses these challenges by training a lightweight Arabic EOU classifier
suitable for production environments.

## 3. Dataset Creation
### 3.1 Data Sources
The dataset was collected from Arabic Saudi conversational videos including:
- Podcasts [youtube-video](https://www.youtube.com/watch?v=XIii4L76hTU)
- Interview [youtube-video](https://www.youtube.com/watch?v=coZSCImsMj8)

The selected sources primarily reflect the Saudi dialect to ensure dialectal relevance
for target deployment scenarios.

### 3.2 Transcription
Audio data was transcribed using Whisper,
followed by manual cleaning to remove obvious transcription artifacts while
preserving natural conversational patterns.

### 3.3 Annotation Strategy
Each utterance was labeled into one of two classes:

- **EOU = 1**: The speaker has completed their turn.
- **EOU = 0**: The speaker is likely to continue speaking.

Annotation decisions were based on:
- Semantic completeness
- Discourse markers
- Contextual continuation likelihood


## 4. Model Selection
### 4.1 Candidate Models
Several Arabic language models were considered:
- AraBERT
- MARBERT
- CAMeL-BERT

### 4.2 Model Choice Justification
bert-base-arabertv02 was selected due to:
- Strong performance on MSA and dialectal Arabic
- Stable tokenizer behavior for short utterances
- Lower inference latency compared to larger models
- Suitability for CPU-based real-time deployment

## 5. Model Training
The model was fine-tuned as a binary sequence classification task.

### Training Configuration
- Base model: bert-base-arabertv02
- Learning rate: 2e-5
- Batch size: 32
- Epochs: 30
- Loss function: Cross-Entropy Loss

## 6. Evaluation
### 6.1 Metrics
The model was evaluated using:
- Accuracy
- ROC-AUC

### 6.2 Results
| Metric | Score |
|------|------|
| Accuracy | 0.68 |
| ROC-AUC | 0.80 |

## 7. Deployment
The fine-tuned model was deployed as a reusable SDK and integrated with LiveKit agents
to perform real-time EOU detection.

The SDK loads the Hugging Face model once and returns an EOU probability for each
incoming transcription segment.


## 8. Arabic EOU SDK

The fine-tuned Arabic EOU model is distributed as a Python package and can be installed
via pip:
    
    pip install arabic-eou

SDK Usage Example

    ```python

    from arabic_eou.runner import ArabicEOURunner

    async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt= STT,              # put your STT
        llm=LLM,               # put your LLM
        tts=TTS,               # put your TTS
        vad=silero.VAD.load(),

        turn_detection=ArabicEOURunner(), # <-- MY SDK
    )```


## 9. Setup and Running the Project

### 9.1 Create a Virtual Environment (Recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate     # Linux / macOS
    venv\Scripts\activate        # Windows


### 9.2 Install Dependencies

    ```bash
    pip install arabic-eou
    pip install livekit-agents
    pip install python-dotenv

### 9.3 Environment Variables

Create a .env file in the project root with the following:

    ```bash
    LIVEKIT_API_KEY=your_livekit_api_key
    LIVEKIT_API_SECRET=your_livekit_api_secret
    LIVEKIT_URL=wss://your-livekit-server

    ELEVENLABS_API_KEY=your_elevenlabs_api_key
    GOOGLE_API_KEY=your_google_api_key


9.4 Running the LiveKit Agent

To start the LiveKit agent, run:

    ```bash
    python main.py console