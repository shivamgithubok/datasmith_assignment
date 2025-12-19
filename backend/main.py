from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
from PIL import Image
import pytesseract
import PyPDF2
import speech_recognition as sr
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv('OPENROUTER_API_KEY')

def extract_content(text: str = None, file = None):
    if file:
        filename = file.filename.lower()
        if filename.endswith(('.jpg', '.png')):
            image = Image.open(file.file)
            extracted_text = pytesseract.image_to_string(image)
            return extracted_text, "OCR"
        elif filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(file.file)
            extracted_text = ''
            for page in reader.pages:
                extracted_text += page.extract_text()
            return extracted_text, "PDF"
        elif filename.endswith(('.mp3', '.wav', '.m4a')):
            try:
                from pydub import AudioSegment
                import io
                # Read file into BytesIO
                file.file.seek(0)
                audio_data = io.BytesIO(file.file.read())
                # Load audio
                if filename.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(audio_data)
                elif filename.endswith('.m4a'):
                    audio = AudioSegment.from_file(audio_data, format='m4a')
                else:
                    audio = AudioSegment.from_wav(audio_data)
                # Convert to WAV in memory
                wav_io = io.BytesIO()
                audio.export(wav_io, format='wav')
                wav_io.seek(0)
                r = sr.Recognizer()
                with sr.AudioFile(wav_io) as source:
                    audio_rec = r.record(source)
                extracted_text = r.recognize_google(audio_rec)
                return extracted_text, "STT"
            except Exception as e:
                return f"Error processing audio: {str(e)}", "STT"
        else:
            return None, "unknown"
    else:
        if text and 'youtube.com' in text:
            try:
                video_id = text.split('v=')[1].split('&')[0]
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                extracted_text = ' '.join([t['text'] for t in transcript])
                return extracted_text, "YouTube"
            except:
                return text, "text"
        return text, "text"

def process_query(extracted_text, user_query):
    prompt = f"""
Given the extracted content: {extracted_text[:1000]}...
And user query: {user_query}
Determine the task: one of [summarization, sentiment_analysis, code_explanation, conversational, transcription_summary, text_extraction]
If unclear, respond with 'CLARIFY: question'
Else, respond with 'TASK: task_name'
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        if result.startswith('CLARIFY:'):
            return {'type': 'clarify', 'question': result[8:].strip()}
        elif result.startswith('TASK:'):
            task = result[5:].strip()
            return {'type': 'task', 'task': task}
        else:
            return {'type': 'clarify', 'question': 'Could you clarify what you want me to do?'}
    except:
        return {'type': 'clarify', 'question': 'Error in processing, please clarify.'}

def perform_task(task, text):
    if task == 'summarization':
        prompt = f"Summarize the following text in: 1-line summary, 3 bullets, 5-sentence summary.\n\n{text}"
    elif task == 'sentiment_analysis':
        sentiment = TextBlob(text).sentiment
        label = 'positive' if sentiment.polarity > 0 else 'negative' if sentiment.polarity < 0 else 'neutral'
        return f"Sentiment: {label}, Confidence: {abs(sentiment.polarity)}, Justification: {sentiment}"
    elif task == 'code_explanation':
        prompt = f"Explain what this code does, detect any bugs, and mention time complexity if applicable.\n\n{text}"
    elif task == 'conversational':
        prompt = f"Respond friendly and helpfully to: {text}"
    elif task == 'transcription_summary':
        prompt = f"Summarize the transcription in: 1-line summary, 3 bullets, 5-sentence summary.\n\n{text}"
    elif task == 'text_extraction':
        return text
    else:
        return "Task not recognized."

    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except:
        return "Error in generating response."

@app.post("/process")
async def process(text: str = Form(None), file: UploadFile = File(None), query: str = Form(...)):
    extracted, source = extract_content(text, file)
    if extracted is None:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)
    result = process_query(extracted, query)
    if result['type'] == 'clarify':
        return {"response": result['question'], "extracted": extracted, "source": source}
    else:
        output = perform_task(result['task'], extracted)
        return {"response": output, "extracted": extracted, "source": source}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)