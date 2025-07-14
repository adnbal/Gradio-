import gradio as gr
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def respond_to_voice(audio_path):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_path)["text"]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI voice assistant that answers questions about CVs, AI jobs, and career advice."},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7
        )

        return f"🗣️ You said: {transcript}\n\n🤖 Assistant: {response['choices'][0]['message']['content']}"
    except Exception as e:
        return f"❌ Error: {e}"

gr.Interface(
    fn=respond_to_voice,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Textbox(),
    title="🎙️ Voice Career Assistant",
    description="Ask your questions by voice — get smart replies using Whisper & GPT-4.",
).launch()
