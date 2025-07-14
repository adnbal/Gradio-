import gradio as gr
import openai
import os
from dotenv import load_dotenv

# Load .env variables (OpenAI API key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def respond_to_voice(audio_path):
    try:
        # Step 1: Transcribe audio to text
        transcript = openai.Audio.transcribe("whisper-1", audio_path)["text"]

        # Step 2: Generate GPT-4 response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI voice assistant that answers questions about CV improvement, job applications, career advice in data science, AI, and business analytics."},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7
        )

        # Return combined result
        return f"🗣️ You said: {transcript}\n\n🤖 Assistant: {response['choices'][0]['message']['content']}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Launch Gradio UI
gr.Interface(
    fn=respond_to_voice,
    inputs=gr.Audio(source="microphone", type="filepath", label="🎤 Speak your question"),
    outputs=gr.Textbox(label="💬 AI Response"),
    title="🎙️ Voice Career Assistant",
    description="Speak your question about CV, AI careers, data science, or analytics roles. This assistant will transcribe and respond using GPT-4."
).launch()
