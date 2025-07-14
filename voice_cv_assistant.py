# voice_cv_assistant.py

import gradio as gr
import openai
import os

# Set OpenAI key from environment or replace with your key directly
openai.api_key = os.getenv("OPENAI_API_KEY")  # or replace with: "sk-..."

# 🔁 Voice input → Whisper → GPT-4 reply
def respond_to_voice(audio_path):
    try:
        # Transcribe voice using Whisper
        transcript = openai.Audio.transcribe("whisper-1", audio_path)
        user_input = transcript["text"]
        
        # Use GPT-4 for reply
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for CV improvement, job matching, and career advice."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        reply = response["choices"][0]["message"]["content"]
        return f"🗣️ You said: {user_input}\n\n🤖 Assistant: {reply}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🎙️ Build the Gradio Interface
iface = gr.Interface(
    fn=respond_to_voice,
    inputs=gr.Audio(source="microphone", type="filepath", label="🎤 Speak your question"),
    outputs=gr.Textbox(label="💬 Assistant Response"),
    title="🎙️ AI Voice Career Assistant",
    description="Ask anything about your CV, job fit, or AI career direction. Powered by OpenAI GPT-4 + Whisper."
)

# 🟢 Launch
if __name__ == "__main__":
    iface.launch()
