# Nova: Agentic AI Chatbot

Nova is an AI agent chatbot with a talking robot mascot. The LLM decides by itself when to
**search the web**, when to **generate an image**, and when to just answer.

## Features
- Robot welcome screen that waves and greets you by voice (browser text-to-speech)
- Live answers from the web, with clickable source links
- AI image generation from a text description
- Voice input (microphone) and voice replies, with a mute button
- Chat memory per session
- Works with any OpenAI-compatible LLM API (Google Gemini by default, Groq and others)

## Tech stack
| Part | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| LLM | Google Gemini through the OpenAI-compatible API (`openai` SDK) |
| Agent | Tool calling in a ReAct-style loop with a step limit |
| Web search tool | DuckDuckGo (`ddgs`) |
| Image tool | Pollinations (image URL, no key needed) |
| Frontend | One HTML file with CSS and JavaScript, Web Speech API for voice |

## Architecture
```
Browser (index.html)
   |  POST /api/chat  {session_id, message}
   v
FastAPI backend (app.py)
   |  messages + tool list
   v
LLM (Gemini)  <------------------+
   |                              |
   |-- wants a tool? --> run tool |  (loop, max 6 steps)
   |      web_search     -> DuckDuckGo results
   |      generate_image -> image URL
   '-- final answer ----> {answer, images, sources} --> Browser
```

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/nova-ai-agent.git
cd nova-ai-agent
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and put your key inside (free key: https://aistudio.google.com/api-keys):
```
LLM_API_KEY=your_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.8-flash
```
Run:
```bash
uvicorn app:app --reload
```
Open http://127.0.0.1:8000 in Chrome or Edge and click "Tap to wake up Nova".
Open http://127.0.0.1:8000/api/debug to check which settings the server is using.

> Never upload your `.env` file. It is listed in `.gitignore`.

## Project structure
```
app.py            backend: config, tools, agent loop, API routes
static/index.html frontend: chat UI, talking robot, voice, image cards
requirements.txt  Python packages
.env.example      template for your settings
```

## Roadmap
- Document RAG: upload PDFs, chunk, embed, store in ChromaDB, add a `search_docs` tool
- Streaming answers (Server-Sent Events)
- Automatic retry and fallback model when the LLM is busy
- Persistent sessions (database) and user login
- Tests, Docker and deployment
