# Nova: Agentic AI Chatbot (Claude + Web Search + Image Generation)

A chatbot with a talking robot mascot. Nova is an **LLM agent** that decides on its own when to
search the web, when to generate an image, and when to just answer.

## Features
- Welcome screen where the robot waves and greets you by voice (browser text-to-speech)
- Live web answers with clickable source links (DuckDuckGo search)
- AI image generation from text (Pollinations, no API key needed)
- Chat memory per session, voice replies toggle, microphone input
- ReAct-style agent loop with a max-step safety limit

## Setup
```bash
cd nova
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env              # then put your ANTHROPIC_API_KEY inside .env
uvicorn app:app --reload
```
Open http://localhost:8000 in **Chrome or Edge** (best voice and mic support) and click
"Tap to wake up Nova". Browsers block auto-playing speech, so the click is what lets the robot talk.

## How it works
```
Browser UI  ->  POST /api/chat  ->  Agent loop (Claude + tools)
                                      |-- web_search      (DuckDuckGo)
                                      |-- generate_image  (Pollinations URL)
                                      '-- final answer + images + sources
```
1. The user message and the tool list go to Claude.
2. If Claude asks for a tool, the backend runs it and sends the result back.
3. This repeats (max 6 steps) until Claude gives a final answer.

## Interview talking points
- Agent vs plain RAG: the LLM chooses tools instead of following a fixed pipeline (ReAct pattern).
- Tool descriptions matter, because they are how the model decides what to call.
- Safety: step limit, tool results treated as data (prompt-injection awareness), API key in .env.
- Limits and next steps: in-memory sessions (use Redis/DB in production), add a RAG tool over
  uploaded PDFs with ChromaDB, streaming responses, evaluation of answer quality.

## Ideas to extend (good for interviews)
- Add a `search_docs` tool: chunk PDFs, embed, store in ChromaDB, retrieve top-k
- Stream tokens to the UI with Server-Sent Events
- Swap Pollinations for another image API behind the same `generate_image` tool
