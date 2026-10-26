import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
    revise_chain,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Multi-Agent Research Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return {"search_text_to_reader": config.SEARCH_TEXT_TO_READER}


@app.get("/api/research")
def research(topic: str, max_chars: int = config.SEARCH_TEXT_TO_READER):
    topic = topic.strip()

    def stream():
        try:
            # 1 - Search
            yield sse("step", {"step": 1, "status": "running"})
            res = build_search_agent().invoke({
                "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
            })
            search_results = res["messages"][-1].content
            yield sse("step", {"step": 1, "status": "done", "output": search_results})

            # 2 - Read
            yield sse("step", {"step": 2, "status": "running"})
            res = build_reader_agent().invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the {config.URLS_TO_READ} most relevant URLs and scrape each one "
                    f"for deeper content.\n\nSearch Results:\n{search_results[:max_chars]}"
                )]
            })
            scraped = res["messages"][-1].content
            yield sse("step", {"step": 2, "status": "done", "output": scraped})

            research_text = (
                f"SEARCH RESULTS:\n{search_results}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{scraped}"
            )[:config.RESEARCH_CHARS]

            # 3 - Write draft
            yield sse("step", {"step": 3, "status": "running"})
            draft = writer_chain.invoke({"topic": topic, "research": research_text})
            yield sse("step", {"step": 3, "status": "done", "output": draft})

            # 4 - First review
            yield sse("step", {"step": 4, "status": "running"})
            review1 = critic_chain.invoke({"report": draft})
            yield sse("step", {"step": 4, "status": "done", "output": review1})

            # 5 - Revise using the feedback
            yield sse("step", {"step": 5, "status": "running"})
            revised = revise_chain.invoke({
                "topic": topic,
                "research": research_text,
                "report": draft,
                "feedback": review1,
            })
            yield sse("step", {"step": 5, "status": "done", "output": revised})

            # 6 - Final review
            yield sse("step", {"step": 6, "status": "running"})
            review2 = critic_chain.invoke({"report": revised})
            yield sse("step", {"step": 6, "status": "done", "output": review2})

            yield sse("done", {})
        except Exception as e:
            yield sse("pipeline_error", {"message": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )