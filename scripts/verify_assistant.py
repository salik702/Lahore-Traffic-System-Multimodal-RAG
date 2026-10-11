"""Read-only live acceptance checks; uses the running API and real providers."""
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

CASES = [
    ("What happens after a road accident?", "ACCIDENT_GUIDANCE"),
    ("What should I do if someone is injured?", "EMERGENCY_GUIDANCE"),
    ("What is the helmet fine in Lahore?", "TRAFFIC_FINE"),
    ("Explain common traffic signs in Pakistan.", "TRAFFIC_SIGN"),
    ("Show me accident images from Lahore.", "ACCIDENT_IMAGE_REQUEST"),
    ("How many accidents occurred in Lahore?", "ACCIDENT_STATISTICS"),
    ("What documents are required for driving?", "TRAFFIC_RULE"),
    ("What should I do if traffic police stop me?", "TRAFFIC_RULE"),
    ("What does a red traffic signal mean?", "TRAFFIC_SIGN"),
    ("How can I report a traffic violation?", "TRAFFIC_RULE"),
    ("A van hit my car at an intersection. What do I do next?", "ACCIDENT_GUIDANCE"),
]


async def main():
    rows = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/api/v1", timeout=180) as client:
        health = (await client.get("/health")).json()
        assert health["rag_anything_ready"], health
        session = None
        for question, intent in CASES:
            start = time.monotonic()
            response = await client.post("/chat", json={"message": question, "top_k": 5})
            response.raise_for_status()
            data = response.json()
            assert data["intent"] == intent, (question, data["intent"])
            assert not re.search(r"[A-Za-z]:[\\/]|ACCIDENT PROVENANCE|Image Path:|Visual Analysis:|\"image_path\"", data["answer"]), data["answer"]
            if intent not in {"TRAFFIC_SIGN", "ACCIDENT_IMAGE_REQUEST"}:
                assert not data["images"], question
            if intent in {"ACCIDENT_GUIDANCE", "EMERGENCY_GUIDANCE"}:
                assert "1122" in data["answer"], data["answer"]
                assert not re.search(r"Rawalpindi|Chakwal|M.B. Din", data["answer"])
            if intent == "ACCIDENT_STATISTICS":
                assert "Lahore" in data["answer"] and "cumulative" in data["answer"].lower()
            if intent == "ACCIDENT_IMAGE_REQUEST":
                assert data["images"] and all(i["region"] == "Lahore" for i in data["images"])
                for image in data["images"]:
                    media = await client.get("http://127.0.0.1:8000" + image["url"])
                    assert media.status_code == 200
            if intent == "TRAFFIC_FINE":
                assert "2,000" in data["answer"] and "helmet" in data["answer"].lower()
            if question == CASES[0][0]:
                session = data["session_id"]
            cited = set(map(int, re.findall(r"\[(\d+)\]", data["answer"])))
            assert cited <= set(range(1, len(data["citations"]) + 1))
            rows.append({"question": question, "intent": intent, "seconds": round(time.monotonic()-start, 2), "model": data["model"], "citations": len(data["citations"]), "images": len(data["images"]), "answer": data["answer"], "passed": True})
            print(f"PASS {intent}: {question}", flush=True)
            # The existing free-tier provider allows 8K tokens/minute. Pace live
            # acceptance calls rather than accidentally benchmarking that limit.
            await asyncio.sleep(12)
        for question in ("What if someone is injured?", "Which number should I call?"):
            response = await client.post("/chat", json={"message": question, "session_id": session})
            response.raise_for_status()
            data = response.json()
            assert data["intent"] == "EMERGENCY_GUIDANCE" and "1122" in data["answer"]
            assert not data["images"]
            rows.append({"question": question, "intent": data["intent"], "answer": data["answer"], "passed": True})
            print(f"PASS follow-up: {question}", flush=True)
            await asyncio.sleep(12)
        for route in ("/fines", "/rules", "/signs", "/sources", "/accidents", "/accidents/images", "/accidents/statistics"):
            response = await client.get(route)
            assert response.status_code == 200, route
            print(f"PASS endpoint: {route}", flush=True)
    report = Path(__file__).resolve().parents[1] / "backend/storage/assistant-integration-results.json"
    report.write_text(json.dumps({"health": health, "cases": rows}, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} passing live cases to {report}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
