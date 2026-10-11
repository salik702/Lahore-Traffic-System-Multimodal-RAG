"""Live checks for general Groq chat, topic switches and retained traffic RAG."""
import asyncio
import json
import re
from pathlib import Path

import httpx


async def main():
    rows = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/api/v1", timeout=180) as client:
        health = (await client.get("/health")).json()
        assert health["groq_configured"], health

        async def ask(question, intent, session=None):
            response = await client.post("/chat", json={"message": question, "session_id": session})
            response.raise_for_status()
            data = response.json()
            assert data["intent"] == intent, (question, data["intent"])
            if intent == "GENERAL_KNOWLEDGE":
                assert data["model"], (question, data["answer"])
                assert not data["citations"] and not data["evidence"] and not data["images"]
                assert data["retrieval_engine"] == "Groq general knowledge"
                assert not data["query_understanding"]["traffic_related"]
            assert not re.search(r"ACCIDENT PROVENANCE|Image Path:|Visual Analysis:", data["answer"])
            rows.append({"question": question, "intent": intent, "model": data["model"], "answer": data["answer"], "citations": len(data["citations"]), "images": len(data["images"]), "passed": True})
            print(f"PASS {intent}: {question}", flush=True)
            # Respect the configured provider's existing free-tier quota.
            await asyncio.sleep(12)
            return data

        general = await ask("Explain photosynthesis in two sentences.", "GENERAL_KNOWLEDGE")
        followup = await ask("What role does sunlight play?", "GENERAL_KNOWLEDGE", general["session_id"])
        assert re.search(r"plant|photosynthesis|sugar|chlorophyll", followup["answer"], re.I)
        await ask("What is fine-tuning a machine learning model?", "GENERAL_KNOWLEDGE")
        await ask("How does a car engine work?", "GENERAL_KNOWLEDGE")
        await ask("What is network traffic?", "GENERAL_KNOWLEDGE")
        fine = await ask("What is the helmet fine in Lahore?", "TRAFFIC_FINE", general["session_id"])
        assert fine["citations"] and "2,000" in fine["answer"]
        resumed = await ask("What about photosynthesis?", "GENERAL_KNOWLEDGE", general["session_id"])
        assert "PKR" not in resumed["answer"]
        accident = await ask("What happens after a road accident?", "ACCIDENT_GUIDANCE")
        assert accident["citations"] and "1122" in accident["answer"] and not accident["images"]
        assert not re.search(r"Rawalpindi|Chakwal|M.B. Din", accident["answer"])
        photos = await ask("Show me accident images from Lahore.", "ACCIDENT_IMAGE_REQUEST")
        assert photos["images"] and all(i["region"] == "Lahore" for i in photos["images"])
        for image in photos["images"]:
            assert (await client.get("http://127.0.0.1:8000" + image["url"])).status_code == 200

    path = Path(__file__).resolve().parents[1] / "backend/storage/general-chat-integration-results.json"
    path.write_text(json.dumps({"health": health, "cases": rows}, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} passing live cases to {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
