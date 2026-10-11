"""Live sign retrieval, provenance and existing chat-route regression checks."""
import asyncio
import json
import re
from pathlib import Path

import httpx


async def main():
    rows = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/api/v1", timeout=100) as client:
        health = (await client.get("/health")).json()
        assert health["rag_anything_ready"], health
        session = None
        cases = [
            ("show me roundabout sign", {"compulsory-roundabout"}),
            ("What does it mean?", {"compulsory-roundabout"}),
            ("show me round-about signs", {"compulsory-roundabout"}),
            ("show me the traffic circle sign", {"compulsory-roundabout"}),
            ("show me a left turn sign", {"turn-left"}),
            ("show me the no entry sign", {"no-entry"}),
            ("show a left turn prohibited sign", {"no-left-turn"}),
            ("show a left curve sign", {"left-bend"}),
            ("show me the speed limit 60 sign", set()),
            ("show me the teleportation sign", set()),
            ("show me road clousre sign", {"road-closed"}),
            ("show me road closure sign", {"road-closed"}),
            ("show me road closuer sign", {"road-closed"}),
            ("show me road closed sign", {"road-closed"}),
            ("show me road closures sign", {"road-closed"}),
            ("show me rounabout sign", {"compulsory-roundabout"}),
            ("show me no parkng sign", {"no-parking"}),
            ("show me narrow brigde sign", {"narrow-bridge"}),
            ("show me pedestrain crossing sign", {"pedestrian-crossing"}),
        ]
        for index, (question, expected) in enumerate(cases):
            response = await client.post("/chat", json={"message": question, "session_id": session if index in {1, 5} else None})
            response.raise_for_status()
            data = response.json()
            assert data["intent"] == "TRAFFIC_SIGN", data
            assert {e["id"] for e in data["evidence"]} == expected, (question, data["evidence"])
            assert bool(data["images"]) == bool(expected), data
            assert not re.search(r"[A-Za-z]:[\\/]|Visual Analysis|Image Path|PROVENANCE", data["answer"])
            if index == 0:
                session = data["session_id"]
                assert "roundabout" in data["answer"].lower()
                assert data["images"][0]["width"] >= 512
                assert data["images"][0]["source_url"] == "https://dla.nhmp.gov.pk/CEEnglish.pdf#page=10"
            if expected == {"road-closed"}:
                assert "project illustration" in data["answer"].lower(), data
                assert not data["citations"][0]["official"], data
            for image in data["images"]:
                media = await client.get("http://127.0.0.1:8000" + image["url"])
                assert media.status_code == 200 and media.headers["content-type"].startswith("image/"), image
            numbers = set(map(int, re.findall(r"\[(\d+)\]", data["answer"])))
            assert numbers <= set(range(1, len(data["citations"]) + 1))
            rows.append({"question": question, "answer": data["answer"], "images": data["images"], "evidence_ids": sorted(expected), "model": data["model"], "passed": True})
            print(f"PASS {question}", flush=True)
        catalogue = (await client.get("/signs", params={"q": "roundabout sign"})).json()["items"]
        assert [s["id"] for s in catalogue] == ["compulsory-roundabout"]
        assert catalogue[0]["image_source_url"].endswith("CEEnglish.pdf#page=10")
        assert [s["id"] for s in (await client.get("/signs", params={"q": "clousre"})).json()["items"]] == ["road-closed"]
        # Every upgraded asset must remain accessible, including signs not used
        # in the original bug report. Never rely on a fabricated media URL.
        all_signs = (await client.get("/signs")).json()["items"]
        upgraded = [s for s in all_signs if "-nhmp-512.png" in (s.get("image_url") or "")]
        assert len(upgraded) == 29
        for sign in upgraded:
            response = await client.get("http://127.0.0.1:8000" + sign["image_url"])
            assert response.status_code == 200, sign["id"]
        for question, intent in [("Explain common traffic signs in Pakistan", "TRAFFIC_SIGN"),
                                 ("What does a red traffic signal mean?", "TRAFFIC_SIGN"),
                                 ("Show me accident photos from Lahore", "ACCIDENT_IMAGE_REQUEST"),
                                 ("Explain how photosynthesis works", "GENERAL_KNOWLEDGE")]:
            # Respect the existing Groq free-tier rate budget for generation.
            await asyncio.sleep(15)
            response = await client.post("/chat", json={"message": question})
            response.raise_for_status()
            data = response.json()
            assert data["intent"] == intent, data
            if question.startswith("Explain common"):
                assert data["images"] and data["citations"], data
                assert all(e["source_id"] != "project-illustration" for e in data["evidence"]), data
            elif "red traffic" in question:
                assert not data["images"] and "stop" in data["answer"].lower(), data
            elif intent == "ACCIDENT_IMAGE_REQUEST":
                assert data["images"] and all(i["region"] == "Lahore" for i in data["images"]), data
            else:
                assert data["model"] and not data["evidence"] and not data["citations"] and not data["images"], data
            rows.append({"question": question, "intent": intent, "answer": data["answer"], "model": data["model"], "passed": True})
            print(f"PASS regression: {question}", flush=True)
    report = Path(__file__).resolve().parents[1] / "backend/storage/sign-integration-results.json"
    report.write_text(json.dumps({"health": health, "cases": rows, "verified_sharp_media_urls": len(upgraded)}, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} passing live cases; {len(upgraded)} sharp media URLs passed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
