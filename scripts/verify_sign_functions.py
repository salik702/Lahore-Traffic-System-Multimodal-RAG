"""Live functional sign matching, strict constraints and pipeline regressions."""
import asyncio
import json
import re
from pathlib import Path

import httpx


async def main():
    cases = [
        ("show me overspeeding sign", {"speed-limit-50"}),
        ("show me over-speeding sign", {"speed-limit-50"}),
        ("show me over speeding sign", {"speed-limit-50"}),
        ("show me speeding sign", {"speed-limit-50"}),
        ("show me no overspeeding sign", {"speed-limit-50"}),
        ("show me excessive speed sign", {"speed-limit-50"}),
        ("show me speed limit sign", {"speed-limit-50"}),
        ("show me speed restrictions sign", {"speed-limit-50"}),
        ("show me the sign that tells drivers not to drive too fast", {"speed-limit-50"}),
        ("show me speed limit 50 sign", {"speed-limit-50"}),
        ("show me overspeeding 50 sign", {"speed-limit-50"}),
        ("show me minimum speed limit sign", {"minimum-speed-30"}),
        ("show me lower speed limit 30 sign", {"minimum-speed-30"}),
        ("show me speed limit 60 sign", set()),
        ("show me overspeeding 60 sign", set()),
        ("show me maximum speed 30 sign", set()),
        ("show me minimum speed 50 sign", set()),
        ("show me no speed limit sign", set()),
        ("show me no right turn sign", set()),
        ("show me rightt turn sign", set()),
        ("show me lane closure sign", set()),
        ("show me teleportation sign", set()),
        ("show me a school sign", set()),
        ("show me petrol pump sign", {"petrol-station"}),
        ("show me a sign where I can buy fuel", {"petrol-station"}),
        ("show me a sign indicating where medical treatment is available", {"hospital"}),
        ("show me the sign where you can have a meal", {"restaurant"}),
        ("show me a sign indicating no parking is allowed", {"no-parking"}),
        ("show me traffic light sign", {"traffic-lights"}),
        ("show me honking prohibited sign", {"no-horn"}),
        ("show me railway crossing without barrier sign", {"level-crossing-without-gate"}),
        ("show me roundabout sign", {"compulsory-roundabout"}),
        ("show me a sign with circular arrows", {"compulsory-roundabout"}),
        ("show me road clousre sign", {"road-closed"}),
        ("show me narrow road sign", {"carriageway-narrows"}),
        ("show me left curve sign", {"left-bend"}),
        ("show me successive bends first right sign", {"double-bend-right"}),
    ]
    rows, sessions = [], {}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/api/v1", timeout=110) as client:
        health = (await client.get("/health")).json()
        assert health["rag_anything_ready"] and health["image_embeddings"]["indexed"], health
        for question, expected in cases:
            response = await client.post("/chat", json={"message": question})
            response.raise_for_status()
            data = response.json()
            assert data["intent"] == "TRAFFIC_SIGN", (question, data)
            assert {item["id"] for item in data["evidence"]} == expected, (question, data)
            assert bool(data["images"]) == bool(expected), (question, data)
            assert not re.search(r"[A-Za-z]:[\\/]|Visual Analysis|Image Path|PROVENANCE|semantic_similarity|image_similarity", data["answer"])
            assert all(set(e["metadata"]) <= {"page", "section", "category", "citation_number"} for e in data["evidence"])
            if expected & {"speed-limit-50", "minimum-speed-30"}:
                assert "catalogue example" in data["answer"].lower(), data
                assert "posted on your road" in data["answer"].lower(), data
            if expected == {"speed-limit-50"}:
                assert data["images"][0]["width"] == 512 and data["images"][0]["height"] == 512
                assert data["images"][0]["source_url"].endswith("CEEnglish.pdf#page=13"), data
                assert "minimum" not in data["answer"].lower()
            if expected == {"road-closed"}:
                assert "project illustration" in data["answer"].lower()
                assert not data["citations"][0]["official"]
            cited = set(map(int, re.findall(r"\[(\d+)\]", data["answer"])))
            assert cited == set(range(1, len(data["citations"]) + 1)), data
            for image in data["images"]:
                media = await client.get("http://127.0.0.1:8000" + image["url"])
                assert media.status_code == 200 and media.headers["content-type"].startswith("image/"), image
            catalogue = (await client.get("/signs", params={"q": question})).json()["items"]
            assert {item["id"] for item in catalogue} == expected, (question, catalogue)
            sessions[question] = data["session_id"]
            rows.append({"question": question, "answer": data["answer"], "images": data["images"], "evidence_ids": sorted(expected), "model": data["model"], "passed": True})
            print(f"PASS {question}", flush=True)

        # Real generation and memory checks are paced for the existing provider.
        followups = [
            ("What does it mean?", sessions["show me overspeeding sign"], {"speed-limit-50"}),
            ("What does it mean?", sessions["show me a sign where I can buy fuel"], {"petrol-station"}),
        ]
        for question, session, expected in followups:
            await asyncio.sleep(15)
            data = (await client.post("/chat", json={"message": question, "session_id": session})).json()
            assert {item["id"] for item in data["evidence"]} == expected, data
            assert data["images"] and data["citations"], data
            if "speed-limit-50" in expected:
                assert "example" in data["answer"].lower(), data
            rows.append({"question": question, "evidence_ids": sorted(expected), "answer": data["answer"], "model": data["model"], "passed": True})
            print(f"PASS follow-up {sorted(expected)}", flush=True)

        for question, intent in [("What happens after a road accident?", "ACCIDENT_GUIDANCE"),
                                 ("What does a red traffic signal mean?", "TRAFFIC_SIGN"),
                                 ("Show me accident images from Lahore", "ACCIDENT_IMAGE_REQUEST"),
                                 ("What is the helmet fine in Lahore?", "TRAFFIC_FINE"),
                                 ("Explain photosynthesis", "GENERAL_KNOWLEDGE")]:
            await asyncio.sleep(15)
            data = (await client.post("/chat", json={"message": question})).json()
            assert data["intent"] == intent, data
            if intent == "ACCIDENT_IMAGE_REQUEST":
                assert data["images"] and all(image["region"] == "Lahore" for image in data["images"]), data
            else:
                assert not data["images"], data
            if intent == "ACCIDENT_GUIDANCE":
                assert "1122" in data["answer"] and not re.search(r"Rawalpindi|Chakwal|PROVENANCE|Image Path", data["answer"]), data
            elif intent == "TRAFFIC_SIGN":
                assert "stop" in data["answer"].lower(), data
            elif intent == "TRAFFIC_FINE":
                assert "2,000" in data["answer"], data
            elif intent == "GENERAL_KNOWLEDGE":
                assert data["model"] and not data["evidence"] and not data["citations"], data
            rows.append({"question": question, "intent": intent, "answer": data["answer"], "model": data["model"], "passed": True})
            print(f"PASS regression {question}", flush=True)
    report = Path(__file__).resolve().parents[1] / "backend/storage/sign-function-integration-results.json"
    report.write_text(json.dumps({"health": health, "cases": rows}, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} passing live cases to {report.name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
