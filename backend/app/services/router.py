"""Task routing with explicit requests, local semantic prototypes and bounded memory."""
import asyncio
import logging
import re
from typing import Any, Callable

from app.core.config import Settings

logger = logging.getLogger(__name__)
INTENTS = {"TRAFFIC_FINE", "TRAFFIC_SIGN", "TRAFFIC_RULE", "ROAD_SAFETY", "ACCIDENT_INFORMATION", "ACCIDENT_GUIDANCE", "EMERGENCY_GUIDANCE", "ACCIDENT_STATISTICS", "ACCIDENT_IMAGE_REQUEST", "DOCUMENT_SEARCH", "OUT_OF_DOMAIN", "VISUAL_SEARCH", "GENERAL_CONVERSATION", "GENERAL_KNOWLEDGE", "SIMULATION_INTELLIGENCE"}
GENERAL_INTENTS = {"GENERAL_KNOWLEDGE", "GENERAL_CONVERSATION", "OUT_OF_DOMAIN"}
PROTOTYPES = {
    "ACCIDENT_GUIDANCE": "What should a driver do following a collision? Stop, help people and follow accident procedures.",
    "EMERGENCY_GUIDANCE": "Someone is hurt, bleeding or unconscious. Need an ambulance and immediate help after a collision.",
    "ACCIDENT_INFORMATION": "Find historical road crash reports, events and incident records in a district.",
    "ACCIDENT_STATISTICS": "Number of road accidents, district totals, statistics and reporting periods.",
    "ACCIDENT_IMAGE_REQUEST": "Show photographs and pictures of road crashes.",
    "TRAFFIC_FINE": "What is the monetary penalty or traffic challan for a driving violation?",
    "TRAFFIC_SIGN": "Explain the meaning of road signs, symbols and traffic lights.",
    "TRAFFIC_RULE": "Driving licence, vehicle documents, police stopping drivers and traffic laws.",
    "ROAD_SAFETY": "Advice for safe driving and preventing road injuries.",
    "GENERAL_KNOWLEDGE": "Explain how cars and machines work, science, history, mathematics, cooking, programming, or general knowledge.",
    "SIMULATION_INTELLIGENCE": "Explain why this intersection is congested, simulation queue lengths, AI controller performance, scenarios, and traffic control strategies.",
}


def normalize(message: str) -> str:
    value = message.lower().strip()
    aliases = {"bike": "motorcycle", "topi": "helmet", "challan": "fine", "gaari": "vehicle", "tasveer": "image", "dikhao": "show", "hadsa": "accident", "aaj": "today", "kitna": "how much"}
    for word, replacement in aliases.items():
        value = re.sub(rf"\b{word}\b", replacement, value)
    return value


def detect_intent(query: str, has_image: bool = False) -> str:
    if has_image:
        return "VISUAL_SEARCH"
    q = normalize(query)
    # Disambiguate task phrases, rather than interpreting each shared word as
    # a traffic request (fine-tuning, signing a document, software crashes).
    if re.search(r"\bfine[- ]?tun\w*|\bsign(?:ing)? (?:in|up|a (?:pdf|document)|the (?:pdf|document))\b|\b(?:digital|electronic) signatures?\b|\b(?:zodiac|star) signs?\b|\b(?:emergency fund|software crash|app crash|computer crash|network traffic)\b", q):
        return "GENERAL_KNOWLEDGE"
    if re.search(r"\b(?:write|compose|tell|create)\b.*\b(?:poem|joke|story)\b", q):
        return "GENERAL_KNOWLEDGE"
    if re.search(r"\b(?:python|javascript|typescript|programming|software|neural networks?|machine learning|deep learning|computer vision|yolo|photosynthesis|astronomy|algebra|cooking|recipes?|electronics)\b", q):
        return "GENERAL_KNOWLEDGE"
    accident = bool(re.search(r"\b(accidents?|crash\w*|collision\w*|injur\w*|hurt|bleeding|unconscious)\b", q))
    image_request = bool(re.search(r"\b(images?|photos?|photographs?|pictures?)\b", q))
    traffic_scene = bool(re.search(r"\b(roads?|vehicles?|cars?|buses|bus|trucks?|motorcycles?|wreckage|fog)\b", q))
    if image_request and (accident or traffic_scene) and not re.search(r"\b(signs?|symbols?|signals?)\b", q):
        return "ACCIDENT_IMAGE_REQUEST"
    if re.search(r"\b(injur\w*|hurt|bleeding|unconscious|ambulance|emergency|1122)\b", q) and not re.search(r"\b(how many|statistics|records?|reports?|dataset|number of)\b", q):
        return "EMERGENCY_GUIDANCE"
    if accident and re.search(r"\b(how many|statistics|totals?|count|number of|rate|trend)\b", q):
        return "ACCIDENT_STATISTICS"
    if accident:
        if re.search(r"\bhow (?:can|do|should)\b.*\breport\b", q):
            return "ACCIDENT_GUIDANCE"
        if re.search(r"\b(after|following|should|do|help|steps?|procedures?|happens?|handle|respond)\b", q) and not re.search(r"\b(show|find|list|records?|dataset|occurred|happened in)\b", q):
            return "ACCIDENT_GUIDANCE"
        return "ACCIDENT_INFORMATION"
    if re.search(r"\b(simulat\w*|digital twin|kalma|ai controller|fixed-time|reinforcement learning|scenario|bottlenecks?|queues?|throughput|waiting time|travel time|closing this road|closed road|longest queue|congested|congestion|traffic strategy|what-if|near-miss)\b", q):
        return "SIMULATION_INTELLIGENCE"
    if re.search(r"\b(fines?|penalt\w*|challan)\b", q):
        return "TRAFFIC_FINE"
    if re.search(r"\b(signs?|symbols?|signals?|red light|traffic light)\b", q):
        return "TRAFFIC_SIGN"
    if re.search(r"\b(sources?|ordinance|section|pdf)\b", q):
        return "DOCUMENT_SEARCH"
    if re.search(r"\b(hello|hi|thanks|thank you)\b", q) and len(q.split()) <= 4:
        return "GENERAL_CONVERSATION"
    if re.search(r"\b(safety|safe|pedestrian|helmet|seat ?belt)\b", q):
        return "ROAD_SAFETY"
    if re.search(r"\b(traffic|road|driving|drive|driver|vehicle|motorcycle|car|parking|license|licence|police)\b", q):
        return "TRAFFIC_RULE"
    return "OUT_OF_DOMAIN"


def resolve_followup(message: str, history: list[dict]) -> tuple[str, str | None]:
    """Carry the last topic only for elliptical questions, never all previous topics."""
    q = normalize(message)
    previous = next((item for item in reversed(history) if item.get("role") == "assistant" and item.get("metadata", {}).get("intent") in INTENTS), None)
    if not previous:
        return q, None
    meta = previous["metadata"]
    prior = meta.get("normalized_query", "").split(". Follow-up:")[0][:600]
    intent = meta.get("intent")
    explicit = detect_intent(q)
    if explicit == "GENERAL_KNOWLEDGE":
        return q, None
    if intent in GENERAL_INTENTS:
        # The generator sees bounded recent general-chat turns; an explicit
        # traffic question switches back to sourced retrieval immediately.
        return q, "GENERAL_KNOWLEDGE" if explicit in GENERAL_INTENTS else None
    if explicit in {"TRAFFIC_FINE", "TRAFFIC_SIGN", "ACCIDENT_STATISTICS", "ACCIDENT_IMAGE_REQUEST"} and explicit != intent:
        return q, None
    if intent in {"ACCIDENT_GUIDANCE", "EMERGENCY_GUIDANCE"}:
        if re.search(r"\b(which|what) (number|service)|who (do|should|can) i call|how (do|can) i call", q):
            return q + " for emergency help after a road accident", "EMERGENCY_GUIDANCE"
        if re.search(r"\b(injur\w*|hurt|bleeding|unconscious)\b", q) and len(q.split()) <= 12:
            return q + " after a road accident", "EMERGENCY_GUIDANCE"
    vehicle_followup = intent == "TRAFFIC_FINE" and bool(re.fullmatch(r"(?:and )?(?:what|how) about (?:a |the )?(?:cars?|motorcycles?|trucks?|buses|bus|rickshaws?)\??", q))
    reference = bool(re.search(r"\b(it|that|this|those|these|they|them)\b", q))
    elliptical = bool(re.fullmatch(r"(?:and )?(?:why|why is that|tell me more|can you explain|give me an example|what next|what else)[?.!]*", q))
    if len(q.split()) <= 10 and (vehicle_followup or (reference or elliptical) and explicit in {"OUT_OF_DOMAIN", intent}):
        if vehicle_followup:
            prior = re.sub(r"\b(motorcycle|car|truck|bus|rickshaw)s?\b", "vehicle", prior)
        if prior:
            return f"{prior}. Follow-up: {q}", intent
    return q, None


async def understand_query(message: str, settings: Settings, history: list[dict] | None = None, semantic_classifier: Callable | None = None) -> dict[str, Any]:
    normalized, inherited = resolve_followup(message, history or [])
    intent = inherited or detect_intent(normalized)
    broad_rule = intent == "TRAFFIC_RULE" and not re.search(r"\b(documents?|licen[cs]e|police|registration|permit|parking|report|violation)\b", normalized)
    if not inherited and (intent in {"OUT_OF_DOMAIN", "ACCIDENT_INFORMATION", "ROAD_SAFETY"} or broad_rule) and semantic_classifier:
        try:
            predicted, confidence, margin = await asyncio.to_thread(semantic_classifier, normalized)
            traffic_anchor = bool(re.search(r"\b(car|vehicle|van|truck|bumper|road|motorcycle|driver)\b", normalized))
            general_mechanics = broad_rule and predicted == "GENERAL_KNOWLEDGE" and confidence >= 0.30 and margin >= 0.12
            if confidence >= 0.48 and margin >= 0.04 or general_mechanics or traffic_anchor and predicted in {"ACCIDENT_GUIDANCE", "EMERGENCY_GUIDANCE"} and confidence >= 0.33 and margin >= 0.08:
                intent = predicted
                if predicted in {"ACCIDENT_GUIDANCE", "EMERGENCY_GUIDANCE"}:
                    focus = "post-collision accident safety and driver duties" if predicted == "ACCIDENT_GUIDANCE" else "injured person accident emergency medical help"
                    normalized = f"{normalized}. {focus}"
        except Exception:
            logger.warning("query_semantic_classifier_unavailable", exc_info=True)
    if intent == "OUT_OF_DOMAIN":
        intent = "GENERAL_KNOWLEDGE"
    indexed = bool(re.search(r"\b(indexed|index|dataset|corpus)\b", normalized))
    recent = bool(re.search(r"\b(today|current|live|now|latest|recent|yesterday)\b", normalized)) and not indexed
    image = intent in {"ACCIDENT_IMAGE_REQUEST", "TRAFFIC_SIGN", "VISUAL_SEARCH"}
    return {"original_query": message, "normalized_query": normalized, "constraint_query": normalized,
            "expanded_queries": [], "intent": intent, "traffic_related": intent not in GENERAL_INTENTS,
            "recency_required": recent, "indexed_requested": indexed,
            "contact_only": intent == "EMERGENCY_GUIDANCE" and bool(re.search(r"\b(?:which|what) (?:number|service)|who.*call", normalize(message))),
            "modalities": ["text", "image"] if image else ["text", "table"], "contextual_followup": bool(inherited)}

