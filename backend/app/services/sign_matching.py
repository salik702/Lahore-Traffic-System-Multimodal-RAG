"""Catalogue identity matching shared by search and the final evidence boundary.

Sign names carry direction, prohibition and numbers that broad embedding similarity
cannot safely substitute. Generic request language is excluded from that identity.
"""
import re

REQUEST_WORDS = set("a an and are as at be can could do does for from how i in is it me of on or please the this that those these to what when where which with would you your show display find retrieve search give tell tells explain describe picture pictures image images photo photos photograph photographs sign signs traffic road roads symbol symbols meaning means mean indicate indicates called look looks like need want pakistan pakistani punjab lahore common different types type examples example about follow up more only drivers".split())
CATEGORIES = {"mandatory", "warning", "prohibitory", "regulatory", "information"}
PROTECTED_WORDS = {"left", "right", "no", "not", "with", "without", "minimum", "maximum", "end", "start"}
VISUAL_WORDS = {"circular", "circle", "round", "triangular", "triangle", "rectangular", "square", "arrow", "arrows", "blue", "red", "green", "yellow", "white", "black", "shaped", "shape", "pointing"}
DESCRIPTION_GLUE = {"available", "nearby", "ahead", "first", "starting", "getting", "facility", "services", "indicating"}
SEMANTIC_MIN_SCORE = 0.55
SEMANTIC_MIN_MARGIN = 0.08


def catalogue_vocabulary(signs: list[dict]) -> frozenset[str]:
    """Learn spelling candidates from actual names/aliases, never a guessed sign."""
    words = REQUEST_WORDS | CATEGORIES | PROTECTED_WORDS
    for sign in signs:
        text = " ".join([sign["id"], sign.get("name", sign.get("title", "")), sign.get("category", ""), *sign.get("aliases", [])])
        words |= set(re.findall(r"[a-z]+", text.lower())) | sign_tokens(text)
    return frozenset(words)


def one_edit_apart(word: str, candidate: str) -> bool:
    """Accept one insertion/deletion/substitution or adjacent letter swap."""
    if abs(len(word) - len(candidate)) > 1:
        return False
    if len(word) == len(candidate):
        changed = [i for i, (a, b) in enumerate(zip(word, candidate)) if a != b]
        if len(changed) == 1:
            return True
        return len(changed) == 2 and changed[1] == changed[0] + 1 and word[changed[0]] == candidate[changed[1]] and word[changed[1]] == candidate[changed[0]]
    longer, shorter = (word, candidate) if len(word) > len(candidate) else (candidate, word)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def correct_sign_query(query: str, vocabulary: frozenset[str]) -> tuple[str, dict[str, str]]:
    """Only correct unknown words with one unambiguous catalogue spelling.

    Valid vocabulary, short words, directions, negations and numbers are left
    intact. Ambiguous corrections remain unresolved rather than changing intent.
    """
    corrections = {}
    candidates = vocabulary - PROTECTED_WORDS

    def replace(match):
        word = match[0].lower()
        if len(word) < 5 or word in vocabulary or sign_tokens(word) <= vocabulary:
            return match[0]
        nearby = {candidate for candidate in candidates if one_edit_apart(word, candidate)}
        # Singular/plural spellings that map to the same identity are equivalent.
        identities = {frozenset(sign_tokens(candidate)) for candidate in nearby}
        if len(identities) == 1 and nearby:
            chosen = min(nearby, key=lambda candidate: (abs(len(candidate) - len(word)), candidate))
            corrections[word] = chosen
            return chosen
        return match[0]

    return re.sub(r"\b[a-zA-Z]+\b", replace, query), corrections


def sign_tokens(value: str) -> set[str]:
    q = value.lower()
    q = re.sub(r"\bround[ -]?abouts?\b", "roundabout", q)
    q = re.sub(r"\b(?:traffic circles?|circular junctions?)\b", "roundabout", q)
    q = re.sub(r"\bu[ -]?turns?\b", "uturn", q)
    q = re.sub(r"\b(?:prohibited|prohibition|forbidden|banned)\b", "no", q)
    q = re.sub(r"\b(?:curve|curves)\b", "bend", q)
    q = re.sub(r"\b(?:railway|railroad)\b", "level", q)
    q = re.sub(r"\bover[- ]?speed(?:ing)?\b", "overspeeding", q)
    aliases = {"bends": "bend", "cycles": "cycle", "lights": "light", "signals": "signal",
               "gates": "gate", "barrier": "gate", "barriers": "gate", "animals": "animal", "crossroads": "crossroad",
               "crossings": "crossing", "narrows": "narrow", "narrowing": "narrow",
               "narrower": "narrow", "limits": "limit", "restrictions": "restriction",
               "successive": "double", "series": "double",
               "ends": "end", "lower": "minimum", "horns": "horn",
               "pedestrians": "pedestrian", "informative": "information", "informatory": "information"}
    return {aliases.get(w, w) for w in re.findall(r"[a-z0-9]+", q)} - REQUEST_WORDS


def sign_subject(query: str) -> set[str]:
    return sign_tokens(query) - CATEGORIES


def visual_description(query: str) -> bool:
    """A visual description can use pixel similarity when no catalogue name fits."""
    return bool(re.search(r"\bsign\b.*\b(?:with|showing|depicting|containing|shaped)\b|\b(?:circular|triangular|rectangular|square|blue|red|green|yellow|white|black)\b.*\bsign\b", query, re.I))


def visual_identity_terms(query: str, known_terms: frozenset[str]) -> set[str]:
    return sign_subject(query) & (known_terms - REQUEST_WORDS - CATEGORIES - VISUAL_WORDS - DESCRIPTION_GLUE)


def visual_identity_compatible(query: str, item: dict, known_terms: frozenset[str] | None = None) -> bool:
    """Pixel retrieval cannot override explicit direction, number or prohibition."""
    subject = sign_subject(query)
    identity = sign_tokens((item.get("name") or item.get("title", "")) + " " + item.get("id", ""))
    constraints = (subject & {"left", "right", "no", "not", "without", "minimum", "maximum", "end"}) | {word for word in subject if word.isdigit()}
    if known_terms is not None:
        constraints |= visual_identity_terms(query, known_terms)
    constraints |= set(item.get("metadata", {}).get("visual_identity_terms", []))
    constraints |= set(item.get("metadata", {}).get("semantic_identity_terms", []))
    category = sign_tokens(item.get("metadata", {}).get("category", item.get("category", "")))
    aliases = item.get("aliases", item.get("metadata", {}).get("aliases", []))
    identities = [identity, *(sign_tokens(alias) for alias in aliases)]
    # An alias inherits the specimen's number, never a requested new limit.
    numbers = {word for word in identity if word.isdigit()}
    return (any(constraints <= candidate | numbers and ("no" in subject) == ("no" in candidate)
                and ("not" in subject) == ("not" in candidate) for candidate in identities)
            and (not subject & CATEGORIES or bool(subject & CATEGORIES & category)))


def matches_sign(query: str, item: dict) -> bool:
    meta = item.get("metadata", {})
    requested_categories = sign_tokens(query) & CATEGORIES
    category = sign_tokens(meta.get("category", item.get("category", ""))) & CATEGORIES
    if requested_categories and not requested_categories & category:
        return False
    subject = sign_subject(query)
    if not subject:
        return True
    title = item.get("name") or item.get("title", "")
    aliases = item.get("aliases", meta.get("aliases", []))
    identities = [sign_tokens(title + " " + item.get("id", "")) - CATEGORIES]
    identities.extend(sign_tokens(alias) - CATEGORIES for alias in aliases)
    numbers = {word for word in identities[0] if word.isdigit()}
    # Do not interchange turn/keep signs, positive/prohibited movements,
    # gated/ungated crossings, directions, or different speed limits.
    for identity in identities:
        if not subject <= identity | numbers:
            continue
        if ("no" in subject) != ("no" in identity):
            continue
        if any(word in identity and word not in subject for word in {"left", "right", "double", "wild", "end", "minimum", "bridge", "crossing", "cycle", "pedestrian", "motorcycle", "vehicle", "bus", "truck"}):
            continue
        if "gate" in subject and ("without" in subject) != ("without" in identity):
            continue
        return True
    return False
