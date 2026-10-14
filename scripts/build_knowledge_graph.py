"""Build the source-grounded Lahore Traffic Intelligence graph corpus.

The builder is deterministic apart from the generated timestamp. It makes no
network or LLM calls. Structured project data, extracted PDF text, persisted
RAG chunks, and the preserved LightRAG GraphML snapshot are ingested into a
temporary SQLite staging store, then exported to XML and visualizations.
"""
from __future__ import annotations

import argparse
import csv
import colorsys
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "graphs" / "knowledge_graph"
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:44] or "item"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "decimal"
    if isinstance(value, (dict, list, tuple)) or value is None:
        return "json"
    return "string"


def json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


MANIFEST_STATUSES = {
    "complete": "processed",
    "partial": "partially_processed",
    "failed": "failed",
}


def source_status(value: Any) -> str:
    """Map a run outcome onto the vocabulary the graph schema allows."""
    text = str(value or "").strip().casefold()
    return MANIFEST_STATUSES.get(text, "unprocessed")


def read_pdf_ingestion_manifest() -> dict[str, Any]:
    """Load the recorded PDF ingestion outcome, or an empty record if absent."""
    path = ROOT / "data/processed/metadata/rag_document_ingestion.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def xml_safe(value: str | None) -> str | None:
    if value is None:
        return None
    return "".join(ch if (ord(ch) in (0x9, 0xA, 0xD) or 0x20 <= ord(ch) <= 0xD7FF
                        or 0xE000 <= ord(ch) <= 0xFFFD or 0x10000 <= ord(ch) <= 0x10FFFF)
                   else "\uFFFD" for ch in value)


def serialize_element(element: ET.Element) -> bytes:
    for item in element.iter():
        item.text = xml_safe(item.text)
        item.tail = xml_safe(item.tail)
        attrs = {key: xml_safe(value) or "" for key, value in item.attrib.items()}
        item.attrib.clear()
        item.attrib.update(attrs)
    return ET.tostring(element, encoding="utf-8")


def stream_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Incrementally decode a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        pos = 0
        eof = False
        started = False
        done = False
        while not done:
            while pos >= len(buffer) and not eof:
                buffer = stream.read(chunk_size)
                pos = 0
                if not buffer:
                    eof = True
            if not started:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos == len(buffer):
                    continue
                if buffer[pos] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                started = True
                pos += 1
            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                    pos += 1
                if pos < len(buffer):
                    if buffer[pos] == "]":
                        done = True
                        break
                    try:
                        item, end = decoder.raw_decode(buffer, pos)
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        buffer = buffer[pos:] + stream.read(chunk_size)
                        pos = 0
                        if len(buffer) < chunk_size:
                            eof = True
                        continue
                    pos = end
                    yield item
                    if pos > chunk_size:
                        buffer = buffer[pos:]
                        pos = 0
                    break
                if eof:
                    raise ValueError(f"Unexpected end of JSON array in {path}")
                buffer = buffer[pos:] + stream.read(chunk_size)
                pos = 0
                if len(buffer) < chunk_size:
                    eof = True


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.executescript("""
        CREATE TABLE sources(
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
          path TEXT, external_id TEXT, media_type TEXT, sha256 TEXT, size_bytes INTEGER,
          discovered INTEGER, processed INTEGER, entity_ref TEXT NOT NULL,
          available INTEGER NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL,
          note TEXT NOT NULL, error TEXT
        );
        CREATE TABLE entities(
          id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
          description TEXT NOT NULL, attrs TEXT NOT NULL, aliases TEXT NOT NULL,
          provenance TEXT NOT NULL, multimodal TEXT NOT NULL, canonical INTEGER NOT NULL,
          dedup_key TEXT UNIQUE
        );
        CREATE TABLE relationships(
          id TEXT PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL,
          type TEXT NOT NULL, label TEXT NOT NULL, direction TEXT NOT NULL,
          description TEXT NOT NULL, attrs TEXT NOT NULL, provenance TEXT NOT NULL,
          dedup_key TEXT UNIQUE
        );
        CREATE TABLE conflicts(
          id TEXT PRIMARY KEY, subject TEXT NOT NULL, field TEXT NOT NULL,
          status TEXT NOT NULL, description TEXT NOT NULL, assertions TEXT NOT NULL
        );
        CREATE INDEX entity_type_idx ON entities(type);
        CREATE INDEX relation_type_idx ON relationships(type);
        CREATE INDEX relation_source_idx ON relationships(source);
        CREATE INDEX relation_target_idx ON relationships(target);
        """)
        self.n = 0
        self.duplicate_entities_merged = 0

    def add_entity(self, *, entity_id: str, name: str, entity_type: str,
                   description: str = "", attrs: dict[str, Any] | None = None,
                   aliases: Iterable[str] = (), provenance: Iterable[dict[str, Any]] = (),
                   multimodal: Iterable[dict[str, Any]] = (), canonical: bool = True,
                   dedup_key: str | None = None) -> str:
        attrs = attrs or {}
        prov = list(provenance)
        alias_list = sorted({str(x) for x in aliases if x is not None and str(x).strip()})
        mm = list(multimodal)
        old = self.db.execute("SELECT aliases,provenance,attrs,multimodal,description FROM entities WHERE id=?", (entity_id,)).fetchone()
        if old:
            self.duplicate_entities_merged += 1
            old_aliases, old_prov, old_attrs, old_mm, old_desc = old
            alias_list = sorted(set(json.loads(old_aliases)) | set(alias_list))
            prov = merge_provenance(json.loads(old_prov), prov)
            old_attrs = json.loads(old_attrs)
            for key, value in attrs.items():
                old_attrs.setdefault(key, value)
            mm = merge_objects(json.loads(old_mm), mm)
            description = description or old_desc
            self.db.execute("UPDATE entities SET aliases=?,provenance=?,attrs=?,multimodal=?,description=? WHERE id=?",
                            (dump(alias_list), dump(prov), dump(old_attrs), dump(mm), description, entity_id))
            return entity_id
        self.db.execute("""INSERT OR IGNORE INTO entities
            (id,name,type,description,attrs,aliases,provenance,multimodal,canonical,dedup_key)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (entity_id, name or entity_id, entity_type, description, dump(attrs), dump(alias_list),
             dump(prov), dump(mm), int(canonical), dedup_key))
        self.n += 1
        if self.n % 5000 == 0:
            self.db.commit()
        return entity_id

    def add_relation(self, source: str, target: str, rel_type: str, label: str,
                     description: str = "", attrs: dict[str, Any] | None = None,
                     provenance: Iterable[dict[str, Any]] = (), direction: str = "directed",
                     identity: str | None = None, relationship_id: str | None = None) -> str:
        # All callers use already-created deterministic IDs. The complete
        # endpoint check runs after export; avoid two SQL reads per graph edge.
        key = identity or dump([source, target, rel_type, label, description, attrs or {}])
        rid = relationship_id or stable_id("r", key)
        self.db.execute("""INSERT OR IGNORE INTO relationships
          (id,source,target,type,label,direction,description,attrs,provenance,dedup_key)
          VALUES(?,?,?,?,?,?,?,?,?,?)""",
          (rid, source, target, rel_type, label, direction, description,
           dump(attrs or {}), dump(merge_provenance([], list(provenance))), key))
        return rid

    def add_conflict(self, subject: str, field: str, description: str,
                     assertions: list[dict[str, str]], status: str = "unresolved") -> None:
        if len(assertions) < 2:
            return
        cid = stable_id("c", dump([subject, field, assertions]))
        self.db.execute("INSERT OR IGNORE INTO conflicts VALUES(?,?,?,?,?,?)",
                        (cid, subject, field, status, description, dump(assertions)))

    def close(self):
        if self.db is None:
            return
        self.db.commit()
        self.db.close()
        self.db = None


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_provenance(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {dump(item): item for item in left}
    for item in right:
        keyed[dump(item)] = item
    return [keyed[key] for key in sorted(keyed)]


def merge_objects(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {dump(item): item for item in left}
    for item in right:
        keyed[dump(item)] = item
    return [keyed[key] for key in sorted(keyed)]


class CorpusBuilder:
    def __init__(self, store: Store):
        self.s = store
        self.file_sources: dict[str, str] = {}
        self.external_sources: dict[str, str] = {}
        self.external_url_sources: dict[str, str] = {}
        self.input_totals: Counter[str] = Counter()
        self.duplicate_candidates = 0
        self.unresolved_relationships = 0
        self.process_failures: list[dict[str, Any]] = []
        self.legacy_entity_ids: dict[str, str] = {}
        self.rag_chunk_ids: set[str] = set()
        self.missing_legacy_entity_chunks = 0
        self.missing_legacy_relationship_chunks = 0
        self.legacy_artifact_inventory: dict[str, Any] = {}

    def source(self, *, key: str, title: str, kind: str, status: str,
               path: str | None = None, external_id: str | None = None,
               media_type: str | None = None, sha256: str | None = None,
               size: int | None = None, discovered: int | None = None,
               processed: int | None = None, available: bool = True,
               description: str = "", note: str = "", error: str | None = None) -> str:
        sid = stable_id("src", key)
        eid = stable_id("e-source", sid)
        self.s.db.execute("""INSERT OR REPLACE INTO sources
          (id,kind,status,path,external_id,media_type,sha256,size_bytes,discovered,processed,
           entity_ref,available,title,description,note,error)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (sid, kind, status, path, external_id, media_type, sha256, size,
           discovered, processed, eid, int(available), title, description, note, error))
        self.s.add_entity(entity_id=eid, name=title, entity_type="Source",
                          description=description or note,
                          attrs={"source_catalog_id": sid, "source_kind": kind,
                                 "processing_status": status, "path": path,
                                 "external_id": external_id, "media_type": media_type,
                                 "sha256": sha256, "size_bytes": size,
                                 "items_discovered": discovered, "items_processed": processed},
                          canonical=False, dedup_key=f"source:{sid}")
        return sid

    def add_file_source(self, relpath: str, *, title: str | None = None, kind: str = "project_file",
                        status: str = "reviewed", discovered: int | None = None,
                        processed: int | None = None, note: str = "") -> str:
        relpath = relpath.replace("\\", "/")
        if relpath in self.file_sources:
            return self.file_sources[relpath]
        path = ROOT / relpath
        exists = path.is_file()
        sid = self.source(key="file:" + relpath, title=title or Path(relpath).name,
                          kind=kind, status=status if exists else "unavailable",
                          path=relpath, media_type=media_type(path) if exists else None,
                          sha256=sha256_file(path) if exists else None,
                          size=path.stat().st_size if exists else None,
                          discovered=discovered, processed=processed,
                          available=exists, description=f"Project source file: {relpath}",
                          note=note or ("File present and catalogued." if exists else "Referenced file is not present."))
        self.file_sources[relpath] = sid
        return sid

    def add_external_sources(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            original = str(row.get("id") or "")
            local = row.get("local_file_path")
            p = ROOT / local if local else None
            exists = bool(p and (p.is_file() or p.is_dir()))
            is_file = bool(p and p.is_file())
            sid = self.source(key="external:" + original,
                              title=row.get("title") or original,
                              kind="external_reference", status="reviewed" if exists else "unavailable",
                              path=str(local).replace("\\", "/") if local else None,
                              external_id=original, media_type=media_type(p) if is_file else None,
                              sha256=sha256_file(p) if is_file else None,
                              size=p.stat().st_size if is_file else None,
                              discovered=1, processed=1 if exists else 0, available=exists,
                              description=row.get("description") or "",
                              note=("Local snapshot or source directory is present; source-catalog entry reviewed." if exists
                                    else "External catalog entry has no available local snapshot."))
            self.external_sources[original] = sid
            for url in (row.get("original_url"), row.get("download_url")):
                if url:
                    self.external_url_sources[str(url).rstrip("/")] = sid

    def ref(self, source_ref: str, *, source_id: str | None = None,
            location: str | None = None, method: str | None = None,
            state: str | None = "processed", chunk_ref: str | None = None) -> dict[str, Any]:
        return {"sourceRef": source_ref, "sourceId": source_id, "location": location,
                "method": method, "state": state, "chunkRef": chunk_ref}

    def resolve_external(self, original: Any) -> str | None:
        return self.external_sources.get(str(original)) if original is not None else None

    def resolve_external_url(self, url: Any) -> str | None:
        return self.external_url_sources.get(str(url).rstrip("/")) if url else None

    def add_source_link(self, subject: str, sid: str, label: str = "SUPPORTED_BY_SOURCE") -> None:
        row = self.s.db.execute("SELECT entity_ref FROM sources WHERE id=?", (sid,)).fetchone()
        if row:
            self.s.add_relation(subject, row[0], label, label.replace("_", " ").title(),
                                description="The entity provenance identifies this source.",
                                provenance=[self.ref(sid, method="entity provenance")],
                                identity=f"source-link:{subject}:{sid}:{label}")

    def vehicle_category(self, name: str, source_ref: str, source_id: str | None,
                         method: str) -> str:
        key = normalize(name)
        eid = stable_id("vehicle-category", key)
        self.s.add_entity(entity_id=eid, name=name.replace("_", " "),
                          entity_type="VehicleCategory",
                          description="Vehicle category label used in project traffic data.",
                          attrs={"normalized_label": key},
                          provenance=[self.ref(source_ref, source_id=source_id, method=method)],
                          dedup_key="vehicle-category:" + key)
        return eid

    def build_sources_catalog(self) -> list[dict[str, Any]]:
        path = ROOT / "data/sources/sources.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        file_sid = self.add_file_source("data/sources/sources.json", kind="source_catalog",
                                        status="processed", discovered=len(rows), processed=len(rows),
                                        note="All source catalog entries parsed.")
        self.add_external_sources(rows)
        self.input_totals["source_catalog_entries"] = len(rows)
        return rows

    def structured_data(self, sources_rows: list[dict[str, Any]]) -> None:
        # Rules and legal citations.
        p = ROOT / "data/raw/structured/rules.json"
        rows = json.loads(p.read_text(encoding="utf-8"))
        rules_sid = self.add_file_source("data/raw/structured/rules.json", kind="traffic_rules",
                                         status="processed", discovered=len(rows), processed=len(rows),
                                         note="All structured rule rows represented as graph entities.")
        for row in rows:
            original = str(row.get("id") or "")
            eid = stable_id("rule", original)
            ext_sid = self.resolve_external(row.get("source_id"))
            prov = [self.ref(rules_sid, source_id=original, method="structured JSON row", location=original)]
            if ext_sid:
                prov.append(self.ref(ext_sid, source_id=str(row.get("source_id")),
                                     method="source catalog citation",
                                     location=f"{row.get('section', '')}, page {row.get('page', '')}".strip(", ")))
            attrs = {k: v for k, v in row.items() if k not in {"title", "summary"}}
            self.s.add_entity(entity_id=eid, name=row.get("title", original), entity_type="TrafficRule",
                              description=row.get("summary", ""), attrs=attrs,
                              aliases=[row.get("id", "")], provenance=prov)
            section = row.get("section")
            if section:
                provision_id = stable_id("legal-provision", f"{row.get('source_id')}:{section}")
                self.s.add_entity(entity_id=provision_id, name=f"{row.get('source_id')} {section}",
                                  entity_type="LegalProvision",
                                  description="Legal provision citation recorded in the curated traffic rule.",
                                  attrs={"source_id": row.get("source_id"), "section": section,
                                         "page": row.get("page"), "region": row.get("region")},
                                  provenance=prov, dedup_key=f"legal-provision:{row.get('source_id')}:{section}")
                self.s.add_relation(eid, provision_id, "GOVERNED_BY", "GOVERNED BY",
                                    description="The rule row cites this legal section.", provenance=prov)
            self.add_source_link(eid, rules_sid)
        self.input_totals["traffic_rules"] = len(rows)

        # Fine schedule: preserve fine matrix values as penalty entities by vehicle class.
        p = ROOT / "data/raw/structured/fines.json"
        fine_rows = json.loads(p.read_text(encoding="utf-8"))
        fine_sid = self.add_file_source("data/raw/structured/fines.json", kind="fine_schedule",
                                        status="processed", discovered=len(fine_rows), processed=len(fine_rows),
                                        note="All schedule rows and non-null vehicle-class amounts represented.")
        violations: dict[int, str] = {}
        for row in fine_rows:
            code = row.get("code")
            key = f"schedule:{code}"
            eid = stable_id("violation", key)
            violations[int(code)] = eid
            attrs = {k: v for k, v in row.items() if k not in {"violation", "keywords"}}
            self.s.add_entity(entity_id=eid, name=row.get("violation", f"Schedule item {code}"),
                              entity_type="TrafficViolation", description=row.get("violation", ""),
                              attrs=attrs, aliases=row.get("keywords", []),
                              provenance=[self.ref(fine_sid, source_id=str(code), method="structured JSON row",
                                                   location=f"Twelfth Schedule item {code}")])
            self.add_source_link(eid, fine_sid)
            for vehicle, amount in row.items():
                if vehicle in {"code", "violation", "keywords"} or amount is None:
                    continue
                vehicle_id = self.vehicle_category(vehicle, fine_sid, str(code), "fine schedule class")
                self.s.add_relation(eid, vehicle_id, "APPLIES_TO", "APPLIES TO VEHICLE CLASS",
                                    description="The schedule contains a separate fine amount for this class.",
                                    provenance=[self.ref(fine_sid, source_id=str(code),
                                                         method="fine schedule row", location=f"{vehicle} amount")])
                penalty_id = stable_id("penalty", f"{code}:{vehicle}")
                self.s.add_entity(entity_id=penalty_id, name=f"Fine for {row.get('violation')} — {vehicle.replace('_', ' ')}",
                                  entity_type="Penalty", description="Fine amount stated in the local fine schedule.",
                                  attrs={"amount": amount, "currency": "PKR", "vehicle_class": vehicle,
                                         "schedule_code": code, "violation": row.get("violation")},
                                  provenance=[self.ref(fine_sid, source_id=str(code), method="structured JSON row",
                                                       location=f"{vehicle} amount")])
                self.s.add_relation(eid, penalty_id, "HAS_PENALTY", "HAS PENALTY",
                                    description=f"Amount: {amount} PKR for schedule vehicle class {vehicle}.",
                                    provenance=[self.ref(fine_sid, source_id=str(code), method="fine schedule row",
                                                         location=vehicle)])
                self.s.add_relation(penalty_id, vehicle_id, "APPLIES_TO", "APPLIES TO VEHICLE CLASS",
                                    provenance=[self.ref(fine_sid, source_id=str(code), method="fine schedule row",
                                                         location=vehicle)])
        self.input_totals["fine_schedule_rows"] = len(fine_rows)

        # Rich verified policy view, linked to schedule items by explicit code.
        v = json.loads((ROOT / "data/raw/structured/vision_fines.json").read_text(encoding="utf-8"))
        vrows = v.get("records", [])
        v_sid = self.add_file_source("data/raw/structured/vision_fines.json", kind="verified_fine_policy",
                                     status="processed", discovered=len(vrows), processed=len(vrows),
                                     note="All policy rows and their source verification fields represented.")
        for row in vrows:
            code = int(row.get("schedule_code"))
            eid = stable_id("verified-violation", str(row.get("violation_id") or code))
            self.s.add_entity(entity_id=eid, name=row.get("violation_name", ""),
                              entity_type="VerifiedTrafficViolation",
                              description=row.get("description", ""),
                              attrs={k: value for k, value in row.items() if k not in {"violation_name", "description"}},
                              provenance=[self.ref(v_sid, source_id=str(row.get("violation_id")),
                                                   method="verified policy JSON row", location=f"schedule item {code}")]
                              + ([self.ref(self.resolve_external_url(row.get("source_url")),
                                           source_id=str(row.get("source_url")), method="source_url field")]
                                 if self.resolve_external_url(row.get("source_url")) else []))
            self.add_source_link(eid, v_sid)
            if code in violations:
                self.s.add_relation(eid, violations[code], "CORRESPONDS_TO_SCHEDULE_ITEM", "CORRESPONDS TO SCHEDULE ITEM",
                                    description="The policy record explicitly carries the same schedule code.",
                                    provenance=[self.ref(v_sid, source_id=str(code), method="schedule_code field")])
            for vehicle, amount in (row.get("amounts") or {}).items():
                if amount is None:
                    continue
                penalty_id = stable_id("verified-penalty", f"{code}:{vehicle}")
                self.s.add_entity(entity_id=penalty_id, name=f"Verified fine for {row.get('violation_name')} — {vehicle.replace('_',' ')}",
                                  entity_type="VerifiedPenalty", description="Fine amount in the verified policy record.",
                                  attrs={"amount": amount, "currency": row.get("currency"),
                                         "vehicle_class": vehicle, "schedule_code": code,
                                         "effective_date": row.get("effective_date"),
                                         "last_verified_on": row.get("last_verified_on")},
                                  provenance=[self.ref(v_sid, source_id=str(row.get("violation_id")),
                                                       method="verified policy amount", location=vehicle)])
                self.s.add_relation(eid, penalty_id, "HAS_VERIFIED_PENALTY", "HAS VERIFIED PENALTY",
                                    provenance=[self.ref(v_sid, source_id=str(row.get("violation_id")), location=vehicle)])
            legal_ref = row.get("legal_reference")
            if legal_ref:
                provision_id = stable_id("legal-provision", f"policy:{legal_ref}")
                self.s.add_entity(entity_id=provision_id, name=str(legal_ref), entity_type="LegalProvision",
                                  description="Legal citation stated in the verified fine policy record.",
                                  attrs={"citation": legal_ref, "jurisdiction": row.get("jurisdiction")},
                                  provenance=[self.ref(v_sid, source_id=str(row.get("violation_id")),
                                                       method="legal_reference field")],
                                  dedup_key="legal-provision-policy:" + normalize(legal_ref))
                self.s.add_relation(eid, provision_id, "GOVERNED_BY", "GOVERNED BY",
                                    description="The verified policy row states this legal reference.",
                                    provenance=[self.ref(v_sid, source_id=str(row.get("violation_id")),
                                                         method="legal_reference field")]
                                    + ([self.ref(self.resolve_external_url(row.get("source_url")),
                                                 source_id=str(row.get("source_url")), method="source_url field")]
                                       if self.resolve_external_url(row.get("source_url")) else []))
        self.input_totals["verified_fine_policy_rows"] = len(vrows)
        occupancy = v.get("occupancy_rule") or {}
        if occupancy:
            rule_id = stable_id("rule", "rule-pillion")
            existing = self.s.db.execute("SELECT 1 FROM entities WHERE id=?", (rule_id,)).fetchone()
            occupancy_prov = [self.ref(v_sid, source_id="occupancy_rule", method="structured policy field",
                                       location=f"{occupancy.get('legal_reference')} page {occupancy.get('pdf_page')}")]
            pdf_source = self.resolve_external("punjab-mvo-1965")
            if pdf_source:
                occupancy_prov.append(self.ref(pdf_source, source_id="punjab-mvo-1965",
                                               method="source catalog and recorded PDF page",
                                               location=f"page {occupancy.get('pdf_page')}"))
            rule_attrs = {"occupancy_rule": {k: value for k, value in occupancy.items() if k != "source_text"},
                          "source_text": occupancy.get("source_text")}
            if existing:
                self.s.add_entity(entity_id=rule_id, name="Only one pillion passenger",
                                  entity_type="TrafficRule",
                                  description="", attrs=rule_attrs,
                                  provenance=occupancy_prov)
            provision_id = stable_id("legal-provision", f"punjab-mvo-1965:{occupancy.get('legal_reference')}")
            self.s.add_entity(entity_id=provision_id, name=f"Punjab Motor Vehicles Ordinance 1965 {occupancy.get('legal_reference')}",
                              entity_type="LegalProvision",
                              description="Legal section citation recorded by the occupancy policy field.",
                              attrs={"source_id": "punjab-mvo-1965", "section": occupancy.get("legal_reference"),
                                     "page": occupancy.get("pdf_page")},
                              provenance=occupancy_prov,
                              dedup_key=f"legal-provision:punjab-mvo-1965:{occupancy.get('legal_reference')}")
            if existing:
                self.s.add_relation(rule_id, provision_id, "GOVERNED_BY", "GOVERNED BY",
                                    description="The rule and occupancy-policy record cite the same legal section.",
                                    provenance=occupancy_prov)

        # Compare the two project representations without choosing silently.
        schedule_by_code = {int(row["code"]): row for row in fine_rows}
        for row in vrows:
            code = int(row["schedule_code"])
            legacy = schedule_by_code.get(code)
            if not legacy:
                continue
            for vehicle, amount in (row.get("amounts") or {}).items():
                other = legacy.get(vehicle)
                if amount is not None and other is not None and amount != other:
                    subject = violations[code]
                    self.s.add_conflict(subject, f"fine_amount.{vehicle}",
                                        f"Two local representations disagree for schedule item {code} and vehicle class {vehicle}.",
                                        [{"sourceRef": fine_sid, "value": str(other)},
                                         {"sourceRef": v_sid, "value": str(amount)}])

        # Traffic signs and source-provided image links.
        signs_path = ROOT / "data/raw/structured/signs.json"
        signs = json.loads(signs_path.read_text(encoding="utf-8"))
        signs_sid = self.add_file_source("data/raw/structured/signs.json", kind="traffic_sign_catalog",
                                         status="processed", discovered=len(signs), processed=len(signs),
                                         note="All traffic sign rows represented; local image availability recorded per asset.")
        sign_assets_path = ROOT / "data/processed/metadata/sign_assets.json"
        sign_asset_data = json.loads(sign_assets_path.read_text(encoding="utf-8"))
        assets_sid = self.add_file_source("data/processed/metadata/sign_assets.json", kind="extracted_sign_assets",
                                          status="processed", discovered=len(sign_asset_data.get("assets", {})),
                                          processed=len(sign_asset_data.get("assets", {})),
                                          note="Extracted PDF sign assets and page metadata represented.")
        assets = sign_asset_data.get("assets", {})
        asset_entities: dict[str, str] = {}
        for asset_name, info in assets.items():
            path_val = info.get("path") or info.get("file") or info.get("image_path") or info.get("image") or asset_name
            asset_path = find_asset_path(str(path_val), asset_name)
            asset_source = self.add_file_source(asset_path, title=str(asset_name), kind="traffic_sign_image",
                                                status="processed" if asset_path and (ROOT / asset_path).is_file() else "unavailable",
                                                discovered=1, processed=1 if asset_path and (ROOT / asset_path).is_file() else 0,
                                                note="Image asset listed in sign asset manifest; no visual classification was inferred.") if asset_path else assets_sid
            image_id = stable_id("image-asset", asset_path or str(asset_name))
            asset_prov = [self.ref(assets_sid, source_id=str(asset_name), method="sign asset manifest",
                                   location=str(info.get("page") or ""))]
            asset_external = self.resolve_external(info.get("source_id"))
            if asset_external:
                asset_prov.append(self.ref(asset_external, source_id=str(info.get("source_id")),
                                           method="source_id field", location=f"page {info.get('page') or ''}".strip()))
            self.s.add_entity(entity_id=image_id, name=str(asset_name), entity_type="ImageAsset",
                              description="Traffic sign image extracted or catalogued by this project.",
                              attrs={k: value for k, value in info.items() if k not in {"path", "file"}},
                              provenance=asset_prov,
                              multimodal=[{"sourceRef": asset_source, "path": asset_path or str(path_val),
                                           "modality": "image", "page": info.get("page"),
                                           "caption": info.get("caption"), "license": info.get("license")}])
            asset_entities[normalize(Path(str(asset_name)).stem)] = image_id
        self.process_sign_image_files()
        for row in signs:
            original = str(row.get("id") or row.get("name") or "")
            eid = stable_id("sign", original)
            image_name = row.get("image")
            mm = []
            if image_name:
                candidate = find_asset_path(str(image_name), str(image_name))
                if candidate:
                    image_sid = self.add_file_source(candidate, title=Path(candidate).name,
                                                     kind="traffic_sign_image",
                                                     status="processed" if (ROOT / candidate).is_file() else "unavailable",
                                                     discovered=1, processed=1 if (ROOT / candidate).is_file() else 0,
                                                     note="Image file referenced by the traffic sign catalog.")
                    mm.append({"sourceRef": image_sid, "path": candidate, "modality": "image"})
            self.s.add_entity(entity_id=eid, name=row.get("name", original), entity_type="TrafficSign",
                              description=row.get("meaning", ""),
                              attrs={k: value for k, value in row.items() if k not in {"id", "name", "meaning", "aliases"}},
                              aliases=row.get("aliases", []),
                              provenance=[self.ref(signs_sid, source_id=original, method="structured sign row")],
                              multimodal=mm)
            if image_name:
                candidate = find_asset_path(str(image_name), str(image_name))
                if candidate:
                    image_id = stable_id("image-asset", candidate)
                    if self.s.db.execute("SELECT 1 FROM entities WHERE id=?", (image_id,)).fetchone():
                        asset_sid = self.file_sources.get(candidate)
                        relation_sources = [self.ref(signs_sid, source_id=original,
                                                     method="image field in sign catalog")]
                        if asset_sid:
                            relation_sources.append(self.ref(asset_sid, source_id=Path(candidate).name,
                                                             method="referenced image file"))
                        self.s.add_relation(eid, image_id, "ASSOCIATED_WITH_IMAGE", "ASSOCIATED WITH IMAGE",
                                            description="The traffic sign row explicitly names this image asset.",
                                            provenance=relation_sources)
            self.add_source_link(eid, signs_sid)
            self.input_totals["traffic_sign_rows"] += 1
        self.input_totals["sign_asset_records"] = len(assets)

    def process_sign_image_files(self) -> None:
        files: list[Path] = []
        for relative in ["data/raw/traffic_signs", "data/raw/images/traffic_signs"]:
            parent = ROOT / relative
            if parent.is_dir():
                files.extend(p for p in parent.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".svg"})
        files = sorted(set(files))
        seen_hashes: set[str] = set()
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            digest = sha256_file(path)
            duplicate = digest in seen_hashes
            seen_hashes.add(digest)
            sid = self.add_file_source(rel, title=path.name, kind="traffic_sign_image",
                                       status="duplicate" if duplicate else "processed",
                                       discovered=1, processed=1,
                                       note="Local traffic-sign image file catalogued; no OCR or visual fact extraction was applied.")
            eid = stable_id("image-asset", rel)
            self.s.add_entity(entity_id=eid, name=path.name, entity_type="ImageAsset",
                              description="Traffic sign image present in the project; visual meaning is not inferred from pixels.",
                              attrs={"path": rel, "sha256": digest, "byte_size": path.stat().st_size},
                              aliases=[path.stem],
                              provenance=[self.ref(sid, source_id=path.name, method="local image file inventory")],
                              multimodal=[{"sourceRef": sid, "path": rel, "modality": "image"}],
                              dedup_key="image-file:" + rel)
        self.input_totals["traffic_sign_image_files"] = len(files)

        # Road safety guidance.
        path = ROOT / "data/raw/structured/guidance.json"
        guidance = json.loads(path.read_text(encoding="utf-8"))
        guide_sid = self.add_file_source("data/raw/structured/guidance.json", kind="road_safety_guidance",
                                         status="processed", discovered=len(guidance), processed=len(guidance),
                                         note="All structured guidance rows represented.")
        for row in guidance:
            original = str(row.get("id") or "")
            eid = stable_id("guidance", original)
            self.s.add_entity(entity_id=eid, name=row.get("title", original), entity_type="RoadSafetyGuideline",
                              description=row.get("summary", ""),
                              attrs={k: value for k, value in row.items() if k not in {"id", "title", "summary", "keywords"}},
                              aliases=row.get("keywords", []),
                              provenance=[self.ref(guide_sid, source_id=original, method="structured JSON row")])
            self.add_source_link(eid, guide_sid)
        self.input_totals["guidance_rows"] = len(guidance)

        # District aggregate records, preserving values as reported.
        p = ROOT / "data/processed/accidents/statistics/rescue1122_districts.json"
        stats = json.loads(p.read_text(encoding="utf-8"))
        stats_sid = self.add_file_source("data/processed/accidents/statistics/rescue1122_districts.json",
                                        kind="district_aggregate_statistics", status="processed",
                                        discovered=len(stats), processed=len(stats),
                                        note="District-level aggregate records represented without converting them to individual incidents.")
        for i, row in enumerate(stats):
            label = row.get("district") or f"District row {i+1}"
            eid = stable_id("district-stat", f"{row.get('type')}:{label}:{i}")
            stat_prov = [self.ref(stats_sid, source_id=str(i), method="JSON record")]
            external = self.resolve_external_url(row.get("source_url"))
            if external:
                stat_prov.append(self.ref(external, source_id=str(row.get("source_url")), method="source_url field"))
            self.s.add_entity(entity_id=eid, name=f"Rescue 1122 aggregate — {label}",
                              entity_type="DistrictStatistic",
                              description="District aggregate values as stored in the project dataset.",
                              attrs=row,
                              provenance=stat_prov)
            self.add_source_link(eid, stats_sid)
        self.input_totals["district_statistics"] = len(stats)

    def process_accidents(self) -> None:
        records_path = ROOT / "data/processed/accidents/records.json"
        # Raw CSV is an upstream source; rows are counted and linked without duplicating its records.
        raw_path = ROOT / "data/raw/accidents/records/rawalpindi_rta.csv"
        raw_count = 0
        if raw_path.is_file():
            with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
                raw_count = max(0, sum(1 for _ in csv.reader(stream)) - 1)
        raw_sid = self.add_file_source("data/raw/accidents/records/rawalpindi_rta.csv",
                                       kind="raw_accident_dataset", status="processed",
                                       discovered=raw_count, processed=raw_count,
                                       note="All patient-level source rows are represented separately and linked by EcNumber to their normalized event entities.")
        records_sid = self.add_file_source("data/processed/accidents/records.json", kind="normalized_accident_dataset",
                                           status="processed", discovered=34806, processed=34806,
                                           note="Each normalized record is represented as an incident entity; fields are preserved as attributes.")
        # Existing derived JSONL is counted and recorded, not duplicated as graph facts.
        jsonl_path = "data/processed/accidents/rag_documents/accidents.jsonl"
        if (ROOT / jsonl_path).is_file():
            n = sum(1 for _ in (ROOT / jsonl_path).open("r", encoding="utf-8"))
            self.add_file_source(jsonl_path, kind="derived_rag_document_export", status="derived",
                                 discovered=n, processed=n,
                                 note="Derived JSONL mirrors normalized records; counted but not duplicated as entities.")

        # Source IDs are sometimes available in records; map them to the source catalog.
        locations: dict[str, str] = {}
        vehicle_categories: dict[str, str] = {}
        accident_kinds: dict[str, str] = {}
        incident_ids: set[str] = set()
        total = 0
        for row in stream_json_array(records_path):
            total += 1
            rid = str(row.get("id") or f"record-{total}")
            event_id = stable_id("incident", rid)
            incident_ids.add(event_id)
            source_id = row.get("source_id")
            source_ref = self.resolve_external(source_id) or records_sid
            prov = [self.ref(records_sid, source_id=rid, method="normalized JSON record", location=rid)]
            if source_ref != records_sid:
                prov.append(self.ref(source_ref, source_id=str(source_id), method="record source_id"))
            all_attrs = dict(row)
            # keep the stable identifier in attributes and preserve all source-specific fields.
            self.s.add_entity(entity_id=event_id, name=row.get("title") or rid,
                              entity_type="AccidentIncident",
                              description=row.get("description") or row.get("title") or "",
                              attrs=all_attrs, aliases=[rid], provenance=prov,
                              dedup_key=f"incident:{rid}")
            loc = row.get("location")
            if loc and normalize(loc):
                loc_key = normalize(loc)
                loc_id = locations.get(loc_key)
                if not loc_id:
                    loc_id = stable_id("location", loc_key)
                    locations[loc_key] = loc_id
                    self.s.add_entity(entity_id=loc_id, name=str(loc), entity_type="Location",
                                      description="Location label as supplied by one or more source records.",
                                      attrs={"normalization": "casefolded whitespace only"},
                                      provenance=[self.ref(records_sid, source_id=rid, method="incident location field")],
                                      dedup_key="location:" + loc_key)
                self.s.add_relation(event_id, loc_id, "OCCURRED_AT", "OCCURRED AT",
                                    description="Location label recorded for this incident.",
                                    provenance=[self.ref(records_sid, source_id=rid, method="location field")])
            kind = row.get("accident_type")
            if kind and normalize(kind):
                key = normalize(kind)
                kind_id = accident_kinds.get(key)
                if not kind_id:
                    kind_id = stable_id("incident-type", key)
                    accident_kinds[key] = kind_id
                    self.s.add_entity(entity_id=kind_id, name=str(kind), entity_type="IncidentType",
                                      description="Incident category label used in the normalized records.",
                                      provenance=[self.ref(records_sid, source_id=rid, method="accident_type field")],
                                      dedup_key="incident-type:" + key)
                self.s.add_relation(event_id, kind_id, "HAS_INCIDENT_TYPE", "HAS INCIDENT TYPE",
                                    provenance=[self.ref(records_sid, source_id=rid, method="accident_type field")])
            for vehicle in row.get("vehicle_types") or []:
                if not vehicle or not normalize(vehicle):
                    continue
                key = normalize(vehicle)
                vehicle_id = vehicle_categories.get(key)
                if not vehicle_id:
                    vehicle_id = self.vehicle_category(str(vehicle), records_sid, rid, "vehicle_types field")
                    vehicle_categories[key] = vehicle_id
                self.s.add_relation(event_id, vehicle_id, "INVOLVES_VEHICLE_TYPE", "INVOLVES VEHICLE TYPE",
                                    provenance=[self.ref(records_sid, source_id=rid, method="vehicle_types field")])
            if total % 5000 == 0:
                self.s.db.commit()
                print(f"[knowledge-graph] represented {total:,}/34,806 accident records", flush=True)
        self.s.db.execute("UPDATE sources SET discovered=?,processed=? WHERE id=?", (total, total, records_sid))
        if total != 34806:
            self.process_failures.append({"source": "data/processed/accidents/records.json",
                                          "error": f"Expected audited record count 34806, parsed {total}."})
        self.input_totals["accident_records"] = total
        self.input_totals["raw_accident_csv_rows"] = raw_count

        # Preserve every row of the upstream patient-level CSV, which is grouped
        # by EcNumber in the normalized event dataset. The relation below follows
        # that explicit source key; no demographic rows are averaged or dropped.
        raw_row_count = 0
        if raw_path.is_file():
            with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for raw_row_count, row in enumerate(csv.DictReader(stream), start=1):
                    ec_number = str(row.get("EcNumber") or "")
                    source_row_no = raw_row_count + 1  # header occupies CSV row 1
                    row_id = stable_id("raw-accident-row", f"{source_row_no}:{dump(row)}")
                    row_entity = self.s.add_entity(entity_id=row_id,
                                                   name=f"Rawalpindi accident source row {source_row_no} (EcNumber {ec_number})",
                                                   entity_type="AccidentSourceRow",
                                                   description="Verbatim normalized field values from one source CSV row.",
                                                   attrs={**row, "csv_row_number": source_row_no},
                                                   aliases=[f"EcNumber:{ec_number}", f"CSV row:{source_row_no}"],
                                                   provenance=[self.ref(raw_sid, source_id=ec_number,
                                                                        location=f"CSV row {source_row_no}",
                                                                        method="CSV row; grouped by EcNumber")],
                                                   dedup_key=f"raw-accident-row:{source_row_no}")
                    external_sid = self.resolve_external("harvard-rawalpindi-rta-2020-2023")
                    if external_sid:
                        self.s.add_entity(entity_id=row_entity,
                                          name=f"Rawalpindi accident source row {source_row_no} (EcNumber {ec_number})",
                                          entity_type="AccidentSourceRow", description="",
                                          provenance=[self.ref(external_sid, source_id=ec_number,
                                                               location=f"CSV row {source_row_no}",
                                                               method="Harvard dataset row")])
                    event_id = stable_id("incident", f"rwp-rta-{ec_number}")
                    if event_id in incident_ids:
                        self.s.add_relation(row_entity, event_id, "GROUPED_UNDER_NORMALIZED_EVENT",
                                            "GROUPED UNDER NORMALIZED EVENT",
                                            description="The normalized ingestion pipeline groups source rows by this exact EcNumber.",
                                            provenance=[self.ref(raw_sid, source_id=ec_number,
                                                                 location=f"CSV row {source_row_no}",
                                                                 method="EcNumber grouping key")],
                                            identity=f"raw-row-event:{source_row_no}:{event_id}")
                    else:
                        self.unresolved_relationships += 1
                    if raw_row_count % 5000 == 0:
                        self.s.db.commit()
                        print(f"[knowledge-graph] represented {raw_row_count:,}/{raw_count:,} raw CSV rows", flush=True)
        if raw_row_count != raw_count:
            self.process_failures.append({"source": "data/raw/accidents/records/rawalpindi_rta.csv",
                                          "error": f"Counted {raw_count} rows but represented {raw_row_count}."})
        self.input_totals["raw_accident_source_rows_represented"] = raw_row_count

        # Independently verified accident image metadata; link only explicit record_id matches.
        image_path = ROOT / "data/processed/accidents/image_metadata.json"
        image_rows = json.loads(image_path.read_text(encoding="utf-8"))
        image_sid = self.add_file_source("data/processed/accidents/image_metadata.json", kind="verified_accident_image_metadata",
                                         status="processed", discovered=len(image_rows), processed=len(image_rows),
                                         note="Image provenance and verification fields represented; incident links require explicit record_id.")
        for row in image_rows:
            original = str(row.get("id") or "")
            eid = stable_id("accident-image", original)
            asset = row.get("image_path")
            asset_rel = str(asset).replace("\\", "/") if asset else None
            asset_exists = bool(asset_rel and (ROOT / asset_rel).is_file())
            asset_sid = self.add_file_source(asset_rel, title=Path(asset_rel).name, kind="accident_image",
                                             status="processed" if asset_exists else "unavailable",
                                             discovered=1, processed=1 if asset_exists else 0,
                                             note="Image file referenced by verified accident metadata.") if asset_rel else image_sid
            image_prov = [self.ref(image_sid, source_id=original, method="verified image metadata row")]
            ext_sid = self.resolve_external(row.get("source_id"))
            if ext_sid:
                image_prov.append(self.ref(ext_sid, source_id=str(row.get("source_id")), method="source_id field"))
            if row.get("source_url"):
                ext_url = self.resolve_external_url(row.get("source_url"))
                if ext_url:
                    image_prov.append(self.ref(ext_url, source_id=str(row.get("source_url")), method="source_url field"))
            self.s.add_entity(entity_id=eid, name=row.get("title") or original, entity_type="AccidentImage",
                              description=row.get("description", ""),
                              attrs={k: v for k, v in row.items() if k not in {"title", "description", "image_path"}},
                              provenance=image_prov,
                              multimodal=[{"sourceRef": asset_sid, "path": asset_rel or "", "modality": "image",
                                           "caption": row.get("title"), "license": row.get("license")}])
            self.add_source_link(eid, image_sid)
            linked_record = row.get("record_id")
            if linked_record:
                incident = stable_id("incident", str(linked_record))
                if self.s.db.execute("SELECT 1 FROM entities WHERE id=?", (incident,)).fetchone():
                    self.s.add_relation(incident, eid, "DOCUMENTED_BY_IMAGE", "DOCUMENTED BY IMAGE",
                                        description="Image metadata explicitly links this image to the record via record_id.",
                                        provenance=[self.ref(image_sid, source_id=original, method="record_id field")])
        self.input_totals["verified_accident_images"] = len(image_rows)

    def process_road_network(self) -> None:
        path = ROOT / "data/simulation/kalma_chowk/network.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = data.get("nodes", {})
        edges = data.get("edges", {})
        sid = self.add_file_source("data/simulation/kalma_chowk/network.json", kind="osm_road_network",
                                   status="processed", discovered=len(nodes) + len(edges),
                                   processed=len(nodes) + len(edges),
                                   note="All OpenStreetMap-derived node and directed road-segment records represented; source metadata and geometries retained.")
        source_row = self.s.db.execute("SELECT entity_ref FROM sources WHERE id=?", (sid,)).fetchone()
        if source_row:
            source_entity = source_row[0]
            self.s.add_entity(entity_id=source_entity, name="Kalma Chowk OpenStreetMap network.json",
                              entity_type="Source", description="Normalized non-synthetic OSM network used by the traffic simulation.",
                              attrs={"network_metadata": {k: data.get(k) for k in ["source", "license", "area", "synthetic", "origin", "bounds", "stats"]}},
                              canonical=False, dedup_key=f"source:{sid}")
        osm_external = self.resolve_external("lahore-digital-twin-2026")
        osm_raw_path = ROOT / "data/simulation/kalma_chowk/kalma_chowk.osm.xml"
        osm_raw_sid = None
        if osm_raw_path.is_file():
            osm_counts = Counter()
            for _, elem in ET.iterparse(osm_raw_path, events=("end",)):
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag != "osm":
                    osm_counts[tag] += 1
                elem.clear()
            osm_raw_sid = self.add_file_source("data/simulation/kalma_chowk/kalma_chowk.osm.xml",
                                               kind="openstreetmap_raw_extract", status="processed",
                                               discovered=sum(osm_counts.values()), processed=sum(osm_counts.values()),
                                               note="Raw OSM node/way/relation XML counted; the normalized network.json is the expanded graph representation.")
        node_ids: dict[str, str] = {}
        for original, row in nodes.items():
            original = str(original)
            eid = stable_id("road-node", original)
            node_ids[original] = eid
            name = row.get("name") or f"OSM node {original}"
            etype = "Intersection" if row.get("incoming") or row.get("outgoing") else "RoadNetworkNode"
            node_prov = [self.ref(sid, source_id=original, method="network.json node")]
            if osm_external:
                node_prov.append(self.ref(osm_external, source_id=original, method="source OSM node identifier"))
            if osm_raw_sid:
                node_prov.append(self.ref(osm_raw_sid, source_id=original, method="OSM node ID"))
            self.s.add_entity(entity_id=eid, name=name, entity_type=etype,
                              description="OpenStreetMap-derived node in the Kalma Chowk simulation network.",
                              attrs={k: v for k, v in row.items() if k not in {"name"}},
                              aliases=[original],
                              provenance=node_prov,
                              dedup_key=f"osm-node:{original}")
        edge_ids: dict[str, str] = {}
        for original, row in edges.items():
            original = str(original)
            eid = stable_id("road-segment", original)
            edge_ids[original] = eid
            name = row.get("name") or row.get("road_name") or f"OSM road segment {original}"
            edge_prov = [self.ref(sid, source_id=original, method="network.json edge")]
            if osm_external:
                edge_prov.append(self.ref(osm_external, source_id=str(row.get("osm_way") or original),
                                          method="OSM way identifier"))
            if osm_raw_sid:
                edge_prov.append(self.ref(osm_raw_sid, source_id=str(row.get("osm_way") or original),
                                          method="OSM way ID"))
            self.s.add_entity(entity_id=eid, name=name, entity_type="RoadSegment",
                              description="Directed road segment as represented in the Kalma Chowk simulation network.",
                              attrs=row, aliases=[original],
                              provenance=edge_prov,
                              dedup_key=f"osm-edge:{original}")
            u = str(row.get("from") or row.get("source") or row.get("u") or "")
            v = str(row.get("to") or row.get("target") or row.get("v") or "")
            if u in node_ids:
                self.s.add_relation(eid, node_ids[u], "STARTS_AT", "STARTS AT",
                                    provenance=[self.ref(sid, source_id=original, method="edge endpoint")])
            if v in node_ids:
                self.s.add_relation(eid, node_ids[v], "ENDS_AT", "ENDS AT",
                                    provenance=[self.ref(sid, source_id=original, method="edge endpoint")])
        self.input_totals["road_network_nodes"] = len(nodes)
        self.input_totals["road_network_segments"] = len(edges)

    def process_simulation_artifacts(self) -> None:
        # SUMO XML is derived from network.json; catalogue rather than duplicating nodes/edges.
        sumo_dir = ROOT / "data/simulation/kalma_chowk/sumo"
        sumo_files = sorted(sumo_dir.glob("*.xml")) if sumo_dir.is_dir() else []
        sumo_items = 0
        for path in sumo_files:
            count = 0
            for _, elem in ET.iterparse(path, events=("end",)):
                if elem.tag.rsplit("}", 1)[-1] != "net":
                    count += 1
                elem.clear()
            sumo_items += max(0, count - 1)
            rel = path.relative_to(ROOT).as_posix()
            self.add_file_source(rel, kind="derived_sumo_network_xml", status="derived",
                                 discovered=max(0, count - 1), processed=max(0, count - 1),
                                 note="SUMO conversion artifact derived from the same OSM network; not duplicated as separate graph entities.")
        self.input_totals["derived_sumo_xml_files"] = len(sumo_files)
        self.input_totals["derived_sumo_xml_items"] = sumo_items

        # The small JSON run report contains recorded simulation outcomes, unlike the binary weights.
        report_path = ROOT / "data/simulation/kalma_chowk/rl_model.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            history = report.get("history") or []
            sid = self.add_file_source("data/simulation/kalma_chowk/rl_model.json", kind="simulation_training_report",
                                       status="processed", discovered=len(history), processed=len(history),
                                       note="Stored training summary and per-episode metrics represented; machine-local model path omitted.")
            run_id = stable_id("simulation-training", "rl_model.json")
            safe_report = {k: v for k, v in report.items() if k not in {"path", "history"}}
            self.s.add_entity(entity_id=run_id, name="Kalma Chowk reinforcement-learning training run",
                              entity_type="SimulationTrainingRun", description="Training summary recorded by the simulation project.",
                              attrs=safe_report,
                              provenance=[self.ref(sid, method="training JSON report")])
            for i, row in enumerate(history):
                episode_id = stable_id("simulation-episode", f"rl_model.json:{i}")
                self.s.add_entity(entity_id=episode_id, name=f"Training episode {i}", entity_type="SimulationEpisode",
                                  description="Per-episode simulation metrics as recorded in the training report.",
                                  attrs={"episode": i, **row},
                                  provenance=[self.ref(sid, source_id=str(i), method="training history row")])
                self.s.add_relation(run_id, episode_id, "HAS_EPISODE", "HAS EPISODE",
                                    provenance=[self.ref(sid, source_id=str(i), method="history order")])
            self.input_totals["rl_training_episodes"] = len(history)
        weights = ROOT / "data/simulation/kalma_chowk/rl_model.pt"
        if weights.is_file():
            self.add_file_source("data/simulation/kalma_chowk/rl_model.pt", kind="simulation_model_weights",
                                 status="reviewed", discovered=1, processed=1,
                                 note="Binary model artifact inventoried; weights are not expanded into graph facts.")

    def semantic_ingestion_status(self) -> dict[str, dict]:
        """Per-PDF semantic ingestion outcome, read from the real run manifest.

        The inventory must not assert an outcome the pipeline never recorded, so
        anything the manifest does not mention is reported as unattempted.
        """
        data = read_pdf_ingestion_manifest()
        parser = data.get("parser") or data.get("requested_parser")
        status: dict[str, dict] = {}
        for item in data.get("documents") or []:
            path = str(item.get("path", "")).replace("\\", "/")
            status[path] = {
                "status": item.get("status", "unknown"),
                "parser": (item.get("processing") or {}).get("parser", parser),
            }
        return status

    def process_pdfs(self) -> None:
        try:
            import pymupdf
        except ImportError as exc:
            self.process_failures.append({"source": "PDFs", "error": f"PyMuPDF unavailable: {exc}"})
            return
        ingested = self.semantic_ingestion_status()
        pdfs = sorted((ROOT / "data/raw/documents").glob("*.pdf"))
        total_pages = total_text_pages = total_semantic = 0
        for pdf in pdfs:
            rel = pdf.relative_to(ROOT).as_posix()
            doc = pymupdf.open(pdf)
            pages = len(doc)
            texts = [page.get_text("text").strip() for page in doc]
            text_pages = sum(bool(text) for text in texts)
            total_pages += pages
            total_text_pages += text_pages
            record = ingested.get(rel)
            outcome = (record or {}).get("status", "not_attempted")
            used_parser = (record or {}).get("parser")
            if outcome == "processed":
                total_semantic += 1
                state = "processed"
                note = (f"PyMuPDF text layer on {text_pages}/{pages} pages; semantic ingestion "
                        f"into the knowledge graph completed with the {used_parser} parser.")
            else:
                state = "partially_processed"
                note = (f"PyMuPDF text layer extracted on {text_pages}/{pages} pages. Semantic "
                        f"extraction outcome recorded by the pipeline: {outcome}.")
            sid = self.add_file_source(rel, kind="traffic_safety_pdf", status=state,
                                       discovered=pages, processed=text_pages, note=note)
            doc_id = stable_id("document", rel)
            self.s.add_entity(entity_id=doc_id, name=pdf.name, entity_type="Document",
                              description=f"Traffic safety PDF in the project. Page text extraction: {text_pages} of {pages} pages.",
                              attrs={"path": rel, "page_count": pages, "pages_with_text": text_pages,
                                     "extraction_method": "PyMuPDF text layer",
                                     "semantic_extraction": outcome,
                                     "semantic_extraction_parser": used_parser},
                              provenance=[self.ref(sid, method="file inventory")])
            for index, text in enumerate(texts, start=1):
                page_id = stable_id("document-page", f"{rel}:{index}")
                state = "processed" if text else "empty_text_layer"
                self.s.add_entity(entity_id=page_id, name=f"{pdf.name} — page {index}",
                                  entity_type="DocumentPage",
                                  description=text if text else "No extractable text layer was present on this PDF page.",
                                  attrs={"page": index, "text_characters": len(text),
                                         "text_extraction_state": state,
                                         "page_text": text},
                                  provenance=[self.ref(sid, source_id=f"page-{index}", location=f"page {index}",
                                                       method="PyMuPDF get_text(text)", state=state)],
                                  multimodal=[{"sourceRef": sid, "path": rel, "modality": "document_page", "page": index}])
                self.s.add_relation(doc_id, page_id, "HAS_PAGE", "HAS PAGE",
                                    description="PDF page order follows the source document.",
                                    provenance=[self.ref(sid, source_id=f"page-{index}", location=f"page {index}",
                                                         method="PDF page sequence")])
            doc.close()
        self.input_totals["pdf_documents"] = len(pdfs)
        self.input_totals["pdf_pages"] = total_pages
        self.input_totals["pdf_pages_with_text"] = total_text_pages
        self.input_totals["pdf_documents_semantically_processed"] = total_semantic

    def process_rag_storage(self) -> None:
        base = ROOT / "backend/rag_storage"
        chunk_path = base / "kv_store_text_chunks.json"
        if chunk_path.is_file():
            chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
            sid = self.add_file_source("backend/rag_storage/kv_store_text_chunks.json", kind="rag_text_chunks",
                                       status="processed", discovered=len(chunks), processed=len(chunks),
                                       note="Persisted LightRAG text chunks represented; chunk content is retained.")
            self.rag_chunk_ids = set(chunks)
            by_doc: dict[str, str] = {}
            for chunk_id, row in chunks.items():
                doc_key = str(row.get("full_doc_id") or "unknown")
                full_doc = by_doc.get(doc_key)
                if full_doc is None:
                    full_doc = stable_id("rag-document", doc_key)
                    by_doc[doc_key] = full_doc
                    self.s.add_entity(entity_id=full_doc, name=doc_key, entity_type="RAGDocument",
                                      description="Document identifier attached to persisted RAG chunks.",
                                      attrs={"full_doc_id": doc_key},
                                      provenance=[self.ref(sid, source_id=doc_key, method="chunk full_doc_id")])
                cid = stable_id("rag-chunk", str(chunk_id))
                content = str(row.get("content") or "")
                self.s.add_entity(entity_id=cid, name=str(chunk_id), entity_type="RAGChunk",
                                  description=content[:4000],
                                  attrs={**row, "chunk_id": chunk_id, "content": content},
                                  provenance=[self.ref(sid, source_id=str(chunk_id), method="persisted RAG chunk",
                                                       chunk_ref=str(chunk_id))])
                self.s.add_relation(full_doc, cid, "CONTAINS_CHUNK", "CONTAINS CHUNK",
                                    provenance=[self.ref(sid, source_id=str(chunk_id), method="full_doc_id")])
            self.input_totals["rag_text_chunks"] = len(chunks)
            self.input_totals["rag_chunk_documents"] = len(by_doc)
        full_path = base / "kv_store_full_docs.json"
        if full_path.is_file():
            docs = json.loads(full_path.read_text(encoding="utf-8"))
            sid = self.add_file_source("backend/rag_storage/kv_store_full_docs.json", kind="rag_full_documents",
                                       status="processed", discovered=len(docs), processed=len(docs),
                                       note="Persisted full-document records retained as RAGDocument entities.")
            for doc_id, row in docs.items():
                eid = stable_id("rag-document", str(doc_id))
                content = row.get("content") or ""
                self.s.add_entity(entity_id=eid, name=str(doc_id), entity_type="RAGFullDocument",
                                  description=str(content)[:4000], attrs={**row, "full_doc_id": doc_id},
                                  provenance=[self.ref(sid, source_id=str(doc_id), method="persisted full document")])
            self.input_totals["rag_full_documents"] = len(docs)
        status_path = base / "kv_store_doc_status.json"
        if status_path.is_file():
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            sid = self.add_file_source("backend/rag_storage/kv_store_doc_status.json", kind="rag_document_status",
                                       status="partially_processed", discovered=len(status_data), processed=len(status_data),
                                       note="Status records retained; an item in handling state is not treated as processed knowledge.")
            for doc_id, row in status_data.items():
                eid = stable_id("rag-doc-status", str(doc_id))
                self.s.add_entity(entity_id=eid, name=str(doc_id), entity_type="RAGDocumentStatus",
                                  description="RAG document ingestion status as persisted.", attrs=row,
                                  provenance=[self.ref(sid, source_id=str(doc_id), method="persisted status record")])
            self.input_totals["rag_document_status_records"] = len(status_data)
        # Empty vector stores and active graph store are accurately catalogued.
        for name, kind in [("vdb_chunks.json", "rag_chunk_vectors"),
                           ("vdb_entities.json", "rag_entity_vectors"),
                           ("vdb_relationships.json", "rag_relationship_vectors")]:
            path = base / name
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                vector_count = len(data.get("data", [])) if isinstance(data, dict) else len(data)
                status = "empty" if not vector_count else "processed"
                self.add_file_source(path.relative_to(ROOT).as_posix(), kind=kind, status=status,
                                     discovered=vector_count, processed=vector_count,
                                     note="Persisted vector records counted; no synthetic vectors generated.")
                self.input_totals[kind] = vector_count
        active = base / "graph_chunk_entity_relation.graphml"
        active_nodes, active_edges = self.import_graphml(active, "active-lightrag", "Current LightRAG graph storage")
        active_source = self.add_file_source(active.relative_to(ROOT).as_posix(), kind="active_lightrag_graph",
                                             status="empty" if not active_nodes and not active_edges else "processed",
                                             discovered=active_nodes + active_edges, processed=active_nodes + active_edges,
                                             note=f"Current graph backend snapshot: {active_nodes} entities and {active_edges} relationships.")
        self.input_totals["active_lightrag_entities"] = active_nodes
        self.input_totals["active_lightrag_relationships"] = active_edges
        # Capture RAG ingestion manifest as a source and keep its errors visible.
        manifest = ROOT / "data/processed/metadata/rag_document_ingestion.json"
        if manifest.is_file():
            data = read_pdf_ingestion_manifest()
            documents = data.get("documents") or []
            processed = sum(item.get("status") == "processed" for item in documents)
            failed = sum(item.get("status") == "failed" for item in documents)
            parser = data.get("parser") or data.get("requested_parser") or "unknown"
            note = (f"RAG-Anything/LightRAG PDF ingestion run using the {parser} parser: "
                    f"{processed} of {len(documents)} documents ingested, {failed} failed. "
                    f"Graph went from {data.get('graph_before', {}).get('entities', 0)} to "
                    f"{data.get('graph_after', {}).get('entities', 0)} entities.")
            sid = self.add_file_source("data/processed/metadata/rag_document_ingestion.json", kind="rag_ingestion_manifest",
                                       status=source_status(data.get("status")),
                                       discovered=len(documents), processed=processed, note=note)
            self.input_totals["rag_pdf_documents_attempted"] = len(documents)
            self.input_totals["rag_pdf_documents_semantic_extraction_failed"] = failed
            self.input_totals["rag_pdf_documents_semantically_processed"] = processed
            self.s.add_entity(entity_id=stable_id("rag-ingestion-attempt", str(data.get("generated_at_utc"))),
                              name="RAG-Anything PDF ingestion attempt",
                              entity_type="RAGIngestionAttempt",
                              description="Recorded PDF ingestion attempt, including parser errors and graph counts.",
                              attrs=data,
                              provenance=[self.ref(sid, method="ingestion manifest JSON")])
        seed_manifest = ROOT / "data/processed/metadata/ingestion_manifest.json"
        if seed_manifest.is_file():
            sid = self.add_file_source("data/processed/metadata/ingestion_manifest.json", kind="rag_seed_ingestion_manifest",
                                       status="partially_processed", discovered=2, processed=1,
                                       note="The manifest records one 7-block multimodal document and two embedding-service failures; its reported structured-source totals are retained as audit metadata.")
            data = json.loads(seed_manifest.read_text(encoding="utf-8"))
            self.s.add_entity(entity_id=stable_id("rag-seed-manifest", str(data.get("completed_at"))),
                              name="Persisted RAG seed-ingestion manifest", entity_type="RAGIngestionManifest",
                              description="Project manifest for structured and multimodal chunk seeding.", attrs=data,
                              provenance=[self.ref(sid, method="ingestion manifest JSON")])

    def import_graphml(self, path: Path, prefix: str, title: str) -> tuple[int, int]:
        if not path.is_file():
            return (0, 0)
        root = ET.parse(path).getroot()
        key_names = {key.get("id"): key.get("attr.name") for key in root.findall("g:key", NS)}
        graph = root.find("g:graph", NS)
        if graph is None:
            return (0, 0)
        node_rows = graph.findall("g:node", NS)
        edge_rows = graph.findall("g:edge", NS)
        if prefix == "legacy" and not node_rows:
            return self.import_legacy_from_existing_xml()
        source_rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        file_sid = self.add_file_source(source_rel, kind="legacy_graph_snapshot" if prefix == "legacy" else "active_lightrag_graph",
                                        status="migrated" if prefix == "legacy" else ("empty" if not node_rows and not edge_rows else "processed"),
                                        discovered=len(node_rows) + len(edge_rows), processed=len(node_rows) + len(edge_rows),
                                        note=f"{title} imported without LLM regeneration; original attributes retained.")
        mapping: dict[str, str] = {}
        chunks = self.rag_chunk_ids
        for node in node_rows:
            old_id = str(node.get("id"))
            data = {key_names.get(item.get("key"), item.get("key")): item.text for item in node.findall("g:data", NS)}
            eid = stable_id("legacy-entity" if prefix == "legacy" else "lightrag-entity", old_id)
            mapping[old_id] = eid
            kind = data.get("entity_type") or "Entity"
            desc = data.get("description") or ""
            attrs = {"legacy_graph": prefix, "legacy_original_id": old_id,
                     "entity_id": data.get("entity_id"), "entity_type": kind,
                     "source_id": data.get("source_id"), "file_path": data.get("file_path"),
                     "created_at": data.get("created_at"), "truncate": data.get("truncate")}
            src_id = data.get("source_id")
            chunk_state = "available" if src_id in chunks else "unavailable" if src_id else None
            if src_id and src_id not in chunks:
                self.missing_legacy_entity_chunks += 1
            prov = [self.ref(file_sid, source_id=src_id, method="GraphML migration",
                             location=data.get("file_path"),
                             state=("available" if src_id in chunks else "unavailable") if src_id else "migrated",
                             chunk_ref=src_id)]
            self.s.add_entity(entity_id=eid, name=old_id.removesuffix(" (" + str(kind) + ")"),
                              entity_type=kind, description=desc, attrs=attrs,
                              provenance=prov, canonical=False,
                              dedup_key=f"graphml:{prefix}:{old_id}")
        for i, edge in enumerate(edge_rows):
            original_source = str(edge.get("source"))
            original_target = str(edge.get("target"))
            if original_source not in mapping or original_target not in mapping:
                self.unresolved_relationships += 1
                continue
            data = {key_names.get(item.get("key"), item.get("key")): item.text for item in edge.findall("g:data", NS)}
            label = data.get("keywords") or data.get("relationship") or "RELATED_TO"
            desc = data.get("description") or ""
            original_sid = data.get("source_id")
            state = "available" if original_sid in chunks else "unavailable" if original_sid else "migrated"
            if original_sid and original_sid not in chunks:
                self.missing_legacy_relationship_chunks += 1
            self.s.add_relation(mapping[original_source], mapping[original_target],
                                "RAG_EXTRACTED_RELATIONSHIP" if prefix == "active-lightrag" else "LEGACY_RAG_RELATIONSHIP",
                                label, description=desc,
                                attrs={**data, "legacy_graph": prefix, "original_source": original_source,
                                       "original_target": original_target, "original_edge_index": i},
                                provenance=[self.ref(file_sid, source_id=original_sid,
                                                     method="GraphML edge migration", location=data.get("file_path"),
                                                     state=state, chunk_ref=original_sid)],
                                direction="undirected" if graph.get("edgedefault") == "undirected" else "directed",
                                identity=f"graphml:{prefix}:{i}:{original_source}:{original_target}:{label}:{desc}")
        if prefix == "legacy":
            self.legacy_entity_ids = mapping
        return len(node_rows), len(edge_rows)

    def import_legacy_from_existing_xml(self) -> tuple[int, int]:
        xml_path = OUT / "knowledge_graph.xml"
        if not xml_path.is_file():
            return (0, 0)
        # On regeneration after old GraphML removal, migrate the exact legacy subset
        # from the prior canonical XML using provenance/attributes preserved there.
        entities: dict[str, str] = {}
        count_nodes = 0
        for _, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag == "entity":
                attrs = {a.get("name"): a.text for a in elem.findall("./attributes/attribute")}
                if attrs.get("legacy_graph") == "legacy":
                    old_id = attrs.get("legacy_original_id") or elem.get("name") or ""
                    eid = stable_id("legacy-entity", old_id)
                    entities[old_id] = eid
                    prov = []
                    for ref in elem.findall("./provenance/reference"):
                        if ref.get("sourceRef"):
                            prov.append({"sourceRef": ref.get("sourceRef"), "sourceId": ref.get("sourceId"),
                                         "location": ref.get("location"), "method": ref.get("method"),
                                         "state": ref.get("state"), "chunkRef": ref.get("chunkRef")})
                            if ref.get("chunkRef") and ref.get("state") == "unavailable":
                                self.missing_legacy_entity_chunks += 1
                    values = {a.get("name"): parse_attr_value(a.text or "", a.get("dataType")) for a in elem.findall("./attributes/attribute")}
                    self.s.add_entity(entity_id=eid, name=elem.get("name") or old_id,
                                      entity_type=elem.get("type") or "Entity",
                                      description=elem.findtext("description") or "", attrs=values,
                                      aliases=[x.text or "" for x in elem.findall("./aliases/alias")],
                                      provenance=prov, canonical=False, dedup_key=f"graphml:legacy:{old_id}")
                    count_nodes += 1
                elem.clear()
        # Re-read relationships; entity parse above clears nodes only, not relation rows.
        count_edges = 0
        for _, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag == "relationship":
                attrs = {a.get("name"): a.text for a in elem.findall("./attributes/attribute")}
                if attrs.get("legacy_graph") == "legacy":
                    old_s = attrs.get("original_source")
                    old_t = attrs.get("original_target")
                    if old_s in entities and old_t in entities:
                        rel_prov = []
                        for ref in elem.findall("./provenance/reference"):
                            rel_prov.append({"sourceRef": ref.get("sourceRef"), "sourceId": ref.get("sourceId"),
                                             "location": ref.get("location"), "method": ref.get("method"),
                                             "state": ref.get("state"), "chunkRef": ref.get("chunkRef")})
                            if ref.get("chunkRef") and ref.get("state") == "unavailable":
                                self.missing_legacy_relationship_chunks += 1
                        values = {a.get("name"): parse_attr_value(a.text or "", a.get("dataType")) for a in elem.findall("./attributes/attribute")}
                        self.s.add_relation(entities[old_s], entities[old_t], "LEGACY_RAG_RELATIONSHIP",
                                            elem.get("label") or elem.get("type") or "RELATED_TO",
                                            description=elem.findtext("description") or "", attrs=values,
                                            provenance=rel_prov, direction=elem.get("direction", "undirected"),
                                            identity=f"graphml:legacy:{values.get('original_edge_index')}:{old_s}:{old_t}:{elem.get('label')}:{elem.findtext('description') or ''}",
                                            relationship_id=elem.get("id"))
                        count_edges += 1
                elem.clear()
        self.legacy_entity_ids = entities
        return count_nodes, count_edges

    def process_legacy_export(self) -> None:
        path = ROOT / "graphs/lahore_traffic_rag_anything.graphml"
        svg_path = ROOT / "graphs/lahore_traffic_rag_anything.svg"
        self.audit_legacy_artifacts(path, svg_path)
        nodes, edges = self.import_graphml(path, "legacy", "Obsolete project graph artifact")
        if not path.exists() and (OUT / "knowledge_graph.xml").is_file():
            previous = OUT / "knowledge_graph.xml"
            old_source = None
            for _, elem in ET.iterparse(previous, events=("end",)):
                if elem.tag == "source" and elem.get("kind") == "legacy_graph_snapshot":
                    old_source = {"sha256": elem.get("sha256"), "size": elem.get("sizeBytes"),
                                  "items": elem.get("itemsDiscovered")}
                    break
            if old_source:
                sid = self.source(key="file:graphs/lahore_traffic_rag_anything.graphml",
                                  title="lahore_traffic_rag_anything.graphml", kind="legacy_graph_snapshot",
                                  status="migrated", path="graphs/lahore_traffic_rag_anything.graphml",
                                  media_type="application/graphml+xml", sha256=old_source["sha256"],
                                  size=int(old_source["size"]) if old_source["size"] else None,
                                  discovered=int(old_source["items"]) if old_source["items"] else None,
                                  processed=int(old_source["items"]) if old_source["items"] else None,
                                  available=True,
                                  description="Historical project LightRAG GraphML snapshot; its contents are preserved in this canonical XML corpus.",
                                  note="Physical legacy artifact removed after migration; graph data and original IDs remain in this corpus.")
                self.file_sources["graphs/lahore_traffic_rag_anything.graphml"] = sid
            # Stored here because import_graphml's path-missing branch is intentionally empty.
            nodes, edges = self.import_legacy_from_existing_xml()
        self.input_totals["legacy_graph_entities"] = nodes
        self.input_totals["legacy_graph_relationships"] = edges
        self.input_totals["legacy_entity_source_chunks_unavailable"] = self.missing_legacy_entity_chunks
        self.input_totals["legacy_relationship_source_chunks_unavailable"] = self.missing_legacy_relationship_chunks

    def audit_legacy_artifacts(self, graph_path: Path, svg_path: Path) -> None:
        prior = {}
        prior_stats = OUT / "knowledge_graph_stats.json"
        if prior_stats.is_file():
            try:
                prior = json.loads(prior_stats.read_text(encoding="utf-8")).get("legacy_artifact_migration", {})
            except (OSError, json.JSONDecodeError):
                prior = {}
        inventory = dict(prior)
        if graph_path.is_file():
            root = ET.parse(graph_path).getroot()
            ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
            inventory["graphml"] = {
                "path": graph_path.relative_to(ROOT).as_posix(), "size_bytes": graph_path.stat().st_size,
                "sha256": sha256_file(graph_path),
                "entities": len(root.findall(".//g:node", ns)),
                "relationships": len(root.findall(".//g:edge", ns)),
                "status": "audited_before_migration",
            }
        elif inventory.get("graphml"):
            inventory["graphml"] = {**inventory["graphml"], "status": "removed_after_migration"}
        if svg_path.is_file():
            root = ET.parse(svg_path).getroot()
            inventory["svg"] = {
                "path": svg_path.relative_to(ROOT).as_posix(), "size_bytes": svg_path.stat().st_size,
                "sha256": sha256_file(svg_path),
                "entities_displayed": sum(1 for elem in root.iter() if elem.get("class") == "graph-node"),
                "relationships_displayed": sum(1 for elem in root.iter() if elem.get("class") == "graph-edge"),
                "status": "audited_before_migration",
            }
        elif inventory.get("svg"):
            inventory["svg"] = {**inventory["svg"], "status": "removed_after_migration"}
        inventory["removed_after_migration"] = bool(inventory.get("graphml") and inventory.get("svg")
                                                      and not graph_path.exists() and not svg_path.exists())
        self.legacy_artifact_inventory = inventory

    def process_simulation_db(self) -> None:
        path = ROOT / "backend/storage/app.db"
        if not path.is_file():
            return
        sid = self.add_file_source("backend/storage/app.db", kind="simulation_database", status="partially_processed",
                                   note="Read-only extraction of twin_experiments and twin_training. The 678 chat messages were excluded; no user conversation content is part of this graph.")
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        counts = {name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                  for name in ["twin_experiments", "twin_training"]}
        for row in con.execute("SELECT * FROM twin_experiments ORDER BY id"):
            keys = [item[1] for item in con.execute("PRAGMA table_info(twin_experiments)")]
            record = dict(zip(keys, row))
            eid = stable_id("simulation-experiment", str(record["id"]))
            attrs = {k: record.get(k) for k in ["id", "kind", "controller", "scenario", "seed", "demand_scale", "duration", "engine", "config", "metrics", "comparison", "created_at"]}
            self.s.add_entity(entity_id=eid, name=f"Traffic simulation experiment {record['id'][:8]}",
                              entity_type="SimulationExperiment", description=record.get("notes") or "Stored traffic simulation experiment.",
                              attrs=attrs, provenance=[self.ref(sid, source_id=str(record["id"]), method="read-only app.db row")])
        training_keys = [item[1] for item in con.execute("PRAGMA table_info(twin_training)")]
        for row in con.execute("SELECT * FROM twin_training ORDER BY id"):
            record = dict(zip(training_keys, row))
            payload = json.loads(record.get("payload") or "{}")
            payload.pop("path", None)  # machine-local filesystem path is not graph knowledge.
            eid = stable_id("simulation-training", str(record["id"]))
            self.s.add_entity(entity_id=eid, name=f"Traffic controller training run {record['id'][:8]}",
                              entity_type="SimulationTrainingRun", description="Stored traffic-control training run.",
                              attrs={"id": record["id"], "status": record["status"], "episodes": record["episodes"],
                                     "payload": payload, "created_at": record["created_at"], "updated_at": record["updated_at"]},
                              provenance=[self.ref(sid, source_id=str(record["id"]), method="read-only app.db row")])
        con.close()
        self.input_totals["simulation_experiments"] = counts["twin_experiments"]
        self.input_totals["simulation_training_runs"] = counts["twin_training"]

    def process_known_files(self, source_rows: list[dict[str, Any]]) -> None:
        # Inventory locally present official snapshots, including duplicate snapshots.
        known_paths = {str(row.get("local_file_path")) for row in source_rows if row.get("local_file_path")}
        seen_html_hashes: set[str] = set()
        for rel in sorted(known_paths):
            p = ROOT / rel
            if not p.is_file():
                continue
            if p.suffix.lower() in {".html", ".htm"}:
                digest = sha256_file(p)
                duplicate = digest in seen_html_hashes
                seen_html_hashes.add(digest)
                self.add_file_source(rel, kind="official_source_snapshot",
                                     status="duplicate" if duplicate else "reviewed",
                                     discovered=1, processed=1,
                                     note="Byte-identical official HTML snapshots are catalogued but do not duplicate graph facts.")
            else:
                self.add_file_source(rel, kind="catalogued_source_file", status="reviewed",
                                     discovered=1, processed=1, note="Local source-catalog file reviewed.")
        # Source-code files that define relevant data preparation/retrieval behavior.
        code_files = ["backend/app/rag/engine.py", "backend/app/ingestion/ingest.py",
                      "backend/app/ingestion/service.py", "backend/app/api/routes/ingestion.py",
                      "backend/app/core/config.py", "scripts/ingest_rag_documents.py",
                      "scripts/build_knowledge_graph.py"]
        for rel in code_files:
            if (ROOT / rel).is_file():
                self.add_file_source(rel, kind="project_architecture_code", status="reviewed",
                                     discovered=1, processed=1,
                                     note="Implementation inspected to document parser, embedding, graph-store, and ingestion behavior; code text is not traffic fact content.")
        # Reference paper is source methodology, separate from traffic facts.
        paper = "Research Paper/rag anything research paper.pdf"
        if (ROOT / paper).is_file():
            self.add_file_source(paper, kind="research_methodology", status="reviewed",
                                 discovered=18, processed=18,
                                 note="All 18 paper pages reviewed; methodological source only, not a Lahore traffic fact source.")

    def build(self) -> dict[str, Any]:
        print("[knowledge-graph] indexing source catalog and structured datasets", flush=True)
        source_rows = self.build_sources_catalog()
        self.structured_data(source_rows)
        print("[knowledge-graph] streaming accident and image records", flush=True)
        self.process_accidents()
        print("[knowledge-graph] importing road network, PDFs, and RAG stores", flush=True)
        self.process_road_network()
        self.process_simulation_artifacts()
        self.process_pdfs()
        self.process_rag_storage()
        self.process_legacy_export()
        self.process_simulation_db()
        self.process_known_files(source_rows)
        print("[knowledge-graph] materializing provenance edges and finalizing", flush=True)
        self.connect_source_edges()
        self.s.db.commit()
        return {"source_rows": source_rows}

    def connect_source_edges(self) -> None:
        """Expose existing provenance as graph edges without inventing associations."""
        rows = self.s.db.execute("SELECT id,type,provenance FROM entities WHERE json_array_length(provenance)>0").fetchall()
        for eid, entity_type, raw in rows:
            if entity_type in {"AccidentIncident", "AccidentSourceRow"}:
                # High-volume record rows retain full source provenance in XML;
                # avoid duplicating each citation as high-degree graph edges.
                continue
            for ref in json.loads(raw):
                sid = ref.get("sourceRef")
                source_row = self.s.db.execute("SELECT entity_ref FROM sources WHERE id=?", (sid,)).fetchone()
                if not source_row:
                    continue
                if self.s.db.execute("SELECT 1 FROM relationships WHERE source=? AND target=? AND type='SUPPORTED_BY_SOURCE'", (eid, source_row[0])).fetchone():
                    continue
                self.s.add_relation(eid, source_row[0], "SUPPORTED_BY_SOURCE", "SUPPORTED BY SOURCE",
                                    description="Source link derived directly from the entity provenance reference.",
                                    provenance=[ref], identity=f"source-link:{eid}:{sid}:SUPPORTED_BY_SOURCE")


def media_type(path: Path | None) -> str | None:
    if not path:
        return None
    return {".pdf": "application/pdf", ".json": "application/json", ".jsonl": "application/x-ndjson",
            ".csv": "text/csv", ".xml": "application/xml", ".graphml": "application/graphml+xml",
            ".html": "text/html", ".svg": "image/svg+xml", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")


def find_asset_path(path_value: str, fallback_name: str) -> str | None:
    candidates = [path_value.replace("\\", "/")]
    if not path_value.startswith("data/"):
        candidates += [f"data/raw/traffic_signs/{path_value}", f"data/raw/images/traffic_signs/{path_value}",
                       f"data/raw/traffic_signs/{fallback_name}", f"data/raw/images/traffic_signs/{fallback_name}"]
    for rel in candidates:
        if rel and (ROOT / rel).is_file():
            return rel
    base = Path(fallback_name).name
    for parent in [ROOT / "data/raw/traffic_signs", ROOT / "data/raw/images/traffic_signs", ROOT / "data/raw/road_images"]:
        if parent.is_dir():
            found = next(parent.rglob(base), None)
            if found and found.is_file():
                return found.relative_to(ROOT).as_posix()
    return None


def parse_attr_value(text: str, kind: str | None) -> Any:
    if kind == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if kind == "integer":
        try:
            return int(text)
        except ValueError:
            pass
    if kind == "decimal":
        try:
            return float(text)
        except ValueError:
            pass
    if kind == "boolean":
        return text.casefold() == "true"
    return text


def xml_metadata(active_graph_entities: int = 0, active_graph_relationships: int = 0) -> ET.Element:
    meta = ET.Element("metadata", {"schemaVersion": SCHEMA_VERSION,
                                   "generatedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "generatorVersion": GENERATOR_VERSION})
    ingestion = read_pdf_ingestion_manifest()
    documents = ingestion.get("documents") or []
    processed = sum(item.get("status") == "processed" for item in documents)
    failed = sum(item.get("status") == "failed" for item in documents)
    parser = ingestion.get("parser") or ingestion.get("requested_parser") or "not recorded"
    ET.SubElement(meta, "title").text = "Lahore Traffic Intelligence source-grounded knowledge graph"
    ET.SubElement(meta, "description").text = "Graph corpus combining project traffic datasets, extracted source-page text, stored RAG chunks, current graph storage, and migrated legacy LightRAG graph evidence."
    ET.SubElement(meta, "paperReference", {"path": "Research Paper/rag anything research paper.pdf",
                                            "title": "RAG-Anything: All-in-One RAG Framework",
                                            "arxiv": "2510.12323", "pagesReviewed": "18"})
    pdf_note = (f"The four traffic PDFs were parsed with the {parser} parser, but "
                f"{processed} of {len(documents) or 4} were ingested into the knowledge graph and "
                f"{failed} were not, as recorded in "
                "data/processed/metadata/rag_document_ingestion.json. PDF page text is present in "
                "this export; PDF-derived graph entities are not.")
    if ingestion.get("provider_limits"):
        limits_map = ingestion["provider_limits"]
        pdf_note += (f" The configured provider allows {limits_map.get('tokens_per_minute')} tokens "
                     f"per minute and {limits_map.get('tokens_per_day')} per day, which is below the "
                     "cost of extracting this corpus.")
    ET.SubElement(meta, "methodology").text = ("RAG-Anything's multimodal atomic content and dual-graph design informed the provenance and multimodal links. "
        "This export is an auditable project corpus: deterministic source records are represented directly; existing LightRAG graph data is migrated as-is; "
        "PDF pages are represented from their extracted text, and entities and relationships attributed to a PDF come only from the recorded ingestion run.")
    ET.SubElement(meta, "implementationProfile").text = (
        "RAG-Anything 1.4.1 with LightRAG NetworkXStorage configured in backend/app/rag/engine.py; "
        f"the active graph snapshot contained {active_graph_entities} entities and "
        f"{active_graph_relationships} relationships at build time."
    )
    ET.SubElement(meta, "limitations")
    limits = meta.find("limitations")
    items = [
        pdf_note,
        "Page text in this export comes from the PyMuPDF text layer. Pages without one are "
        "still represented, and OCR-derived text is only present where the ingestion run recorded it.",
        f"The active LightRAG graph held {active_graph_entities} entities and "
        f"{active_graph_relationships} relationships when this export was built; entity and "
        "relationship descriptions come from the extraction model and are not independently verified.",
        "The old project graph is a 89-node/172-edge LightRAG GraphML snapshot; its original node/edge fields and chunk ids are preserved, while unavailable chunk references are marked.",
        "The normalized accident dataset covers multiple locations and sources, including Rawalpindi and other Pakistan locations; records are not relabelled as Lahore incidents.",
    ]
    for item in items:
        ET.SubElement(limits, "limitation").text = item
    return meta


def write_xml(store: Store, path: Path, active_graph_entities: int = 0,
              active_graph_relationships: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<knowledgeGraph>\n')
        stream.write(ET.tostring(xml_metadata(active_graph_entities, active_graph_relationships), encoding="utf-8"))
        stream.write(b"\n<sources>\n")
        cur = store.db.execute("SELECT id,kind,status,path,external_id,media_type,sha256,size_bytes,discovered,processed,entity_ref,available,title,description,note,error FROM sources ORDER BY id")
        for row in cur:
            (sid, kind, status, path_value, external, med, sha, size, discovered, processed,
             entity_ref, available, title, desc, note, error) = row
            attrs = {"id": sid, "kind": kind, "status": status, "available": str(bool(available)).lower()}
            for key, value in [("path", path_value), ("externalId", external), ("mediaType", med), ("sha256", sha),
                               ("sizeBytes", size), ("itemsDiscovered", discovered), ("itemsProcessed", processed),
                               ("entityRef", entity_ref)]:
                if value is not None:
                    attrs[key] = str(value)
            node = ET.Element("source", attrs)
            ET.SubElement(node, "title").text = title or ""
            ET.SubElement(node, "description").text = desc or ""
            ET.SubElement(node, "processingNote").text = note or ""
            if error:
                ET.SubElement(node, "error").text = error
            stream.write(serialize_element(node)); stream.write(b"\n")
        stream.write(b"</sources>\n<entities>\n")
        for row in store.db.execute("SELECT id,name,type,description,attrs,aliases,provenance,multimodal,canonical FROM entities ORDER BY id"):
            eid, name, typ, desc, attrs, aliases, prov, multimodal, canonical = row
            node = ET.Element("entity", {"id": eid, "name": name, "type": typ, "canonical": str(bool(canonical)).lower()})
            ET.SubElement(node, "description").text = desc or ""
            at = ET.SubElement(node, "attributes")
            for key, value in sorted(json.loads(attrs).items()):
                el = ET.SubElement(at, "attribute", {"name": str(key), "dataType": json_type(value)})
                el.text = json_value(value)
            als = ET.SubElement(node, "aliases")
            for alias in json.loads(aliases):
                ET.SubElement(als, "alias").text = alias
            p = ET.SubElement(node, "provenance")
            for ref in json.loads(prov):
                add_ref(p, ref)
            mm = ET.SubElement(node, "multimodal")
            for asset in json.loads(multimodal):
                attrs2 = {k: str(v) for k, v in asset.items() if v is not None and k in {"sourceRef", "path", "modality", "page", "caption", "license"}}
                if "sourceRef" in attrs2 and "path" in attrs2 and "modality" in attrs2:
                    ET.SubElement(mm, "asset", attrs2)
            stream.write(serialize_element(node)); stream.write(b"\n")
        stream.write(b"</entities>\n<relationships>\n")
        for row in store.db.execute("SELECT id,source,target,type,label,direction,description,attrs,provenance FROM relationships ORDER BY id"):
            rid, source, target, typ, label, direction, desc, attrs, prov = row
            node = ET.Element("relationship", {"id": rid, "source": source, "target": target,
                                                  "type": typ, "label": label, "direction": direction})
            ET.SubElement(node, "description").text = desc or ""
            at = ET.SubElement(node, "attributes")
            for key, value in sorted(json.loads(attrs).items()):
                el = ET.SubElement(at, "attribute", {"name": str(key), "dataType": json_type(value)})
                el.text = json_value(value)
            p = ET.SubElement(node, "provenance")
            for ref in json.loads(prov):
                add_ref(p, ref)
            stream.write(serialize_element(node)); stream.write(b"\n")
        stream.write(b"</relationships>\n<conflicts>\n")
        for cid, subject, field, status, desc, assertions in store.db.execute("SELECT * FROM conflicts ORDER BY id"):
            node = ET.Element("conflict", {"id": cid, "subjectEntityRef": subject, "field": field, "status": status})
            ET.SubElement(node, "description").text = desc
            for assertion in json.loads(assertions):
                el = ET.SubElement(node, "assertion", {"sourceRef": assertion["sourceRef"]})
                el.text = assertion["value"]
            stream.write(serialize_element(node)); stream.write(b"\n")
        stream.write(b"</conflicts>\n</knowledgeGraph>\n")


def add_ref(parent: ET.Element, ref: dict[str, Any]) -> None:
    attrs = {"sourceRef": str(ref["sourceRef"])}
    for key, xml_key in [("sourceId", "sourceId"), ("location", "location"), ("method", "method"),
                         ("state", "state"), ("chunkRef", "chunkRef")]:
        if ref.get(key) is not None and str(ref.get(key)) != "":
            attrs[xml_key] = str(ref[key])
    ET.SubElement(parent, "reference", attrs)


def validate_corpus(xml_path: Path, xsd_path: Path, store: Store) -> dict[str, Any]:
    from lxml import etree
    schema_doc = etree.parse(str(xsd_path))
    schema = etree.XMLSchema(schema_doc)
    # The corpus is hundreds of megabytes, which is past libxml2's default parser
    # limits; without this the DOM build fails with "Memory allocation failed".
    doc = etree.parse(str(xml_path), etree.XMLParser(huge_tree=True))
    schema.assertValid(doc)
    # Independent streaming endpoint / duplicate / source reference checks.
    entity_ids: set[str] = set()
    relationship_ids: set[str] = set()
    entity_count = relation_count = source_count = conflict_count = 0
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "source":
            source_count += 1
        elif elem.tag == "entity":
            eid = elem.get("id")
            if eid in entity_ids:
                raise ValueError(f"duplicate entity id: {eid}")
            entity_ids.add(eid); entity_count += 1
        elif elem.tag == "relationship":
            rid = elem.get("id")
            if rid in relationship_ids:
                raise ValueError(f"duplicate relationship id: {rid}")
            relationship_ids.add(rid); relation_count += 1
        elif elem.tag == "conflict":
            conflict_count += 1
        elem.clear()
    # XSD keyrefs guarantee endpoints and provenance source IDs; compare with staging counts.
    expected = (store.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
                store.db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0],
                store.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                store.db.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])
    if (entity_count, relation_count, source_count, conflict_count) != expected:
        raise ValueError(f"XML count mismatch: XML={(entity_count, relation_count, source_count, conflict_count)} staging={expected}")
    if store.db.execute("SELECT COUNT(*) FROM relationships r LEFT JOIN entities s ON s.id=r.source LEFT JOIN entities t ON t.id=r.target WHERE s.id IS NULL OR t.id IS NULL").fetchone()[0]:
        raise ValueError("Relationship endpoint check failed")
    missing_refs = store.db.execute("""SELECT COUNT(*) FROM entities e, json_each(e.provenance) p
       LEFT JOIN sources s ON s.id=json_extract(p.value,'$.sourceRef') WHERE s.id IS NULL""").fetchone()[0]
    missing_rel_refs = store.db.execute("""SELECT COUNT(*) FROM relationships r, json_each(r.provenance) p
       LEFT JOIN sources s ON s.id=json_extract(p.value,'$.sourceRef') WHERE s.id IS NULL""").fetchone()[0]
    if missing_refs or missing_rel_refs:
        raise ValueError(f"Missing source refs: entity={missing_refs}, relationship={missing_rel_refs}")
    return {"xsd_validation": "passed", "xml_parse": "passed", "duplicate_entity_ids": 0,
            "duplicate_relationship_ids": 0, "invalid_relationship_endpoints": 0,
            "invalid_source_references": 0, "entity_count": entity_count,
            "relationship_count": relation_count, "source_count": source_count,
            "conflict_count": conflict_count, "xml_bytes": xml_path.stat().st_size}


def provenance_payload(parent: ET.Element) -> list[dict[str, Any]]:
    result = []
    for ref in parent.findall("./provenance/reference"):
        result.append({key: ref.get(key) for key in
                       ("sourceRef", "sourceId", "location", "method", "state", "chunkRef")
                       if ref.get(key) is not None})
    return result


def parse_xml_entities(xml_path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]], Counter[str], Counter[str]]:
    nodes: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    source_ref_counts: Counter[str] = Counter()
    type_index: Counter[str] = Counter()
    type_order: list[str] = []
    # The staged XML entity elements are individually bounded; positions are deterministic by type and ID.
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "entity":
            typ = elem.get("type") or "Entity"
            type_counts[typ] += 1
            refs = provenance_payload(elem)
            for ref in {item.get("sourceRef") for item in refs}:
                if ref:
                    source_ref_counts[ref] += 1
            attrs = {a.get("name"): parse_attr_value(a.text or "", a.get("dataType"))
                     for a in elem.findall("./attributes/attribute")}
            nodes.append({"id": elem.get("id"), "name": elem.get("name"), "type": typ,
                          "description": elem.findtext("description") or "",
                          "attrs": attrs, "provenance": refs,
                          "aliases": [a.text or "" for a in elem.findall("./aliases/alias")]})
            elem.clear()
    nodes.sort(key=lambda item: item["id"])
    # Type clusters arranged around a circle; nodes placed on a regular spiral within each cluster.
    types = sorted({node["type"] for node in nodes})
    type_anchor: dict[str, tuple[float, float]] = {}
    radius = 420 + min(1600, 40 * len(types))
    for i, typ in enumerate(types):
        angle = 2 * math.pi * i / max(1, len(types))
        type_anchor[typ] = (radius * math.cos(angle), radius * math.sin(angle))
    positions: dict[str, tuple[float, float]] = {}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_type[node["type"]].append(node)
    for typ, group in by_type.items():
        ax, ay = type_anchor[typ]
        n = len(group)
        for i, node in enumerate(group):
            # Sunflower disk, compact for event-heavy types.
            rr = 20 * math.sqrt(i + 1)
            ang = i * 2.399963229728653
            positions[node["id"]] = (ax + rr * math.cos(ang), ay + rr * math.sin(ang))
    return nodes, positions, type_counts, source_ref_counts


def xml_relationships(xml_path: Path) -> Iterator[dict[str, Any]]:
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "relationship":
            yield {"id": elem.get("id"), "source": elem.get("source"), "target": elem.get("target"),
                   "type": elem.get("type"), "label": elem.get("label"),
                   "direction": elem.get("direction"), "description": elem.findtext("description") or "",
                   "provenance": provenance_payload(elem),
                   "attrs": {a.get("name"): parse_attr_value(a.text or "", a.get("dataType"))
                             for a in elem.findall("./attributes/attribute")}}
            elem.clear()


def color_map(types: Iterable[str]) -> dict[str, str]:
    colors = {}
    for index, typ in enumerate(sorted(set(types))):
        hue = (index * 0.618033988749895) % 1.0
        saturation = 0.68 if index % 2 == 0 else 0.78
        lightness = 0.43 + 0.045 * (index % 3)
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors[typ] = f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
    return colors


def write_full_png(xml_path: Path, out_path: Path, nodes: list[dict[str, Any]],
                   positions: dict[str, tuple[float, float]], type_counts: Counter[str],
                   rel_counts: Counter[str], source_count: int) -> dict[str, Any]:
    """Draw a readable, type-level network aggregated from every XML record."""
    import networkx as nx
    from PIL import Image, ImageDraw, ImageFont

    width, height = 4600, 3400
    entity_colors = color_map(type_counts)
    relation_colors = color_map(rel_counts)
    pair_relations: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    type_by_id = {node["id"]: node["type"] for node in nodes}
    relationship_count = 0
    for relation in xml_relationships(xml_path):
        source_type = type_by_id.get(relation["source"])
        target_type = type_by_id.get(relation["target"])
        if source_type is None or target_type is None:
            raise ValueError(f"Missing relationship endpoint in visualization: {relation['id']}")
        pair_relations[(source_type, target_type)][relation["type"]] += 1
        relationship_count += 1
    if relationship_count != sum(rel_counts.values()):
        raise ValueError("Relationship count mismatch while aggregating the PNG graph")

    type_graph = nx.Graph()
    type_graph.add_nodes_from(sorted(type_counts))
    for (source_type, target_type), relations in pair_relations.items():
        weight = math.log1p(sum(relations.values()))
        if type_graph.has_edge(source_type, target_type):
            type_graph[source_type][target_type]["weight"] += weight
        else:
            type_graph.add_edge(source_type, target_type, weight=weight)
    layout = nx.spring_layout(type_graph, seed=42, iterations=160, k=0.8, weight="weight")

    background = Image.new("RGB", (width, height), "#f3f6fa")
    draw = ImageDraw.Draw(background)

    def load_font(size: int):
        for font_path in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    title_font = load_font(46)
    subtitle_font = load_font(24)
    section_font = load_font(26)
    node_font = load_font(22)
    count_font = load_font(19)
    legend_font = load_font(19)
    note_font = load_font(17)

    draw.text((70, 38), "Lahore Traffic Intelligence · Knowledge Graph", font=title_font, fill="#17324f")
    draw.text((72, 101),
              f"Type-level graph from {len(nodes):,} entities · {relationship_count:,} relationships · {source_count:,} sources in knowledge_graph.xml",
              font=subtitle_font, fill="#51657b")

    graph_panel = (60, 155, 3040, 3330)
    legend_panel = (3070, 155, 4540, 3330)
    draw.rounded_rectangle(graph_panel, radius=20, fill="#ffffff", outline="#d5dee9", width=2)
    draw.rounded_rectangle(legend_panel, radius=20, fill="#ffffff", outline="#d5dee9", width=2)
    draw.text((96, 178), f"{len(type_counts)} entity types · {len(pair_relations)} observed type-to-type links",
              font=section_font, fill="#203850")

    plot = (350, 285, 2750, 3120)
    xs = [point[0] for point in layout.values()] or [0.0]
    ys = [point[1] for point in layout.values()] or [0.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    plot_left, plot_top, plot_right, plot_bottom = plot
    centers = {
        typ: (round(plot_left + (layout[typ][0] - min_x) / span_x * (plot_right - plot_left)),
              round(plot_top + (layout[typ][1] - min_y) / span_y * (plot_bottom - plot_top)))
        for typ in type_counts
    }
    box_sizes = {}
    for typ in type_counts:
        label_width = draw.textbbox((0, 0), typ, font=node_font)[2]
        box_sizes[typ] = (max(250, min(390, label_width + 52)), 92)

    ordered_types = sorted(type_counts)
    for iteration in range(600):
        overlaps = 0
        for first_index, first in enumerate(ordered_types):
            first_x, first_y = centers[first]
            first_width, first_height = box_sizes[first]
            for second_index in range(first_index + 1, len(ordered_types)):
                second = ordered_types[second_index]
                second_x, second_y = centers[second]
                second_width, second_height = box_sizes[second]
                overlap_x = (first_width + second_width) / 2 + 24 - abs(second_x - first_x)
                overlap_y = (first_height + second_height) / 2 + 24 - abs(second_y - first_y)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                overlaps += 1
                if overlap_x < overlap_y:
                    direction = 1 if second_x > first_x else -1 if second_x < first_x else (1 if (first_index + second_index) % 2 else -1)
                    shift = overlap_x / 2 + 1
                    first_x = min(max(first_x - direction * shift, plot_left + first_width / 2 + 8),
                                  plot_right - first_width / 2 - 8)
                    second_x = min(max(second_x + direction * shift, plot_left + second_width / 2 + 8),
                                   plot_right - second_width / 2 - 8)
                    centers[first] = (first_x, first_y)
                    centers[second] = (second_x, second_y)
                else:
                    direction = 1 if second_y > first_y else -1 if second_y < first_y else (1 if (first_index + second_index) % 2 else -1)
                    shift = overlap_y / 2 + 1
                    first_y = min(max(first_y - direction * shift, plot_top + first_height / 2 + 8),
                                  plot_bottom - first_height / 2 - 8)
                    second_y = min(max(second_y + direction * shift, plot_top + second_height / 2 + 8),
                                   plot_bottom - second_height / 2 - 8)
                    centers[first] = (first_x, first_y)
                    centers[second] = (second_x, second_y)
                first_x, first_y = centers[first]
        if overlaps == 0:
            break

    def border_point(typ: str, toward: tuple[float, float]) -> tuple[float, float]:
        cx, cy = centers[typ]
        dx, dy = toward[0] - cx, toward[1] - cy
        half_width, half_height = box_sizes[typ][0] / 2 + 2, box_sizes[typ][1] / 2 + 2
        scales = [half_width / abs(dx) if dx else float("inf"),
                  half_height / abs(dy) if dy else float("inf")]
        scale = min(scales)
        return cx + dx * scale, cy + dy * scale

    relation_draw = ImageDraw.Draw(background)
    for (source_type, target_type), relations in sorted(pair_relations.items()):
        total = sum(relations.values())
        dominant_type = max(relations, key=relations.get)
        color = relation_colors[dominant_type]
        line_width = min(10, 2 + int(math.log10(total + 1) * 1.5))
        source_center, target_center = centers[source_type], centers[target_type]
        if source_type == target_type:
            cx, cy = source_center
            half_width, half_height = box_sizes[source_type][0] / 2, box_sizes[source_type][1] / 2
            start_point = (cx + half_width, cy - 14)
            control_one = (cx + half_width + 80, cy - half_height - 95)
            control_two = (cx - half_width - 80, cy - half_height - 95)
            end_point = (cx - half_width, cy - 14)
            curve = []
            for step in range(31):
                t = step / 30
                u = 1 - t
                curve.append((u**3 * start_point[0] + 3 * u**2 * t * control_one[0]
                              + 3 * u * t**2 * control_two[0] + t**3 * end_point[0],
                              u**3 * start_point[1] + 3 * u**2 * t * control_one[1]
                              + 3 * u * t**2 * control_two[1] + t**3 * end_point[1]))
            relation_draw.line(curve, fill=color, width=line_width)
            dx, dy = end_point[0] - curve[-2][0], end_point[1] - curve[-2][1]
            length = max(math.hypot(dx, dy), 1.0)
            ux, uy = dx / length, dy / length
            base_x, base_y = end_point[0] - ux * 20, end_point[1] - uy * 20
            wing = (-uy * 11, ux * 11)
            relation_draw.polygon([end_point, (base_x + wing[0], base_y + wing[1]),
                                   (base_x - wing[0], base_y - wing[1])], fill=color)
            continue
        start_point = border_point(source_type, target_center)
        end_point = border_point(target_type, source_center)
        dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
        length = max(math.hypot(dx, dy), 1.0)
        ux, uy = dx / length, dy / length
        arrow_length = 20
        arrow_width = 11
        base_x, base_y = end_point[0] - ux * arrow_length, end_point[1] - uy * arrow_length
        perpendicular = (-uy * arrow_width, ux * arrow_width)
        relation_draw.line((start_point, (base_x, base_y)), fill=color, width=line_width)
        relation_draw.polygon([end_point,
                               (base_x + perpendicular[0], base_y + perpendicular[1]),
                               (base_x - perpendicular[0], base_y - perpendicular[1])], fill=color)

    # Draw labeled rectangular group nodes over the relationship edges.
    for typ, count in type_counts.items():
        cx, cy = centers[typ]
        box_width, box_height = box_sizes[typ]
        color = entity_colors[typ]
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        pale = tuple(round(channel * 0.18 + 255 * 0.82) for channel in rgb)
        fill = "#%02x%02x%02x" % pale
        rect = (cx - box_width // 2, cy - box_height // 2,
                cx + box_width // 2, cy + box_height // 2)
        draw.rounded_rectangle(rect, radius=18, fill=fill, outline=color, width=4)
        name_bbox = draw.textbbox((0, 0), typ, font=node_font)
        name_width = name_bbox[2] - name_bbox[0]
        count_label = f"{count:,} records"
        count_bbox = draw.textbbox((0, 0), count_label, font=count_font)
        count_width = count_bbox[2] - count_bbox[0]
        draw.text((cx - name_width / 2, cy - 29), typ, font=node_font, fill="#1d3248")
        draw.text((cx - count_width / 2, cy + 5), count_label, font=count_font, fill="#51657b")

    draw.text((96, 3240),
              "Boxes aggregate entities by XML type; arrows aggregate observed relationships by source and target type. Individual records and provenance remain in the XML corpus.",
              font=note_font, fill="#586b80")

    legend_x = 3110
    draw.text((legend_x, 180), "Relationship types", font=section_font, fill="#203850")
    y = 224
    for typ, count in sorted(rel_counts.items(), key=lambda item: (-item[1], item[0])):
        color = relation_colors[typ]
        draw.line((legend_x, y + 12, legend_x + 48, y + 12), fill=color, width=5)
        draw.polygon([(legend_x + 48, y + 12), (legend_x + 36, y + 5),
                      (legend_x + 36, y + 19)], fill=color)
        draw.text((legend_x + 62, y), f"{typ} · {count:,}", font=legend_font, fill="#273c53")
        y += 33

    y += 22
    draw.text((legend_x, y), "Largest type-level links", font=section_font, fill="#203850")
    y += 47
    major_pairs = sorted(pair_relations.items(), key=lambda item: (-sum(item[1].values()), item[0]))[:15]
    for (source_type, target_type), relations in major_pairs:
        total = sum(relations.values())
        dominant_type = max(relations, key=relations.get)
        draw.text((legend_x, y), f"{source_type} → {target_type}", font=count_font, fill="#273c53")
        y += 25
        draw.text((legend_x + 18, y), f"{dominant_type} · {total:,} relationships",
                  font=note_font, fill=relation_colors[dominant_type])
        y += 36
    if y > legend_panel[3] - 20:
        raise ValueError(f"PNG legend exceeds canvas: y={y}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(out_path, format="PNG", optimize=True)
    with Image.open(out_path) as check:
        check.verify()
    with Image.open(out_path) as check:
        image_size = check.size
    if len(type_counts) != type_graph.number_of_nodes() or relationship_count != sum(rel_counts.values()):
        raise ValueError("PNG visualization counts differ from the XML graph")
    return {"status": "generated_from_xml", "format": "PNG",
            "entity_records_summarized": len(nodes), "relationship_records_summarized": relationship_count,
            "visible_entity_type_nodes": type_graph.number_of_nodes(),
            "visible_type_pair_edges": len(pair_relations), "sources": source_count,
            "entity_types": len(type_counts), "relationship_types": len(rel_counts),
            "width": image_size[0], "height": image_size[1], "xml_parse": "passed",
            "layout": "deterministic spring layout on entity types; relationship counts aggregated from XML",
            "file_bytes": out_path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="knowledge-graph-", dir=temp_parent) as temp:
        store = Store(Path(temp) / "stage.sqlite3")
        builder = CorpusBuilder(store)
        try:
            builder.build()
            xml_path = out / "knowledge_graph.xml"
            xsd_path = ROOT / "scripts" / "knowledge_graph.xsd"
            write_xml(store, xml_path,
                      builder.input_totals.get("active_lightrag_entities", 0),
                      builder.input_totals.get("active_lightrag_relationships", 0))
            validation = validate_corpus(xml_path, xsd_path, store)
            nodes, positions, type_counts, source_ref_counts = parse_xml_entities(xml_path)
            rel_counts = Counter(dict(store.db.execute("SELECT type,COUNT(*) FROM relationships GROUP BY type")))
            image_status = write_full_png(
                xml_path, out / "knowledge_graph_visualization.png", nodes, positions,
                type_counts, rel_counts, validation["source_count"])
        finally:
            # Windows refuses to delete the staging file while the connection is
            # open, which would replace the real failure with a cleanup error.
            store.close()
        old_graph_svgs = [
            "knowledge_graph_visualization.svg",
            "knowledge_graph_overview.svg",
            "knowledge_graph_road_network.svg",
            "knowledge_graph_laws_and_signs.svg",
        ]
        removed_svg_files = []
        for filename in old_graph_svgs:
            old_svg = out / filename
            if old_svg.is_file():
                old_svg.unlink()
                removed_svg_files.append(filename)
        remaining_svg_files = [filename for filename in old_graph_svgs if (out / filename).exists()]
        if remaining_svg_files:
            raise RuntimeError(f"Graph SVG cleanup failed: {remaining_svg_files}")
        # Keep the requested deliverables directory limited to the source XML and PNG image.
        for filename in ("knowledge_graph_visualization.html", "knowledge_graph_stats.json",
                         "knowledge_graph.xsd", "README.md"):
            artifact = out / filename
            if artifact.is_file():
                artifact.unlink()
        for artifact in OUT.glob("*.html"):
            artifact.unlink()
    print(json.dumps({"output_dir": str(out), "entity_count": validation["entity_count"],
                      "relationship_count": validation["relationship_count"], "source_count": validation["source_count"],
                      "xml_bytes": validation["xml_bytes"], "validation": validation["xsd_validation"],
                      "image": image_status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
