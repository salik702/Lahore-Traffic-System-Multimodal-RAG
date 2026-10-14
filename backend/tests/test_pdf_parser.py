import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from app.rag import pdf_parser
from app.rag.engine import (
    RAGAnythingEngine,
    caption_only_image_payload,
    mineru_cli_supports_legacy_flags,
)


def build_pdf(path: Path, pages: list[str], *, image_pages: dict[int, bytes] | None = None) -> Path:
    document = pymupdf.open()
    for index, body in enumerate(pages):
        page = document.new_page()
        page.insert_text((72, 100), body, fontsize=11)
        if image_pages and index in image_pages:
            page.insert_image(pymupdf.Rect(72, 200, 372, 400), stream=image_pages[index])
    document.save(path)
    document.close()
    return path


def png_bytes(width: int = 400, height: int = 300) -> bytes:
    from PIL import Image

    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class ParserTests(unittest.TestCase):
    def test_extracts_text_pages_with_page_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = build_pdf(Path(folder) / "text.pdf", ["First page body", "Second page body"])
            blocks = pdf_parser.parse_pdf(pdf, folder)
        text = [block for block in blocks if block["type"] == "text"]
        self.assertEqual(len(text), 2)
        self.assertEqual([block["page_idx"] for block in text], [0, 1])
        self.assertIn("First page body", text[0]["text"])

    def test_extracts_content_figures_but_skips_page_scans(self):
        with tempfile.TemporaryDirectory() as folder:
            # A small in-flow figure is a document figure; a full-page image is a
            # scan of the page and would only duplicate the text already indexed.
            pdf = build_pdf(Path(folder) / "figures.pdf", ["Figure page"], image_pages={0: png_bytes()})
            blocks = pdf_parser.parse_pdf(pdf, folder)
            figures = [block for block in blocks if block["type"] == "image"]
            self.assertEqual(len(figures), 1)
            self.assertTrue(Path(figures[0]["img_path"]).is_file())

    def test_skips_icons_below_the_figure_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = build_pdf(Path(folder) / "icons.pdf", ["Icon page"], image_pages={0: png_bytes(64, 64)})
            blocks = pdf_parser.parse_pdf(pdf, folder)
        self.assertEqual([block for block in blocks if block["type"] == "image"], [])

    def test_empty_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            blank = Path(folder) / "blank.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(blank)
            document.close()
            with self.assertRaises(ValueError):
                pdf_parser.parse_pdf(blank, folder)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            pdf_parser.parse_pdf("does-not-exist.pdf")

    def test_parser_class_satisfies_the_raganything_protocol(self):
        from raganything.parser import Parser

        cls = pdf_parser.build_parser_class()
        self.assertTrue(issubclass(cls, Parser))
        instance = cls()
        self.assertTrue(instance.check_installation())
        with self.assertRaises(NotImplementedError):
            instance.parse_image("anything.png")
        with self.assertRaises(NotImplementedError):
            instance.parse_document("notes.docx")

    def test_raganything_options_are_forwarded_without_crashing(self):
        # RAG-Anything passes its own parsing options straight through to the
        # parser; unknown keys must be dropped, not raised.
        with tempfile.TemporaryDirectory() as folder:
            pdf = build_pdf(Path(folder) / "kwargs.pdf", ["Forwarded options"])
            blocks = pdf_parser.build_parser_class()().parse_pdf(
                pdf, folder, method="auto", parse_method="auto",
                backend="vlm-http-client", source="huggingface", timeout=30)
        self.assertTrue(any(block["type"] == "text" for block in blocks))


class ParserResolutionTests(unittest.TestCase):
    def engine(self, **overrides):
        return RAGAnythingEngine(SimpleNamespace(**{"rag_parser": "mineru", **overrides}))

    def test_mineru_is_kept_when_the_cli_matches(self):
        engine = self.engine()
        engine.resolve_parser()
        # The probe result depends on the machine; only the decision must be valid.
        supported, _ = mineru_cli_supports_legacy_flags()
        self.assertEqual(engine.resolve_parser(), "mineru" if supported else "pymupdf")
        self.assertTrue(engine.parser_note)

    def test_explicit_pymupdf_is_honoured(self):
        engine = self.engine(rag_parser="pymupdf")
        self.assertEqual(engine.resolve_parser(), "pymupdf")

    def test_other_parsers_are_passed_through(self):
        engine = self.engine(rag_parser="docling")
        self.assertEqual(engine.resolve_parser(), "docling")


class CaptionOnlyImagePayloadTests(unittest.TestCase):
    PROMPT = ("Image Content Analysis:\nImage Path: {path}\nCaptions: {caption}\n"
              "Footnotes: None\n\nVisual Analysis: ")

    def payload(self, path: str, caption: str) -> dict:
        return json.loads(caption_only_image_payload(self.PROMPT.format(path=path, caption=caption), None))

    def test_document_figures_are_named_after_their_source_file(self):
        payload = self.payload("data/processed/documents/doc/images/p0011_x161.png", "Colour and fasten seat belt")
        self.assertEqual(payload["entity_info"]["entity_name"], "document_figure_p0011_x161")
        self.assertIn("Colour and fasten seat belt", payload["detailed_description"])

    def test_absent_captions_are_not_treated_as_text(self):
        payload = self.payload("data/x/images/p0002_x9.png", "None")
        self.assertNotIn("None", payload["detailed_description"])

    def test_verified_photographs_keep_their_provenance_identity(self):
        payload = self.payload("data/processed/accidents/images/pk-acc-img-004.jpg",
                               "Verified photograph of the Rawalpindi truck accident; CC BY-SA 4.0")
        self.assertTrue(payload["entity_info"]["entity_name"].startswith("verified_accident_image_"))

    def test_no_visual_analysis_is_claimed_for_document_figures(self):
        payload = self.payload("data/x/images/p1.png", "Stop line marking")
        self.assertIn("not analysed", payload["detailed_description"])


class MineruProbeTests(unittest.TestCase):
    def test_probe_never_raises(self):
        supported, reason = mineru_cli_supports_legacy_flags()
        self.assertIsInstance(supported, bool)
        self.assertTrue(reason)


class _FakeStatus:
    """Minimal ``doc_status`` stand-in for the pre-insert bookkeeping."""

    def __init__(self):
        self.lookups = []
        self.deleted = []

    async def get_by_id(self, doc_id):
        self.lookups.append(doc_id)
        return None

    async def delete(self, doc_id):
        self.deleted.append(doc_id)


class _FailingStatus(_FakeStatus):
    async def get_by_id(self, doc_id):
        await super().get_by_id(doc_id)
        return {"doc_id": doc_id, "status": "failed"}


class _FakeProcessor:
    @staticmethod
    def _generate_content_based_doc_id(content):
        return "doc-" + str(len(content))


class ParserProtocolTests(unittest.TestCase):
    def test_raganything_exposes_the_content_id_helper(self):
        # prepare_document derives the document id through this method; it comes
        # from a mixin, not from an attribute, so a rename breaks silently.
        from raganything import RAGAnything

        self.assertTrue(hasattr(RAGAnything, "_generate_content_based_doc_id"))


class DocumentStatusTests(unittest.IsolatedAsyncioTestCase):
    """A document is only 'ingested' when doc_status says so, and the enum must
    serialize to its value so that comparison works."""

    async def test_status_enum_is_normalized_to_its_value(self):
        from lightrag.base import DocStatus

        engine = RAGAnythingEngine(SimpleNamespace())
        engine.rag = SimpleNamespace(lightrag=SimpleNamespace(
            doc_status=SimpleNamespace(get_by_id=_returning(
                {"doc_id": "doc-1", "status": DocStatus.PROCESSED, "chunks_count": 3})),
        ))
        status = await engine.document_status("doc-1")
        self.assertEqual(status["status"], "processed")
        self.assertEqual(status["chunks_count"], 3)

    async def test_missing_row_is_reported(self):
        engine = RAGAnythingEngine(SimpleNamespace())
        engine.rag = SimpleNamespace(lightrag=SimpleNamespace(
            doc_status=SimpleNamespace(get_by_id=_returning(None)),
        ))
        self.assertEqual((await engine.document_status("doc-1"))["status"], "missing")

    async def test_processed_without_chunks_is_not_treated_as_persisted(self):
        # LightRAG marks a document processed even when extraction raised and
        # nothing was written; only a real chunk count proves an insert happened.
        from lightrag.base import DocStatus

        engine = RAGAnythingEngine(SimpleNamespace())
        engine.rag = SimpleNamespace(lightrag=SimpleNamespace(
            doc_status=SimpleNamespace(get_by_id=_returning(
                {"doc_id": "doc-1", "status": DocStatus.PROCESSED, "chunks_count": None})),
        ))
        status = await engine.document_status("doc-1")
        self.assertEqual(status["status"], "processed")
        self.assertFalse(status["persisted"])

    async def test_processed_with_chunks_is_persisted(self):
        from lightrag.base import DocStatus

        engine = RAGAnythingEngine(SimpleNamespace())
        engine.rag = SimpleNamespace(lightrag=SimpleNamespace(
            doc_status=SimpleNamespace(get_by_id=_returning(
                {"doc_id": "doc-1", "status": DocStatus.PROCESSED, "chunks_count": 12})),
        ))
        self.assertTrue((await engine.document_status("doc-1"))["persisted"])


def _returning(value):
    async def get_by_id(doc_id):
        return value

    return get_by_id


class PrepareDocumentTests(unittest.IsolatedAsyncioTestCase):
    """``prepare_document`` runs before every insert and must resolve an id alone."""

    def build_engine(self, folder, status):
        engine = RAGAnythingEngine(SimpleNamespace(data_dir=Path(folder)))
        engine.parser = "pymupdf"
        engine.rag = SimpleNamespace(
            doc_parser=pdf_parser.build_parser_class()(),
            _generate_content_based_doc_id=_FakeProcessor()._generate_content_based_doc_id,
            lightrag=SimpleNamespace(doc_status=status, adelete_by_doc_id=_record([])),
        )
        return engine

    async def test_resolves_an_id_without_touching_the_llm(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = build_pdf(Path(folder) / "prep.pdf", ["Section 39 sets the speed limit."])
            status = _FakeStatus()
            engine = self.build_engine(folder, status)
            doc_id = await engine.prepare_document(pdf)
            self.assertEqual(doc_id, "doc-1")
            self.assertEqual(status.lookups, ["doc-1"])

    async def test_existing_partial_insert_is_cleared_first(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = build_pdf(Path(folder) / "prep.pdf", ["Section 39 sets the speed limit."])
            deleted = []
            status = _FailingStatus()
            engine = self.build_engine(folder, status)
            engine.rag.lightrag.adelete_by_doc_id = _record(deleted)
            doc_id = await engine.prepare_document(pdf)
            # A stale row is exactly what makes LightRAG skip a rerun.
            self.assertEqual(doc_id, "doc-1")
            self.assertEqual(deleted, ["doc-1"])
            self.assertEqual(status.deleted, ["doc-1"])


def _record(sink):
    async def delete(target):
        sink.append(target)
        return SimpleNamespace(status="success")

    return delete


if __name__ == "__main__":
    unittest.main()