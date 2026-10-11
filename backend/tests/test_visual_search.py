import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.api.routes import visual_search
from app.rag.engine import RAGAnythingEngine


class VisualSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings = SimpleNamespace(upload_dir=Path(self.tmp.name), max_upload_mb=8, groq_api_key="test-key", groq_vision_model="configured-model")
        self.repository = SimpleNamespace(list_signs=Mock(return_value=[]))
        self.rag = SimpleNamespace(ready=True, query_image=AsyncMock(side_effect=RuntimeError("vision unavailable")))
        self.request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings, repository=self.repository, rag=self.rag)))

    def upload(self):
        data = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(data, format="PNG")
        data.seek(0)
        return UploadFile(data, filename="stop-sign.png", headers=Headers({"content-type": "image/png"}))

    async def test_provider_failure_does_not_guess_from_filename(self):
        provider = Mock()
        provider.chat.completions.create = AsyncMock(side_effect=TimeoutError("permission denied"))
        with patch("groq.AsyncGroq", return_value=provider):
            result = await visual_search(self.request, self.upload())
        self.assertIn("could not verify", result["analysis"])
        self.assertEqual(result["related_signs"], [])
        self.assertEqual(result["engine"], "visual analysis unavailable")
        self.repository.list_signs.assert_not_called()
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    async def test_available_multimodal_analysis_is_preserved(self):
        self.rag.query_image = AsyncMock(return_value="A red octagonal stop sign. Stop before proceeding safely.")
        self.repository.list_signs.return_value = [{"id": "stop", "name": "Stop"}]
        result = await visual_search(self.request, self.upload())
        self.assertEqual(result["engine"], "RAG-Anything multimodal")
        self.assertEqual(result["related_signs"][0]["id"], "stop")
        self.assertNotIn("stop-sign.png", result["analysis"])

    async def test_raw_visual_parser_output_is_not_returned(self):
        self.rag.query_image = AsyncMock(return_value="Image Path: C:\\private\\sign.png\nVisual Analysis: raw")
        self.request.app.state.settings.groq_api_key = ""
        result = await visual_search(self.request, self.upload())
        self.assertNotIn("Image Path", result["analysis"])
        self.assertEqual(result["related_signs"], [])


class UploadedImageEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sign.png"
        self.path.write_bytes(b"encoded by the mocked existing RAG image processor")
        self.processor = SimpleNamespace(_encode_image_to_base64=Mock(return_value="image-data"),
            modal_caption_func=AsyncMock(return_value="A red octagonal stop sign. Stop before proceeding safely."))
        self.engine = RAGAnythingEngine(SimpleNamespace())
        self.engine.rag = SimpleNamespace(modal_processors={"image": self.processor}, aquery_with_multimodal=AsyncMock())

    async def test_visual_evidence_precedes_catalogue_retrieval_without_graph_expansion(self):
        answer = await self.engine.query_image("Explain this sign", self.path)
        self.assertIn("stop sign", answer)
        self.engine.rag.aquery_with_multimodal.assert_not_called()
        self.assertTrue(self.processor.modal_caption_func.call_args.kwargs["_require_visual_evidence"])
        self.assertEqual(self.processor.modal_caption_func.call_args.kwargs["max_completion_tokens"], 600)

    async def test_visual_provider_failure_cannot_become_a_fake_caption(self):
        self.processor.modal_caption_func.side_effect = TimeoutError("vision unavailable")
        with self.assertRaises(TimeoutError):
            await self.engine.query_image("Explain this sign", self.path)
        self.engine.rag.aquery_with_multimodal.assert_not_called()

    async def test_missing_image_and_raw_processing_output_are_not_evidence(self):
        self.assertIsNone(await self.engine.query_image("Explain", self.path.parent / "missing.png"))
        self.processor.modal_caption_func.assert_not_called()
        self.processor.modal_caption_func.return_value = 'Image Path: C:\\private\\sign.png'
        self.assertIsNone(await self.engine.query_image("Explain", self.path))
