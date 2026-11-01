from __future__ import annotations

import io

import fitz
import pytest

from rag_lab.core import Passage, RAGService, clean_filename, extract_sections, lexical_rescore, split_text
from rag_lab.evaluate import retrieval_metrics
from rag_lab.settings import Settings


class DummyEmbedder:
    dimension = 3

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float("priority" in text.lower()), float("laptop" in text.lower()), 0.1] for text in texts]


class TenantStore:
    def __init__(self):
        self.rows: dict[tuple[str, str], list[Passage]] = {}

    def replace(self, *, tenant, source_id, file_name, rows):
        self.rows[(tenant, source_id)] = [Passage(id, content, file_name, page, 0.88)
                                          for id, page, content, _ in rows]

    def search(self, *, tenant, vector, limit):
        return [passage for (owner, _), group in self.rows.items() if owner == tenant
                for passage in group][:limit]


def test_chunks_are_overlapping_and_deterministic():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = split_text(words, chunk_words=10, overlap_words=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks == split_text(words, chunk_words=10, overlap_words=3)
    assert split_text("  ") == []
    with pytest.raises(ValueError):
        split_text(words, chunk_words=3, overlap_words=3)


def test_pdf_preserves_pages():
    pdf = fitz.open()
    for line in ["First page", "Second page"]:
        page = pdf.new_page()
        page.insert_text((40, 50), line)
    sections = extract_sections("guide.pdf", pdf.tobytes())
    assert [(item.page, item.text.strip()) for item in sections] == [(1, "First page"), (2, "Second page")]


def test_extension_guard_and_filename_sanitization():
    assert clean_filename(r"C:\docs\policy.md") == "policy.md"
    assert extract_sections("note.md", b"Hello")[0].text == "Hello"
    with pytest.raises(ValueError):
        extract_sections("payload.exe", b"ignored")
    with pytest.raises(ValueError):
        extract_sections("blank.pdf", b"ignored")


def test_tenant_isolation_and_document_replacement():
    settings = Settings(tenant_api_keys_json='{"org_a":"twelve-characters-a","org_b":"twelve-characters-b"}')
    store = TenantStore()
    service = RAGService(settings, embedder=DummyEmbedder(), store=store)
    a = service.ingest("org_a", "policy.md", b"Priority tickets get an answer quickly.")
    b = service.ingest("org_b", "policy.md", b"Laptops are shipped after onboarding.")
    assert a["source_id"] != b["source_id"]
    assert {x.file_name for x in service.search("org_a", "priority")} == {"policy.md"}
    assert all("Laptops" not in x.content for x in service.search("org_a", "priority"))
    first = service.search("org_a", "priority")[0].id
    service.ingest("org_a", "policy.md", b"Priority tickets get an answer quickly.")
    assert service.search("org_a", "priority")[0].id == first
    service.ingest("org_a", "policy.md", b"A new document replaces the old text.")
    assert all("Priority" not in x.content for x in service.search("org_a", "priority"))


def test_retrieval_only_mode_has_grounded_sources():
    service = RAGService(Settings(), embedder=DummyEmbedder(), store=TenantStore())
    service.ingest("org", "handbook.md", b"Priority one incidents receive a 15 minute initial response.")
    reply = service.ask("org", "priority one")
    assert reply["mode"] == "retrieval_only"
    assert reply["sources"][0]["file_name"] == "handbook.md"
    assert reply["sources"][0]["ref"] == "[1]"
    assert not service.ask("different-org", "priority one")["sources"]


def test_lexical_reranker_and_evaluation():
    hits = [Passage("a", "unrelated paragraph", "wrong.md", 1, 0.9),
            Passage("b", "priority incident response", "sla.md", 1, 0.9)]
    assert lexical_rescore("priority incident", hits)[0].file_name == "sla.md"
    scores = retrieval_metrics([{"expected_sources": ["sla.md"]}, {"expected_sources": ["policy.md"]}],
                               [["sla.md", "wrong.md"], ["wrong.md", "policy.md"]])
    assert scores == {"recall_at_k": 1.0, "mrr_at_k": 0.75, "queries": 2}
