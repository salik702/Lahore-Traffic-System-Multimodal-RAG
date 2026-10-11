"""Compatibility entry points; official monetary values never fall back to an LLM/index."""
from .fines import FineDatabase


def official_records(settings):
    db=FineDatabase(settings.data_dir)
    return db.records, {'error':db.error,'valid':db.valid}


async def retrieve_legal(kind,rag,repository,settings):
    # RAG may supply explanations elsewhere, but cannot verify current fine amounts.
    return FineDatabase(settings.data_dir).lookup(kind,{'class':'motorcycle'},'Punjab')


async def explain(result,settings):
    return result.get('ai_explanation','Local model findings require manual review; this is not an official challan.'), 'Evidence-based local explanation'
