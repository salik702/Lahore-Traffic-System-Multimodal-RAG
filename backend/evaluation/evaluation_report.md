# Lahore Traffic System Multimodal RAG
## Retriever Evaluation Report

## Methodology
This report evaluates the existing persisted LightRAG chunk index without rebuilding it. Queries use verified structured-corpus questions, and relevance labels point to exact persisted chunk IDs.

## Configuration
- Dataset examples: 18
- Retrieval cutoff: 10
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Embedding dimensions: 384
- LangSmith: Enabled (lahore-traffic-retriever-evaluation)
## Aggregate Metrics

| Metric | Value |
|---|---:|
| Average precision@1 | 0.6111 |
| Average precision@3 | 0.2963 |
| Average precision@5 | 0.1889 |
| Average precision@10 | 0.1000 |
| Average recall@1 | 0.5833 |
| Average recall@3 | 0.8333 |
| Average recall@5 | 0.8889 |
| Average recall@10 | 0.9444 |
| Average mrr | 0.7228 |
| Average ndcg@3 | 0.7368 |
| Average ndcg@5 | 0.7583 |
| Average ndcg@10 | 0.7768 |
| Average hit_rate | 0.9444 |
| Average empty_retrieval | 0.0000 |

## Metric Definitions
- **Precision@K:** fraction of the first K unique retrieved chunks that are relevant.
- **Recall@K:** fraction of labelled relevant chunks retrieved in the first K results.
- **MRR:** reciprocal rank of the first relevant result.
- **NDCG@K:** graded relevance with rank discount through K.
- **Hit rate:** fraction of queries with at least one relevant result.
- **Empty retrieval:** fraction of queries returning no chunks.

## Limitations
- The current verified dataset contains 18 examples and is not a replacement for broader human-reviewed coverage.
- Context precision and context recall require answer/context annotations beyond the current chunk relevance labels.
- LangSmith traces and dataset are available under the project 'rag-retriever-evaluation'.

## Per-query Results

| # | Category | MRR | Hit rate | Retrieved relevant |
|---:|---|---:|---:|---:|
| 1 | rule | 1.000 | 1 | 1 |
| 2 | rule | 1.000 | 1 | 1 |
| 3 | rule | 1.000 | 1 | 1 |
| 4 | rule | 1.000 | 1 | 1 |
| 5 | rule | 1.000 | 1 | 1 |
| 6 | fine | 0.500 | 1 | 1 |
| 7 | fine | 1.000 | 1 | 1 |
| 8 | fine | 1.000 | 1 | 1 |
| 9 | fine | 1.000 | 1 | 1 |
| 10 | fine | 1.000 | 1 | 1 |
| 11 | sign | 0.333 | 1 | 1 |
| 12 | sign | 0.200 | 1 | 1 |
| 13 | sign | 0.143 | 1 | 1 |
| 14 | sign | 0.500 | 1 | 1 |
| 15 | sign | 0.333 | 1 | 1 |
| 16 | multi-doc | 1.000 | 1 | 2 |
| 17 | multi-doc | 1.000 | 1 | 1 |
| 18 | out-of-domain | 0.000 | 0 | 0 |
