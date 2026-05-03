# Embedding Model Options for Incident Similarity

**Context:** Project already wires `langchain_ollama.OllamaEmbeddings` and `langchain_openai.AzureOpenAIEmbeddings` through a `ProviderKind` switch in `src/orchestrator/llm.py`. Goal: pick a dev-time embedder that is free, swap-clean to Azure `text-embedding-3-small` (1536) or `-3-large` (3072), and survives an air-gapped final deployment.

**Azure target dims:** 3-small = 1536, 3-large = 3072. Both support `dimensions` param so any `>=1024` source vector can later be Matryoshka-truncated server-side once on Azure.

## 1. Local Options (work in dev AND air-gapped prod)

| Model | Dim | Ctx | LangChain class | License | Notes |
|---|---|---|---|---|---|
| **Ollama `bge-m3`** | **1024** | 8K | `langchain_ollama.OllamaEmbeddings` | MIT (BAAI) | Strongest open multilingual model in the 16GB-class; 567M params, 1.2GB pull. Multi-functional (dense+sparse+colbert). |
| Ollama `nomic-embed-text` v1.5 | 768 | 2K (8K trained) | same | Apache-2.0 | 137M params, 274MB. Beats `text-embedding-3-small` per Nomic. Smallest dim hurts re-use. |
| Ollama `mxbai-embed-large` | 1024 | 512 | same | Apache-2.0 | 335M, MTEB-SOTA-for-size (Mar 2024). Tiny 512-token ctx is the catch. |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | `langchain_huggingface.HuggingFaceEmbeddings` | Apache-2.0 | 80MB, fastest on CPU. Dim too small for Azure swap. |
| `BAAI/bge-small-en-v1.5` | 384 | 512 | same | MIT | Fast CPU baseline; same dim issue. |
| `BAAI/bge-large-en-v1.5` | **1024** | 512 | same | MIT | Strong English-only baseline; 1.3GB. |

Sources: [ollama.com/library/bge-m3](https://ollama.com/library/bge-m3), [nomic-embed-text](https://ollama.com/library/nomic-embed-text), [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large).

## 2. Free Cloud APIs with First-Class LangChain Packages

| Provider | Model | Dim | Free tier | LangChain import | Source |
|---|---|---|---|---|---|
| **Voyage AI** | `voyage-3.5` / `voyage-4` | **1024** (also 256/512/2048) | **200M tokens free** per account, one-off | `from langchain_voyageai import VoyageAIEmbeddings` | [docs.voyageai.com/docs/pricing](https://docs.voyageai.com/docs/pricing) |
| **Cohere** | `embed-v4.0` | 256/512/1024/**1536** (default) | Trial key: 2,000 inputs/min, no daily cap stated; non-commercial only | `from langchain_cohere import CohereEmbeddings` | [docs.cohere.com/docs/rate-limits](https://docs.cohere.com/docs/rate-limits) |
| **Cohere** | `embed-english-v3.0` | 1024 | same | same | same |
| **Mistral** | `mistral-embed` | 1024 | "Experiment" tier free w/ phone verification | `from langchain_mistralai import MistralAIEmbeddings` | [docs.mistral.ai/.../text_embeddings](https://docs.mistral.ai/studio-api/knowledge-rag/embeddings/text_embeddings/) |
| **Jina** | `jina-embeddings-v3` | 1024 (Matryoshka 32-1024) | **10M tokens** auto-issued on signup | `from langchain_community.embeddings import JinaEmbeddings` | [jina.ai/embeddings](https://jina.ai/embeddings/) FAQ |
| **HuggingFace Inference Providers** | any HF embedding model | varies | **$0.10/mo** routed credits (Free tier) | `from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings` | [huggingface.co/.../pricing](https://huggingface.co/docs/inference-providers/pricing) |
| **Together AI** | `BAAI/bge-large-en-v1.5`, `togethercomputer/m2-bert-80M-8k-retrieval` | 1024 / 768 | $1 signup credit; not a sustained free tier | `from langchain_together import TogetherEmbeddings` | [together.ai/pricing](https://www.together.ai/pricing) |

Notes: HF Inference free credits = $0.10/mo (effectively trivial — ~few hundred req); good for one-off probes, not iteration. Together's free credit is a marketing one-off, not a free tier.

## 3. Dim-compatibility for Azure swap

Azure `text-embedding-3-small/large` accept a `dimensions` request parameter that Matryoshka-truncates server-side, so re-use of the **vector store schema** only requires picking a **fixed dim now** and matching it later. No source listed produces 1536 natively except **Cohere `embed-v4.0` (default 1536)** — exact byte-for-byte clean swap to `3-small`. Everything else (1024) requires either re-embedding on switch or using Azure with `dimensions=1024`. Re-embedding is anyway the right move (different model = different semantics), so don't optimize for "no-recompute".

## 4. Recommendation

**Wire Ollama `bge-m3` first.** Reasons, in order:

1. **Zero new infra** — Ollama is already running; just `ollama pull bge-m3`. Existing `OllamaEmbeddings` provider branch needs no code change.
2. **Same-binary dev-and-prod path** — air-gapped target keeps working without a second adapter.
3. **1024 dim** matches the most likely Azure target (`3-small` with `dimensions=1024`); the schema migration story is one config flip.
4. **Quality** — bge-m3 is the strongest open embed model that fits a 16GB box (1.2GB, 567M params, 8K ctx, multi-lingual).

**Add Voyage AI as the cloud "second opinion" lane** (200M tokens free, one-off but huge), reachable via `langchain_voyageai.VoyageAIEmbeddings`. Useful for A/B-comparing local vs hosted recall without burning Azure credit. Add a `voyage` branch alongside `ollama` / `azure_openai` in `_build_embedding`.

**Avoid for dev-first:** HF Inference (credits trivially small), Together (no real free tier), Mistral (rate-limited free tier requires phone), Jina (10M tokens burns fast on incident corpora).

**Trade-off being honest about:** picking bge-m3 means switching to Azure later *will* require a re-embed of the historical incident corpus. That's correct anyway — different model, different vector space — so the "no-recompute" goal is a mirage. Plan the migration as a one-time backfill, not a config flip.
