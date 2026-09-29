<h1 align="center">LexVerse: An Interaction Environment for Chinese Legal Agents</h1>

<p align="center">
  <a href="examples/lexverse_tutorial.ipynb">📓 Tutorial Notebook</a> ·
  <a href="#quick-start">🚀 Quick Start</a> ·
  <a href="#using-lexverse">📖 Usage</a>
</p>

---

LexVerse is an interaction environment for Chinese legal agents. It organizes laws, cases, legal questions, legal concepts, and document templates behind a unified interface for search, retrieval, provenance, and controlled data access.

> 💡 **New to LexVerse?** Start with [`examples/lexverse_tutorial.ipynb`](examples/lexverse_tutorial.ipynb) — a runnable Jupyter notebook covering all core APIs with real data.

## The LexVerse Environment

LexVerse is organized into three layers.

<p align="center">
  <img src="img/lexverse-architecture.png" alt="LexVerse architecture diagram" width="100%">
</p>

## Data Collections

| Collection          | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `legal_laws`      | Laws, regulations, judicial interpretations, and legal articles |
| `legal_cases`     | Cases and judicial decisions                                    |
| `legal_qa`        | Legal questions and scenario data                               |
| `legal_concepts`  | Legal terms, concepts, and definitions                          |
| `legal_templates` | Document templates and registered assets                        |

### Dataset Sources and Attribution

The table below maps each dataset currently used by LexVerse to its online source. The links identify the upstream project, dataset release, or official source page. Each source retains its own license and attribution terms. Check those terms before sharing a copy of the data.

| Collection | Dataset / source                                                                             |
| ---------- | -------------------------------------------------------------------------------------------- |
| QA         | [J1Bench (LC, KQ)](https://github.com/FudanDISC/J1Bench)                                      |
| QA         | [法答网 (Fadawang)](https://www.court.gov.cn/search.html?content=%E6%B3%95%E7%AD%94%E7%BD%91) |
| QA         | [DLawBench](https://github.com/SKYLENAGE-AI/DLawBench)                                        |
| Concepts   | [中文法律术语汇编](https://terms.legalhub.cn/)                                                |
| Concepts   | [OwnThink](https://github.com/ownthink/KnowledgeGraphData)                                    |
| Cases      | [J1Bench (CI, CR)](https://github.com/FudanDISC/J1Bench)                                      |
| Cases      | [AgentsCourt / SimuCourt](https://github.com/Hezhitao2021/SimuCourt)                          |
| Cases      | [Legal-world](https://github.com/chidaic/Legal-world)                                         |
| Cases      | [MASER](https://github.com/FudanDISC/MASER)                                                   |
| Cases      | [JuDGE](https://github.com/oneal2000/JuDGE)                                                   |
| Cases      | [LeCaRDv2](https://github.com/THUIR/LeCaRDv2)                                                 |
| Cases      | [MSLR-Bench](https://github.com/yuwenhan07/MSLR-Bench)                                        |
| Cases      | [MUSER](https://github.com/THUlawtech/MUSER)                                                  |
| Cases      | [LexChain](https://github.com/thunlp/LexChain)                                                |
| Laws       | [LawRefBook / Laws](https://github.com/RanKKI/LawRefBook)                                     |
| Templates  | [最高人民法院、中国法院网文书模板](https://law.wkinfo.com.cn/document-templates/list)         |

For record-level provenance, use the `provenance` field returned by the SDK/API. It identifies the collection, source, relative path, and snapshot used for a record.

## Quick Start

Python 3.10+ is required.

Clone LexVerse and install the package:

```bash
git clone https://github.com/thunlp/LexVerse.git
cd LexVerse
python -m pip install -e .
python -m pip install -U huggingface_hub
```

Download the official data (about 3.2 GB) and its matching prebuilt index.
If a repository requires authentication, run `huggingface-cli login` with an
account that has access before downloading:

```bash
huggingface-cli download thunlp/LexVerse-data \
  --repo-type dataset \
  --local-dir ./data

huggingface-cli download thunlp/LexVerse-index \
  --repo-type dataset \
  --local-dir ./.lexverse/official
```

The official `.lexverse/official` directory is already built for the corresponding
official data release. Users of the official release do not need to run
`lexverse-env build-index`.

```bash
# Check the downloaded data and index
lexverse-env doctor --data-dir ./data --state-dir ./.lexverse
```

The environment can then be used directly:

```python
from lexverse_env import CorpusEnv

with CorpusEnv.open("data", ".lexverse") as env:
    print(env.search_records("legal_laws", query="劳动合同解除", limit=5))
```

## Using LexVerse

### Index Setup and Management

#### Get the Official Index

The official index is published separately from the source data at
[`thunlp/LexVerse-index`](https://huggingface.co/datasets/thunlp/LexVerse-index).
Download it into `.lexverse/official` alongside the matching `data` directory:

```bash
huggingface-cli download thunlp/LexVerse-index \
  --repo-type dataset \
  --local-dir ./.lexverse/official
```

Keep the data and index release versions matched. `manifest.json` binds an
official index to the Hugging Face repository, immutable commit SHA, dataset
revision, and a complete SHA-256 file manifest. A normal open performs a fast
path-and-size check. To verify every file's contents, run:

```bash
lexverse-env doctor \
  --data-dir ./data \
  --state-dir ./.lexverse \
  --verify-data
```

#### Build an Index for Custom Data

Run `build-index` only when adding or changing local data, loaders,
projections, or collections. Register a new source as described in
[Add a Data Source](#add-a-data-source), then build a local index:

```bash
lexverse-env inventory --data-dir ./user_data --state-dir ./.lexverse/user

lexverse-env build-index \
  --data-dir ./user_data \
  --state-dir ./.lexverse/user \
  --source-type local \
  --dataset-id my-legal-data \
  --progress
```

`dataset-revision` is optional for local data. If omitted, LexVerse derives it
from the content manifest. For development, add `--max-records 100` to create a
small partial index.

To build an official index from a Hugging Face dataset, provide its
repository and revision:

```bash
lexverse-env build-index \
  --data-dir ./data \
  --state-dir ./.lexverse/official \
  --source-type huggingface \
  --hf-repo-id thunlp/LexVerse-data \
  --huggingface-commit-sha <40-character-commit-sha> \
  --dataset-revision <revision-used-for-download> \
  --progress
```

LexVerse records every file's relative path, size, and SHA-256 digest. File
modification times are deliberately excluded, so an index remains reusable
after matching data is copied or extracted on another machine.

Official and user files are never mixed. `CorpusEnv.open("data", ".lexverse")`
opens `.lexverse/official` and automatically adds the user layer when both
`user_data` and `.lexverse/user` exist. User records take precedence if the two
layers contain the same record or asset ID. Stable searches are merged globally;
relevance searches interleave results from the user and official layers.

Because user data is mutable, LexVerse fully verifies its SHA-256 manifest when
opening the user layer. An open SDK or MCP session also checks the user-data stat
signature before every local search or read and reruns full verification after a
change. After editing `user_data`, rebuild only `.lexverse/user` and reopen the
environment or restart the MCP server. The official index is unaffected.

### Common Workflow

The standard interaction pattern is:

```text
list_collections
    -> describe_collection
    -> search_records
    -> get_record
    -> read_record_part / get_asset
```

Run these steps in order: search first, then read the records returned by the search. Use `read_record_part` for a large field and `get_asset` only after a template record returns an asset ID.

### How to Use

Python SDK, CLI, and MCP Server expose the same environment operations. Choose one entry point; you do not need to use all three. For a first run, the Python SDK is the easiest way to see each response.

- Use the **Python SDK** to write programs and integrate an Agent.
- Use the **CLI** to inspect data from a terminal.
- Use the **MCP Server** to connect LexVerse to another Agent client.

#### Python SDK

```python
from lexverse_env import CorpusEnv

with CorpusEnv.open("data", ".lexverse") as env:
    print(env.list_collections())
    print(env.describe_collection("legal_laws"))

    page = env.search_records(
        collection="legal_laws",
        query="劳动合同解除",
        limit=5,
    )

    for item in page["items"]:
        record = env.get_record(item["id"])
        print(record["provenance"])
        print(record["record"])
```

Search results contain stable IDs and minimal key fields. Read large fields in bounded slices with a JSON Pointer:

```python
part = env.read_record_part(
    record_id,
    path="/documents/0/content",
    offset=0,
    limit=12000,
)
```

Pass `next_cursor` and `next_offset` back unchanged when continuing a query or a record read.

#### CLI

Search records:

```bash
lexverse-env search \
  --data-dir ./data \
  --state-dir ./.lexverse \
  --collection legal_cases \
  --query "劳动合同解除" \
  --limit 5
```

Read a record returned by search:

```bash
lexverse-env get \
  --data-dir ./data \
  --state-dir ./.lexverse \
  "<record-id>"
```

#### MCP Server

Install the optional MCP dependency and start the stdio server:

```bash
python -m pip install -e '.[mcp]'
export LEXVERSE_DATA_DIR=/path/to/LexVerse/data
export LEXVERSE_STATE_DIR=/path/to/LexVerse/.lexverse
# Optional custom-data layer
export LEXVERSE_USER_DATA_DIR=/path/to/LexVerse/user_data
export LEXVERSE_USER_STATE_DIR=/path/to/LexVerse/.lexverse/user
lexverse-mcp
```

The server exposes the same operations as the SDK:

```text
list_collections       describe_collection
search_records         get_record
read_record_part       get_asset
get_manifest
```

### External Integrations

#### Pkulaw MCP

Pkulaw is an optional external Provider. Set its token and use the environment's semantic APIs; the HTTP/SSE bridge, service routing, official tool names, and parameter conversion are built in:

```python
import os
from lexverse_env import CorpusEnv

os.environ["PKULAW_TOKEN"] = "Bearer <your-token>"

with CorpusEnv.open("data", ".lexverse") as env:
    page = env.search_records(
        collection="legal_laws",
        provider="pkulaw",
        query="劳动合同解除",
        limit=5,
    )

    cases = env.search_records(
        collection="legal_cases",
        provider="pkulaw",
        query="房屋租赁到期后拒退押金",
        limit=3,
    )
```

The default search mode is semantic. Select the documented keyword services with `filters={"search_mode": "keyword"}`. The built-in registry routes record searches across four Pkulaw endpoints:

| Environment operation | Pkulaw tools |
| --- | --- |
| `search_records("legal_laws", ...)` | `search_article` / `get_law_list` |
| `search_records("legal_cases", ...)` | `search_case` / `get_case_list` |

Every registered endpoint has a `PKULAW_<SERVICE>_MCP_ENDPOINT` override, such as `PKULAW_CASE_SEMANTIC_MCP_ENDPOINT`. The original `PKULAW_LAW_MCP_ENDPOINT` remains an alias for the semantic law endpoint. Applications may alternatively pass `pkulaw_endpoints`, `pkulaw_call_tool`, or a complete `pkulaw_provider` to `CorpusEnv.open`. Keep credentials outside the repository.

## Extending LexVerse

### Add a Data Source

Register the source, select a loader, define its projection, and rebuild the index:

```text
JSON/JSONL source
    -> SourceRegistry entry
    -> loader
    -> searchable and filter projections
    -> index build
```

Source registration and projections live primarily in `lexverse_env/registry.py`; loader implementations live in `lexverse_env/loaders.py`.

### Add a Collection

A collection should define:

- a stable collection name;
- file patterns and loader type;
- searchable and filter fields;
- minimal key fields for search responses;
- source and provenance behavior.

New collections should use the same public calls:

```python
env.search_records(collection="new_collection", query="...")
env.get_record(record_id)
```

### Add an External Provider

Implement the Provider contract and keep the public environment API unchanged:

```python
class CustomProvider:
    name = "custom"

    @property
    def available(self) -> bool:
        ...

    def search(self, request):
        ...

    def get(self, record_id):
        ...
```

Use explicit provider selection from an Agent:

```python
env.search_records(
    collection="legal_cases",
    provider="custom",
    query="合同纠纷",
)
```

External IDs should be namespaced so they cannot be confused with local records. `PkulawMcpProvider` is the reference implementation.

### Add an MCP Tool

Add the corresponding Python environment method first, then expose it through `lexverse_env/mcp_server.py`. Keep argument validation and error conversion in the environment layer, and keep MCP stdout reserved for JSON-RPC messages.

## Design Boundaries

- Data access is limited to registered collections and data roots.
- Callers use stable record and asset IDs instead of physical paths.
- Large records and attachments are read through bounded interfaces.
- Search and record responses retain source and snapshot information.
- External Providers are called only when explicitly configured and selected.

## Project Structure

```text
LexVerse/
├── data/              # Official data; do not add custom files here
├── user_data/         # Optional custom data
├── lexverse_env/      # SDK, index, Providers, and MCP Server
├── examples/          # Runnable tutorial notebook
├── tests/             # Unit tests
├── img/               # Architecture diagrams
└── .lexverse/
    ├── official/      # Downloaded official index
    └── user/          # Generated custom index
```

## License

Dataset licenses and usage restrictions remain governed by the original sources; preserve `provenance` when using records.

## Contact Us

For technical issues and feature requests, please use GitHub Issues.
If you have any questions, feedback, or would like to get in touch, please feel free to reach out to us via email at xieh@tsinghua.edu.cn.
