<h1 align="center">LexVerse</h1>

> This branch contains the beta LexVerse data environment. For the current LexVerse version, see the [`main` branch](https://github.com/thunlp/LexVerse/tree/main).

<p align="center">
  <a href="examples/lexverse_tutorial.ipynb">📓 Tutorial Notebook</a> ·
  <a href="#use-lexverse">📖 Usage</a>
</p>

---

LexVerse data environment provides a unified interface for Chinese laws, cases, legal questions, concepts, and document templates.

> 💡 **New to LexVerse?** Start with [`examples/lexverse_tutorial.ipynb`](examples/lexverse_tutorial.ipynb) — a runnable Jupyter notebook covering all core APIs with real data.

## Data Environment

The LexVerse data environment is organized into three layers.

<p align="center">
  <img src="img/lexverse-architecture.png" alt="LexVerse architecture diagram" width="100%">
</p>

### Data Collections

LexVerse includes the collections and data sources below. Check each source’s license and attribution terms before sharing the data.

<table>
  <thead><tr><th>Collection</th><th>Description</th><th>Dataset / source</th></tr></thead>
  <tbody>
    <tr><td><code>legal_laws</code></td><td>Laws, regulations, judicial interpretations, and legal articles</td><td><a href="https://github.com/RanKKI/LawRefBook">LawRefBook / Laws</a></td></tr>
    <tr><td><code>legal_cases</code></td><td>Cases and judicial decisions</td><td><a href="https://github.com/FudanDISC/J1Bench">J1Bench (CI, CR)</a><br><a href="https://github.com/Hezhitao2021/SimuCourt">AgentsCourt / SimuCourt</a><br><a href="https://github.com/chidaic/Legal-world">Legal-world</a><br><a href="https://github.com/FudanDISC/MASER">MASER</a><br><a href="https://github.com/oneal2000/JuDGE">JuDGE</a><br><a href="https://github.com/THUIR/LeCaRDv2">LeCaRDv2</a><br><a href="https://github.com/yuwenhan07/MSLR-Bench">MSLR-Bench</a><br><a href="https://github.com/THUlawtech/MUSER">MUSER</a><br><a href="https://github.com/thunlp/LexChain">LexChain</a></td></tr>
    <tr><td><code>legal_qa</code></td><td>Legal questions and scenario data</td><td><a href="https://github.com/FudanDISC/J1Bench">J1Bench (LC, KQ)</a><br><a href="https://www.court.gov.cn/search.html?content=%E6%B3%95%E7%AD%94%E7%BD%91">法答网 (Fadawang)</a><br><a href="https://github.com/SKYLENAGE-AI/DLawBench">DLawBench</a></td></tr>
    <tr><td><code>legal_concepts</code></td><td>Legal terms, concepts, and definitions</td><td><a href="https://terms.legalhub.cn/">中文法律术语汇编</a><br><a href="https://github.com/ownthink/KnowledgeGraphData">OwnThink</a></td></tr>
    <tr><td><code>legal_templates</code></td><td>Document templates and registered assets</td><td><a href="https://law.wkinfo.com.cn/document-templates/list">最高人民法院、中国法院网文书模板</a></td></tr>
  </tbody>
</table>

The data and prebuilt index can be downloaded from Hugging Face.

## Use LexVerse

The Python SDK, CLI, and MCP server provide access to the same collections.

### Python SDK

```python
from lexverse_env import CorpusEnv

with CorpusEnv.open("data", ".lexverse") as env:
    page = env.search_records("legal_laws", query="劳动合同解除", limit=5)
    for item in page["items"]:
        print(env.get_record(item["id"])["record"])
```

### CLI

```bash
lexverse-env search \
  --data-dir ./data \
  --state-dir ./.lexverse \
  --collection legal_cases \
  --query "劳动合同解除" \
  --limit 5
```

### MCP Server

```bash
python -m pip install -e '.[mcp]'
export LEXVERSE_DATA_DIR=/path/to/LexVerse/data
export LEXVERSE_STATE_DIR=/path/to/LexVerse/.lexverse
lexverse-mcp
```

Pkulaw is available as an optional external provider. Set `PKULAW_TOKEN` and
select `provider="pkulaw"` in `search_records` to use it.

## Extend LexVerse

- Add data sources and projections in `lexverse_env/registry.py` and loaders in `lexverse_env/loaders.py`, then rebuild the index.
- Add external providers through the provider interface; `PkulawMcpProvider` is an example.
- Add environment methods before exposing new MCP tools in `lexverse_env/mcp_server.py`.

## License

Dataset licenses and usage restrictions remain governed by their original sources.

## Contact Us

For issues and feature requests, use GitHub Issues. You can also email xieh@tsinghua.edu.cn.
