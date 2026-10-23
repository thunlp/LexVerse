"""A disposable FTS5 cache with collection-wide BM25 statistics."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from filelock import FileLock


def create_tokenizer():
    import jieba
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def tokenize(tokenizer, text):
    return [word.strip() for word in tokenizer.cut(text) if word.strip() and any(c.isalnum() for c in word)]


class KeywordSearch:
    def __init__(self, indexes, cache_dir: Path):
        self.indexes = sorted(indexes, key=lambda index: index.index_hash)
        identity = [1, [(index.index_hash, index.manifest['files']['documents.sqlite'])
                        for index in self.indexes]]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path = cache_dir / (key + '.sqlite')
        self.tokenizer = create_tokenizer()

    def prepare(self):
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + '.lock'):
            if self.path.exists():
                return
            with tempfile.TemporaryDirectory(dir=self.path.parent) as temporary:
                staged = Path(temporary) / 'keywords.sqlite'
                self._build(staged)
                staged.rename(self.path)

    def _build(self, path):
        with closing(sqlite3.connect(path)) as db:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY, index_hash TEXT, chunk_id TEXT, '
                       'kind TEXT, title TEXT, article TEXT, page INTEGER)')
            db.execute('CREATE VIRTUAL TABLE keywords USING fts5(tokens)')
            ordinal = 0
            for index in self.indexes:
                rows = index.db.execute(
                    "SELECT chunks.id,keywords.tokens,json_extract(metadata,'$.kind'),"
                    "json_extract(metadata,'$.title'),json_extract(metadata,'$.article'),"
                    "json_extract(metadata,'$.page') FROM keywords JOIN chunks ON chunks.id=keywords.id"
                )
                while batch := rows.fetchmany(1000):
                    db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?,?)',
                                   [(ordinal + n, index.index_hash, row[0], *row[2:])
                                    for n, row in enumerate(batch, 1)])
                    db.executemany('INSERT INTO keywords(rowid,tokens) VALUES (?,?)',
                                   [(ordinal + n, row[1]) for n, row in enumerate(batch, 1)])
                    ordinal += len(batch)
            db.execute('CREATE INDEX by_source ON chunks(index_hash)')
            db.execute("INSERT INTO keywords(keywords) VALUES ('optimize')")
            db.commit()

    def search(self, query, index_hashes, filters, limit):
        terms = list(dict.fromkeys(tokenize(self.tokenizer, query)))[:100]
        if not terms or not index_hashes:
            return []
        self.prepare()
        expression = ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)
        # Column names originate only from validated KnowledgeFilters.
        conditions = ''.join(f' AND chunks.{key}=?' for key in filters)
        placeholders = ','.join('?' for _ in index_hashes)
        with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            return db.execute(
                'SELECT index_hash,chunk_id,-bm25(keywords) FROM keywords JOIN chunks ON chunks.id=keywords.rowid '
                f'WHERE keywords MATCH ? AND index_hash IN ({placeholders})' + conditions +
                ' ORDER BY bm25(keywords),index_hash,chunk_id LIMIT ?',
                [expression, *index_hashes, *filters.values(), limit],
            ).fetchall()
