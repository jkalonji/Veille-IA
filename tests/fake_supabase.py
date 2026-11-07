"""In-memory stand-in for the supabase-py client, covering the query calls the
AI layer cake code makes (select / eq / gte / in_ / order / range, insert,
upsert, update). Enough to run the daily cake steps end to end in tests."""

from types import SimpleNamespace


class _Query:
    def __init__(self, db, table):
        self.db, self.table_name = db, table
        self.filters, self.orders, self.window = [], [], None
        self.action, self.payload, self.conflict = "select", None, None

    # -- building ---------------------------------------------------------
    def select(self, cols="*"):
        return self

    def eq(self, col, value):
        self.filters.append(lambda r: r.get(col) == value)
        return self

    def gte(self, col, value):
        self.filters.append(lambda r: r.get(col) is not None and str(r.get(col)) >= str(value))
        return self

    def in_(self, col, values):
        values = set(values)
        self.filters.append(lambda r: r.get(col) in values)
        return self

    def order(self, col, desc=False):
        self.orders.append((col, desc))
        return self

    def range(self, start, end):
        self.window = (start, end)
        return self

    def insert(self, rows):
        self.action, self.payload = "insert", rows
        return self

    def upsert(self, rows, on_conflict=None):
        self.action, self.payload, self.conflict = "upsert", rows, on_conflict
        return self

    def update(self, values):
        self.action, self.payload = "update", values
        return self

    # -- running ----------------------------------------------------------
    def execute(self):
        rows = self.db.tables.setdefault(self.table_name, [])
        if self.action == "select":
            out = [dict(r) for r in rows if all(f(r) for f in self.filters)]
            for col, desc in reversed(self.orders):
                out.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=desc)
            if self.window:
                out = out[self.window[0]:self.window[1] + 1]
            return SimpleNamespace(data=out)
        payload = self.payload if isinstance(self.payload, list) else [self.payload]
        if self.action == "update":
            hit = [r for r in rows if all(f(r) for f in self.filters)]
            for r in hit:
                r.update(self.payload)
            return SimpleNamespace(data=[dict(r) for r in hit])
        saved = []
        keys = self.conflict.split(",") if self.conflict else None
        for p in payload:
            existing = next((r for r in rows if keys and all(r.get(k) == p.get(k) for k in keys)), None)
            if existing is not None and self.action == "upsert":
                existing.update(p)
                saved.append(dict(existing))
            else:
                row = {"id": self.db.next_id(), **p}
                rows.append(row)
                saved.append(dict(row))
        return SimpleNamespace(data=saved)


class FakeSupabase:
    def __init__(self):
        self.tables: dict[str, list[dict]] = {}
        self._id = 0

    def next_id(self):
        self._id += 1
        return self._id

    def table(self, name):
        return _Query(self, name)
