import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
from .db import Db, now, dumps, loads
from .hashing import content_id, detect_kind

class SourceEvicted(Exception):
    """Source was known but its bytes were evicted; client must re-upload (HTTP 410)."""

@dataclass
class SourceRecord:
    source_id: str
    path: Path
    kind: str
    width: Optional[int]
    height: Optional[int]
    exif: Optional[dict]
    rrdata: Optional[dict]

class SourceStore:
    def __init__(self, db: Db, root: Path, max_bytes: int):
        self.db = db
        self.root = root
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, content: bytes, filename: str, exif: dict | None = None,
            rrdata: dict | None = None, client_width: int | None = None,
            client_height: int | None = None) -> dict:
        sid = content_id(content)
        kind = detect_kind(filename)
        row = self.db.query_one("SELECT * FROM sources WHERE id=?", (sid,))
        if row and not row["evicted"] and Path(row["path"]).exists():
            # dedup hit; refresh sidecars if newly provided
            self.db.execute(
                "UPDATE sources SET last_used=?, exif=COALESCE(?,exif), rrdata=COALESCE(?,rrdata) WHERE id=?",
                (now(), dumps(exif) if exif else None, dumps(rrdata) if rrdata else None, sid))
            return {"source_id": sid, "kind": row["kind"],
                    "width": row["width"], "height": row["height"]}

        ext = os.path.splitext(filename)[1].lower() or ".bin"
        path = self.root / f"{sid}{ext}"
        path.write_bytes(content)

        width, height = client_width, client_height
        if kind != "raw":
            try:
                with Image.open(io.BytesIO(content)) as im:
                    width, height = im.size
            except Exception:
                pass

        self.db.execute(
            "INSERT OR REPLACE INTO sources(id,path,kind,width,height,exif,rrdata,size,created,last_used,evicted)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,0)",
            (sid, str(path), kind, width, height,
             dumps(exif) if exif else None, dumps(rrdata) if rrdata else None,
             len(content), now(), now()))
        self._enforce_limit()
        return {"source_id": sid, "kind": kind, "width": width, "height": height}

    def get(self, source_id: str) -> Optional[SourceRecord]:
        row = self.db.query_one("SELECT * FROM sources WHERE id=?", (source_id,))
        if row is None:
            return None
        if row["evicted"] or not Path(row["path"]).exists():
            raise SourceEvicted(source_id)
        self.db.execute("UPDATE sources SET last_used=? WHERE id=?", (now(), source_id))
        return SourceRecord(source_id=source_id, path=Path(row["path"]), kind=row["kind"],
                            width=row["width"], height=row["height"],
                            exif=loads(row["exif"]), rrdata=loads(row["rrdata"]))

    def _enforce_limit(self):
        rows = self.db.query(
            "SELECT id,path,size FROM sources WHERE evicted=0 ORDER BY last_used ASC")
        total = sum(r["size"] for r in rows)
        for r in rows:
            if total <= self.max_bytes:
                break
            try:
                path = Path(r["path"])
                path.unlink(missing_ok=True)
                # Clean up sidecars derived from this source (e.g. the
                # per-source depth-map cache "<path>.depth.png" from the
                # mask capability's carve path) -- otherwise they'd dangle
                # forever pointing at an evicted source.
                for sidecar in path.parent.glob(path.name + ".*"):
                    try:
                        sidecar.unlink()
                    except OSError:
                        pass
            finally:
                self.db.execute("UPDATE sources SET evicted=1 WHERE id=?", (r["id"],))
                total -= r["size"]
