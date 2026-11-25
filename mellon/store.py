"""One node cache shared by every Mellon server that points at the same store. MELLON_STORE is a folder or an fsspec
URL (s3://<bucket>/<prefix>, its endpoint and key in the environment). A node's outputs are nodes/<key>.json under it,
its files under nodes/<key>/. A file in a value is {"file": "<ref>"}, a path under the root: one another program put
there, or a stored node's own, such as nodes/<key>/depth_videos/0.mp4. MELLON_NODES names a JSON file of each node
folder's revision, which `python mellon/store.py blocks/*` prints. A value naming a file on this machine, such as a file
field's path under Mellon's work folder, keys by the file's content: the same path with other bytes is another key, and
the same bytes under another name or on another host are the same one."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import fsspec
import rfc8785
from fsspec.implementations.local import LocalFileSystem

RUNNER = {
    "device",
    "auto_offload",
    "trust_remote_code",
    "load_block_button",
}  # where it runs, not what it makes
MELLON = "mellon@5fd24292"  # the revision of a node that is not a folder: Mellon's own, at our fork's base
FILES = ("video", "image", "file")


_hashes: dict = {}  # a file's SHA-256 by (path, size, mtime), so an unchanged file is read once per process


def content(value, work_dir: Path):
    """A param's value with each string that names a file here (relative to work_dir, or absolute) as {"sha256": …}."""
    if isinstance(value, list):
        return [content(v, work_dir) for v in value]
    if not isinstance(value, str) or not value or "://" in value:
        return value
    path = Path(value) if Path(value).is_absolute() else work_dir / value
    try:
        stat = path.stat()
    except (OSError, ValueError):
        return value
    if not path.is_file():
        return value
    seen = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if seen not in _hashes:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        _hashes[seen] = digest.hexdigest()
    return {"sha256": _hashes[seen]}


def graph_keys(nodes: dict, revisions: dict, work_dir: str | Path = ".") -> dict:
    """Every node's key: SHA-256 of the JCS of {module, action, revision, params}; a link hashes its source's key, and
    a file named here hashes its content."""
    keys = {}

    def key(node_id):
        if node_id not in keys:
            node, params = nodes[node_id], {}
            for name, p in node["params"].items():
                if name in RUNNER or p.get("display", "").startswith("ui_"):
                    continue
                params[name] = (
                    {"key": key(p["sourceId"]), "sourceKey": p["sourceKey"]}
                    if p.get("sourceId")
                    else {"value": content(p.get("value"), Path(work_dir))}
                )
            repo_id = node["params"].get("repo_id", {}).get("value")
            body = {
                "module": node["module"],
                "action": node["action"],
                "revision": revisions.get(repo_id, MELLON),
                "params": params,
            }
            keys[node_id] = hashlib.sha256(rfc8785.dumps(body)).hexdigest()
        return keys[node_id]

    for node_id in nodes:
        key(node_id)
    return keys


class Store:
    def __init__(self, url: str | None, local: str | Path):
        self.fs, self.root = fsspec.core.url_to_fs(url) if url else (None, None)
        self.local = Path(local)  # where this machine keeps the remote files it has read

    def exists(self, key: str) -> bool:
        return bool(self.fs) and self.fs.exists(f"{self.root}/nodes/{key}.json")

    def refs(self, key: str) -> dict | None:
        """The stored outputs as written, each file as {"file": ref}."""
        if not (self.fs and self.exists(key)):
            return None
        return json.loads(self.fs.cat_file(f"{self.root}/nodes/{key}.json"))

    def resolve(self, value):
        """A value with each {"file": ref} as a path on this machine, read from the store once."""
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if not (isinstance(value, dict) and set(value) == {"file"} and self.fs):
            return value
        if isinstance(self.fs, LocalFileSystem):
            return f"{self.root}/{value['file']}"
        local = self.local / value["file"]
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            self.fs.get_file(f"{self.root}/{value['file']}", str(local))
        return str(local)

    def put(self, key: str, outputs: dict, declared: dict) -> dict | None:
        """Files first, the marker last. None, and nothing stored, if an output is neither JSON nor a file on disk."""
        if not self.fs:
            return None
        stored = {}
        for name, value in outputs.items():
            if name == "doc":
                continue
            if declared.get(name, {}).get("type") in FILES:
                paths = value if isinstance(value, list) else [value]
                if not all(isinstance(p, str) and os.path.isfile(p) for p in paths):
                    return None  # an image in memory, not a file: the node is not stored
                refs = []
                for index, path in enumerate(paths):
                    ref = f"nodes/{key}/{name}/{index}{Path(path).suffix}"
                    self.fs.makedirs(f"{self.root}/nodes/{key}/{name}", exist_ok=True)
                    self.fs.put_file(path, f"{self.root}/{ref}")
                    refs.append({"file": ref})
                stored[name] = refs if isinstance(value, list) else refs[0]
            else:
                try:
                    json.dumps(value)
                except TypeError:
                    return (
                        None  # a live object, such as a tensor or a model: the node is not stored
                    )
                stored[name] = value
        self.fs.pipe_file(f"{self.root}/nodes/{key}.json", json.dumps(stored).encode())
        return stored


def revision(folder: Path) -> str:
    """A folder's content, as its loader reads it: the key's stand-in for a Hub commit."""
    digest = hashlib.sha256()
    for f in sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.name != "README.md" and "__pycache__" not in p.parts
    ):
        digest.update(f.relative_to(folder).as_posix().encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":  # python -m mellon.store blocks/* > nodes.json
    import sys

    print(
        json.dumps({"nodes": {f: {"revision": revision(Path(f))} for f in sys.argv[1:]}}, indent=2)
    )

store = Store(
    os.environ.get("MELLON_STORE"),
    os.environ.get("MELLON_LOCAL", Path(tempfile.gettempdir()) / "mellon-store"),
)
revisions = (
    {
        repo: n["revision"]
        for repo, n in json.loads(Path(os.environ["MELLON_NODES"]).read_text())["nodes"].items()
    }
    if os.environ.get("MELLON_NODES")
    else {}
)
