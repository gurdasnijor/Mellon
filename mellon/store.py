"""One node cache shared by every Mellon server that points at the same store. MELLON_STORE is a folder or an fsspec
URL (s3://<bucket>/<prefix>, its endpoint and key in the environment). A node's outputs are nodes/<key>.json under it,
its files under nodes/<key>/. A file in a value is {"file": "<ref>"}, a path under the root: one another program put
there, or a stored node's own, such as nodes/<key>/depth_videos/0.mp4. MELLON_NODES names a JSON file of each node
folder's revision, which `python mellon/store.py nodes.json blocks/*` writes, and where a folder's node runs if not
here. A value naming a file on this machine, such as a file
field's path under Mellon's work folder, keys by the file's content: the same path with other bytes is another key, and
the same bytes under another name or on another host are the same one."""

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

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


def file_here(value, work_dir: Path) -> Path | None:
    """The file on this machine a param's string names, relative to work_dir or absolute; None for any other value
    (a prompt is a string too, and may be too long to be a path)."""
    if not isinstance(value, str) or not value or "://" in value:
        return None
    path = Path(value) if Path(value).is_absolute() else Path(work_dir) / value
    try:
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def content(value, work_dir: Path):
    """A param's value with each string that names a file here as {"sha256": …}."""
    if isinstance(value, list):
        return [content(v, work_dir) for v in value]
    path = file_here(value, work_dir)
    if path is None:
        return value
    stat = path.stat()
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


class Stale(Exception):
    """A saved graph changed since it was opened, or one of that name exists: the save is refused."""


def graph_path(path: str) -> str | None:
    """graphs/<path> for a saved graph's path, <workspace>/<name>.json, or None if it isn't one."""
    parts = PurePosixPath(path).parts
    if len(parts) < 2 or not path.endswith(".json") or any(p in ("", ".", "..") or p.startswith("/") for p in parts):
        return None
    return f"graphs/{path}"


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
        """A value with each {"file": ref} in it as a path on this machine, read from the store once."""
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, dict) and set(value) != {"file"}:
            return {k: self.resolve(v) for k, v in value.items()}
        if not (isinstance(value, dict) and self.fs):
            return value
        if isinstance(self.fs, LocalFileSystem):
            return f"{self.root}/{value['file']}"
        local = self.local / value["file"]
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            self.fs.get_file(f"{self.root}/{value['file']}", str(local))
        return str(local)

    def graphs(self) -> list[str]:
        """Every saved graph's path under graphs/, such as dev/a composition.json."""
        if not self.fs or not self.fs.exists(f"{self.root}/graphs"):
            return []
        base = f"{self.root}/graphs/"
        return sorted(f[len(base) :] for f in self.fs.find(base) if f.endswith(".json"))

    def _etag(self, path: str, data: bytes | None = None) -> str:
        """The object's ETag; a folder's is the MD5 its bytes would have as one S3 put, which is R2's."""
        if isinstance(self.fs, LocalFileSystem):
            return '"' + hashlib.md5(self.fs.cat_file(path) if data is None else data).hexdigest() + '"'
        return self.fs.info(path)["ETag"]

    def graph(self, ref: str) -> tuple[bytes, str] | None:
        """A saved graph's bytes and ETag, ref being graphs/<path>."""
        path = f"{self.root}/{ref}"
        if not (self.fs and self.fs.exists(path)):
            return None
        data = self.fs.cat_file(path)
        return data, self._etag(path, data)

    def save_graph(self, ref: str, data: bytes, etag: str | None) -> str:
        """Writes graphs/<path> if it is still as opened (etag), or doesn't exist yet (no etag), and returns its new
        ETag; raises Stale otherwise. R2 checks the condition itself (If-Match, If-None-Match)."""
        path = f"{self.root}/{ref}"
        if isinstance(self.fs, LocalFileSystem):  # one server per folder: check, then write
            exists = self.fs.exists(path)
            if (self._etag(path) if exists else None) != etag:
                raise Stale(ref)
            self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
            self.fs.pipe_file(path, data)
            return self._etag(path, data)
        from botocore.exceptions import ClientError

        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            return self.fs.pipe_file(path, data, **condition)["ETag"]
        except OSError as e:
            cause = e.__cause__
            if isinstance(cause, ClientError) and cause.response["Error"]["Code"] in ("PreconditionFailed", "412"):
                raise Stale(ref) from e
            raise

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
                stored[name] = self._files(value, key, name, iter(range(1 << 30)))
        self.fs.pipe_file(f"{self.root}/nodes/{key}.json", json.dumps(stored).encode())
        return stored

    def _files(self, value, key: str, name: str, index):
        """A JSON output with each absolute path of a file on this machine in it stored as {"file": ref}, as a key
        hashes a file a value names: a selection's clip is read back wherever the selection is."""
        if isinstance(value, list):
            return [self._files(v, key, name, index) for v in value]
        if isinstance(value, dict):
            return {k: self._files(v, key, name, index) for k, v in value.items()}
        if not (isinstance(value, str) and os.path.isabs(value) and os.path.isfile(value)):
            return value
        ref = f"nodes/{key}/{name}/{next(index)}{Path(value).suffix}"
        self.fs.makedirs(f"{self.root}/nodes/{key}/{name}", exist_ok=True)
        self.fs.put_file(value, f"{self.root}/{ref}")
        return {"file": ref}


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


if __name__ == "__main__":  # python mellon/store.py nodes.json blocks/*: the revisions, the rest of the file kept
    import sys

    out = Path(sys.argv[1])
    kept = json.loads(out.read_text()) if out.exists() else {}
    nodes = {f: {"revision": revision(Path(f))} for f in sys.argv[2:]}
    out.write_text(json.dumps({**kept, "nodes": nodes}, indent=2) + "\n")

store = Store(
    os.environ.get("MELLON_STORE"),
    os.environ.get("MELLON_LOCAL", Path(tempfile.gettempdir()) / "mellon-store"),
)
_nodes = json.loads(Path(os.environ["MELLON_NODES"]).read_text()) if os.environ.get("MELLON_NODES") else {}
revisions = {repo: n["revision"] for repo, n in _nodes.get("nodes", {}).items()}
# Where a folder's node runs when not here: "modal:<app>/<class>", whose run_graph runs it (mellon/remote.py). Not in
# a key, so moving a node between hosts never changes one.
placement = _nodes.get("placement", {})
