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
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import fsspec
import rfc8785
from fsspec.implementations.local import LocalFileSystem
from opentelemetry import metrics

RUNNER = {
    "device",
    "auto_offload",
    "trust_remote_code",
    "load_block_button",
    "library",
}  # where it runs, or a control on the node (the Library field), not what it makes
MELLON = "mellon@5fd24292"  # the revision of a node that is not a folder: Mellon's own, at our fork's base
FILES = ("video", "image", "file")
SHA256 = re.compile(r"sha256:([0-9a-f]{64})")  # a file named by its content, as a library entry names it


# Seconds, from a store request's milliseconds to a render's minutes; bytes, from a key's JSON to a take (telemetry.py)
SECONDS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600]
BYTES = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
_meter = metrics.get_meter("mellon")
store_s = _meter.create_histogram(
    "mellon.store.duration", unit="s", description="One operation on the store. By op.",
    explicit_bucket_boundaries_advisory=SECONDS,
)
store_bytes = _meter.create_histogram(
    "mellon.store.size", unit="By", description="What one operation on the store read or wrote. By op.",
    explicit_bucket_boundaries_advisory=BYTES,
)


@contextmanager
def measured(op: str):
    """The block's seconds in mellon.store.duration, and the bytes it sets in mellon.store.size."""
    moved = {"bytes": None}
    start = time.perf_counter()
    try:
        yield moved
    finally:
        store_s.record(time.perf_counter() - start, {"op": op})
        if moved["bytes"] is not None:
            store_bytes.record(moved["bytes"], {"op": op})


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
    if isinstance(value, str) and (named := SHA256.fullmatch(value)):
        return {"sha256": named[1]}
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


def typed(value, kind: str | None):
    """A value as DynamicBlockNode.execute casts it by its param's declared type, so 5 typed in the canvas, which it
    posts as "5", keys as the 5 a graph loads; one that does not cast stays as sent."""
    if kind not in ("int", "float") or value is None or isinstance(value, bool):
        return value
    try:
        return int(value) if kind == "int" else float(value)
    except (TypeError, ValueError):
        return value


def graph_keys(nodes: dict, revisions: dict, work_dir: str | Path = ".", types: dict | None = None) -> dict:
    """Every node's key: SHA-256 of the JCS of {module, action, revision, params}; a link hashes its source's key, a
    file named here hashes its content, and a value its folder declares int or float (types) hashes as cast."""
    keys = {}

    def key(node_id):
        if node_id not in keys:
            node, params = nodes[node_id], {}
            repo_id = node["params"].get("repo_id", {}).get("value")
            declared = (types or {}).get(repo_id, {})
            for name, p in node["params"].items():
                if name in RUNNER or p.get("display", "").startswith("ui_"):
                    continue
                params[name] = (
                    {"key": key(p["sourceId"]), "sourceKey": p["sourceKey"]}
                    if p.get("sourceId")
                    else {"value": content(typed(p.get("value"), declared.get(name)), Path(work_dir))}
                )
            ours = repo_id in revisions  # a node of one of our folders keys as the Dynamic Block Node that runs it,
            body = {  # whichever sidebar entry made it (custom.H3's, or Dynamic Block Node's with the folder loaded)
                "module": "modules.ModularDiffusers" if ours else node["module"],
                "action": "DynamicBlockNode" if ours else node["action"],
                "revision": revisions.get(repo_id, MELLON),
                "params": params,
            }
            keys[node_id] = hashlib.sha256(rfc8785.dumps(body)).hexdigest()
        return keys[node_id]

    for node_id in nodes:
        key(node_id)
    return keys


def library_entry(params: dict, revision: str | None, work_dir: str | Path, outputs: dict) -> dict | None:
    """A stored node's library entry, if no link feeds it (a clip, a person): its folder, revision and params as its
    key reads them, a file as sha256:<hex>, so writing them back into a node of that folder gives the same key and
    so its stored result, even once the file is gone; and a name, an output's "name" if it has one."""
    folder = params.get("repo_id", {}).get("value")
    if not folder or any(p.get("sourceId") for p in params.values()):
        return None

    def named(value):
        if isinstance(value, list):
            return [named(v) for v in value]
        return f"sha256:{value['sha256']}" if isinstance(value, dict) and set(value) == {"sha256"} else value

    values = {
        name: named(content(p.get("value"), Path(work_dir)))
        for name, p in params.items()
        if name not in RUNNER and name != "repo_id" and not p.get("display", "").startswith("ui_") and p.get("display") != "output"
    }
    name = next((v["name"] for v in outputs.values() if isinstance(v, dict) and isinstance(v.get("name"), str)), None)
    return {"folder": folder, "revision": revision, "name": name, "params": values}


class Stale(Exception):
    """A saved graph changed since it was opened, or one of that name exists: the save is refused."""


def graph_path(path: str) -> str | None:
    """graphs/<path> for a saved graph's path, <workspace>/<name>.json, or None if it isn't one."""
    parts = PurePosixPath(path).parts
    if len(parts) < 2 or not path.endswith(".json") or any(p in ("", ".", "..") or p.startswith("/") for p in parts):
        return None
    return f"graphs/{path}"


def render_path(saved: str | None, when: float) -> str:
    """Where a render posted from the canvas is kept as a saved graph: graphs/<workspace>/<name>/renders/<UTC time>.json
    for the saved graph <workspace>/<name>.json it was opened from (store:graphs/..., as the canvas has it; a render
    opened and run again files under its composition), graphs/renders/<UTC time>.json for a graph never saved."""
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(when)) + f".{int(when % 1 * 1000):03d}Z"
    name = saved.removeprefix("store:graphs/").removesuffix(".json") if saved and saved.startswith("store:graphs/") else ""
    name = name.split("/renders/")[0]
    return f"graphs/{name}/renders/{stamp}.json" if name else f"graphs/renders/{stamp}.json"


class Store:
    def __init__(self, url: str | None, local: str | Path):
        self.fs, self.root = fsspec.core.url_to_fs(url) if url else (None, None)
        self.local = Path(local)  # where this machine keeps the remote files it has read
        self.found: dict = {}  # the last read()'s stored outputs, by key

    def exists(self, key: str) -> bool:
        if not self.fs:
            return False
        with measured("exists"):
            return self.fs.exists(f"{self.root}/nodes/{key}.json")

    def read(self, keys) -> dict:
        """Each key's stored outputs, or None where nothing is stored, in one pass: a GET per key, all at once (a
        graph's 18 from a Mac in 0.4 s, where a HEAD and then a read for each took 7.4). A key's outputs never change,
        its key being what made them, so refs() answers from this pass until the next one; a key not stored then is
        asked again."""
        keys = list(dict.fromkeys(keys))
        if not (self.fs and keys):
            return dict.fromkeys(keys)
        paths = {self.fs._strip_protocol(f"{self.root}/nodes/{key}.json"): key for key in keys}
        with measured("read") as moved:
            # max_concurrency=1: s3fs asks an object's size (a HEAD) first only to read a large one in parts
            data = self.fs.cat(list(paths), on_error="return", max_concurrency=1)
            moved["bytes"] = sum(len(d) for d in data.values() if isinstance(d, bytes))
        for d in data.values():
            if isinstance(d, Exception) and not isinstance(d, FileNotFoundError):
                raise d
        self.found = {paths[p]: json.loads(d) for p, d in data.items() if isinstance(d, bytes)}
        return {key: self.found.get(key) for key in keys}

    def refs(self, key: str) -> dict | None:
        """The stored outputs as written, each file as {"file": ref}: one GET, none if the last read() found them."""
        if key in self.found:
            return self.found[key]
        if not self.fs:
            return None
        with measured("refs") as moved:
            try:
                data = self.fs.cat_file(f"{self.root}/nodes/{key}.json", max_concurrency=1)
            except FileNotFoundError:
                return None
            moved["bytes"] = len(data)
        return json.loads(data)

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
            # Read to a name of its own, then renamed into place: a file at this path is a whole one, so a second
            # reader (the canvas's preview of the node beside the node using it) never reads one half written
            local.parent.mkdir(parents=True, exist_ok=True)
            fd, part = tempfile.mkstemp(dir=local.parent, prefix=f".{local.name}.", suffix=".part")
            os.close(fd)
            try:
                with measured("resolve") as moved:
                    self.fs.get_file(f"{self.root}/{value['file']}", part)
                    moved["bytes"] = os.path.getsize(part)
                os.replace(part, local)
            finally:
                if os.path.exists(part):
                    os.remove(part)
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

    def library(self, folder: str) -> list[dict]:
        """A folder's library: the entries its stored nodes wrote, each with its key and when it was stored."""
        base = f"{self.root}/libraries/{folder}/"
        if not self.fs or not self.fs.exists(base):
            return []
        found = {p: i for p, i in self.fs.find(base, detail=True).items() if p.endswith(".json")}
        data = self.fs.cat(list(found)) if found else {}
        entries = []
        for path, info in found.items():
            stored = info.get("mtime") or info.get("LastModified")
            stored = stored.timestamp() if hasattr(stored, "timestamp") else stored
            entries.append({"key": path.rsplit("/", 1)[1][: -len(".json")], "modified": stored, **json.loads(data[path])})
        return sorted(entries, key=lambda e: -(e["modified"] or 0))

    def url(self, value) -> str | None:
        """A stored file's presigned URL, which answers range reads, so a player streams it from the bucket rather than
        through this machine; None for a folder store or a value that is not {"file": ref}. botocore signs for its
        region: R2's is auto (AWS_DEFAULT_REGION), or the URL is refused."""
        if not (isinstance(value, dict) and set(value) == {"file"} and self.fs) or isinstance(self.fs, LocalFileSystem):
            return None
        return self.fs.url(f"{self.root}/{value['file']}", expires=3600)

    def put(self, key: str, outputs: dict, declared: dict, entry: dict | None = None) -> dict | None:
        """Files first, the marker last, then the node's library entry if it has one. None, and nothing stored, if an
        output is neither JSON nor a file on disk."""
        with measured("put") as moved:
            return self._put(key, outputs, declared, entry, moved)

    def _put(self, key: str, outputs: dict, declared: dict, entry: dict | None, moved: dict) -> dict | None:
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
                    moved["bytes"] = (moved["bytes"] or 0) + os.path.getsize(path)
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
        if entry:
            self.fs.makedirs(f"{self.root}/libraries/{entry['folder']}", exist_ok=True)
            self.fs.pipe_file(f"{self.root}/libraries/{entry['folder']}/{key}.json", json.dumps(entry).encode())
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


CONFIG = "mellon_pipeline_config.json"


def runs(config: dict) -> dict:
    """What of a folder's Mellon config can change its node's outputs: what DynamicBlockNode.execute reads (the dtype,
    the names of the inputs it passes, the outputs it returns and the models, and the params' types it casts by) and
    the outputs' types, which the store reads. Not labels, display, defaults, options or order, which only the canvas
    reads: a folder's node skips Mellon's param checks, so the value the canvas sends is what the block gets, and that
    value is in the key."""
    custom = config.get("node_params", {}).get("custom", {})
    names = {kind: custom.get(kind, []) for kind in ("input_names", "output_names", "model_input_names")}
    params = custom.get("params", {})
    types = {name: params.get(name, {}).get("type") for kind in names.values() for name in kind}
    return {"default_dtype": config.get("default_dtype"), **names, "types": types}


def revision(folder: Path) -> str:
    """What runs in a folder, as its loader reads it: the key's stand-in for a Hub commit. Every file byte for byte but
    Markdown, which nothing runs, and the Mellon config, of which only what runs (runs())."""
    digest = hashlib.sha256()
    for f in sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix != ".md" and "__pycache__" not in p.parts
    ):
        data = rfc8785.dumps(runs(json.loads(f.read_bytes()))) if f.name == CONFIG else f.read_bytes()
        digest.update(f.relative_to(folder).as_posix().encode() + b"\0" + data)
    return digest.hexdigest()


def numbers(folder: Path) -> dict:
    """The params of a folder's node that execute casts to a number (int or float), which its key casts the same."""
    config = folder / CONFIG
    types = runs(json.loads(config.read_bytes()))["types"] if config.exists() else {}
    return {name: kind for name, kind in types.items() if kind in ("int", "float")}


if __name__ == "__main__":  # python mellon/store.py nodes.json blocks/*: the revisions, the rest of the file kept
    import sys

    out = Path(sys.argv[1])
    kept = json.loads(out.read_text()) if out.exists() else {}
    nodes = {f: {"revision": revision(Path(f)), "types": numbers(Path(f))} for f in sys.argv[2:]}
    out.write_text(json.dumps({**kept, "nodes": nodes}, indent=2) + "\n")

store = Store(
    os.environ.get("MELLON_STORE"),
    os.environ.get("MELLON_LOCAL", Path(tempfile.gettempdir()) / "mellon-store"),
)
_nodes = json.loads(Path(os.environ["MELLON_NODES"]).read_text()) if os.environ.get("MELLON_NODES") else {}
revisions = {repo: n["revision"] for repo, n in _nodes.get("nodes", {}).items()}
types = {repo: n.get("types", {}) for repo, n in _nodes.get("nodes", {}).items()}
# Where a folder's node runs when not here: "modal:<app>/<class>", whose run_graph runs it (mellon/remote.py). Not in
# a key, so moving a node between hosts never changes one.
placement = _nodes.get("placement", {})
