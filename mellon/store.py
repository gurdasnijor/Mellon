"""One node cache shared by every Mellon server that points at the same store. MELLON_STORE is a folder or an fsspec
URL (s3://<bucket>/<prefix>, its endpoint and key in the environment). A node's outputs are nodes/<key>.json under it,
its files under nodes/<key>/. A file in a value is {"file": "<ref>"}, a path under the root: one another program put
there, or a stored node's own, such as nodes/<key>/depth_videos/0.mp4. MELLON_NODES names a JSON file of each node
folder's revision, which `python mellon/store.py nodes.json blocks/*` writes, and where a folder's node runs if not
here. A value naming a file on this machine, such as a file
field's path under Mellon's work folder, keys by the file's content: the same path with other bytes is another key, and
the same bytes under another name or on another host are the same one. An upload, and a file a library entry's node
read, is put under files/<hex>/<name> and named sha256:<hex>/<name>, which keys as the file does and which any host
reads it by."""

import hashlib
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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
# A file named by its content, and the name it came with: sha256:<hex>/<name>, kept at files/<hex>/<name>. A library
# entry written before names were kept has sha256:<hex> alone, which keys the same and names no stored file.
SHA256 = re.compile(r"sha256:([0-9a-f]{64})(?:/([^/]+))?")


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
    key reads them, a file on this machine as sha256:<hex>/<name>, so writing them back into a node of that folder gives
    the same key and so its stored result, even once the file is gone; and a name, an output's "name" if it has one.
    Its "files", those files' paths, are put in the store with it (Store.put) and not written in it."""
    folder = params.get("repo_id", {}).get("value")
    if not folder or any(p.get("sourceId") for p in params.values()):
        return None

    files = []

    def named(value):
        if isinstance(value, list):
            return [named(v) for v in value]
        path = file_here(value, Path(work_dir))
        if path is None:
            return value
        files.append(str(path))
        return named_value(str(path))

    values = {
        name: named(p.get("value"))
        for name, p in params.items()
        if name not in RUNNER and name != "repo_id" and not p.get("display", "").startswith("ui_") and p.get("display") != "output"
    }
    name = next((v["name"] for v in outputs.values() if isinstance(v, dict) and isinstance(v.get("name"), str)), None)
    return {"folder": folder, "revision": revision, "name": name, "params": values, "files": files}


def named_value(path: str, name: str | None = None) -> str:
    """A file's name by its content and its own name (or the one given, an upload's), sha256:<hex>/<name>."""
    name = PurePosixPath((name or Path(path).name).replace("\\", "/")).name
    if name in ("", ".", ".."):
        raise ValueError(f"No file name in {name!r}")
    return f"sha256:{content(os.path.abspath(path), Path('/'))['sha256']}/{name}"


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


def render_videos(canvas: dict) -> tuple[list, list]:
    """A render's video outputs as (node id, output) pairs: its finished videos, those of the nodes nothing reads from
    (Assemble's), and its takes, those of the nodes placed elsewhere (Render's)."""
    read_from = {e.get("source") for e in canvas.get("edges", [])}

    def videos(node):
        params = node.get("data", {}).get("params", {})
        return [(node["id"], n) for n, p in params.items() if p.get("display") == "output" and p.get("type") == "video"]

    nodes = canvas.get("nodes", [])
    finished = [v for n in nodes if n["id"] not in read_from for v in videos(n)]
    placed = [n for n in nodes if placement.get(n.get("data", {}).get("params", {}).get("repo_id", {}).get("value"))]
    return finished, [v for n in placed for v in videos(n) if v not in finished]


class Store:
    def __init__(self, url: str | None, local: str | Path):
        self.fs, self.root = fsspec.core.url_to_fs(url) if url else (None, None)
        self.local = Path(local)  # where this machine keeps the remote files it has read
        self.found: dict = {}  # the last read()'s stored outputs, by key
        self.files_url = os.environ.get("MELLON_STORE_FILES")  # s3://<bucket>/<prefix>: where put_file() puts files
        self._s3 = None

    def exists(self, key: str) -> bool:
        if not self.fs:
            return False
        with measured("exists"):
            return self.fs.exists(f"{self.root}/nodes/{key}.json")

    def read(self, keys) -> dict:
        """Each key's stored outputs, or None where nothing is stored, in one pass: a GET per key, all at once. A
        graph's 18 entries took 0.5 s through s3fs and 0.4 s through a mounted bucket (a folder, as H3's container
        has), where a check and then a read for each, one after another, took 7.4 and 6.9. A key's outputs never
        change, its key being what made them, so refs() answers from this pass until the next one; a key not stored
        then is asked again."""
        keys = list(dict.fromkeys(keys))
        if not (self.fs and keys):
            return dict.fromkeys(keys)
        with measured("read") as moved, ThreadPoolExecutor(32) as pool:
            data = dict(zip(keys, pool.map(self._entry, keys)))
            moved["bytes"] = sum(len(d) for d in data.values() if d is not None)
        self.found = {key: json.loads(d) for key, d in data.items() if d is not None}
        return {key: self.found.get(key) for key in keys}

    def _entry(self, key: str) -> bytes | None:
        try:  # max_concurrency=1: s3fs asks an object's size (a HEAD) first only to read a large one in parts
            return self.fs.cat_file(f"{self.root}/nodes/{key}.json", max_concurrency=1)
        except FileNotFoundError:
            return None

    def refs(self, key: str) -> dict | None:
        """The stored outputs as written, each file as {"file": ref}: one GET, none if the last read() found them."""
        if key in self.found:
            return self.found[key]
        if not self.fs:
            return None
        with measured("refs") as moved:
            data = self._entry(key)
            moved["bytes"] = len(data) if data is not None else None
        return json.loads(data) if data is not None else None

    def resolve(self, value):
        """A value with each {"file": ref} in it, and each file named by its content (sha256:<hex>/<name>), as a path
        on this machine, read from the store once."""
        if isinstance(value, str) and (named := SHA256.fullmatch(value)) and named[2]:
            value = {"file": f"files/{named[1]}/{named[2]}"}
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

    def renders(self) -> list[dict]:
        """Every render kept with its run's keys, newest first: where it is saved, the graph it was opened from, when it
        ran, and its videos as the store has them now, the finished ones if any is stored, else the takes stored so
        far. A render kept without keys (before they were kept) is left out."""
        saved = [g for g in self.graphs() if g.startswith("renders/") or "/renders/" in g]
        if not saved:
            return []
        data = self.fs.cat([f"{self.root}/graphs/{g}" for g in saved])
        canvases = {g: json.loads(data[f"{self.root}/graphs/{g}"]) for g in saved}
        canvases = {g: c for g, c in canvases.items() if isinstance(c.get("keys"), dict)}
        wanted = {g: render_videos(c) for g, c in canvases.items()}
        keys = {canvases[g]["keys"].get(i) for g, (f, t) in wanted.items() for i, _ in f + t} - {None}
        with ThreadPoolExecutor(32) as pool:  # one GET each, all at once, as read() does, leaving its pass alone
            found = {k: json.loads(d) for k, d in zip(keys, pool.map(self._entry, keys)) if d is not None}
        renders = []
        for g, canvas in canvases.items():
            labels = {n["id"]: n.get("data", {}).get("label") for n in canvas["nodes"]}

            def stored(pairs):
                out = []
                for node, name in pairs:
                    value = found.get(canvas["keys"].get(node), {}).get(name)
                    for ref in value if isinstance(value, list) else [value]:
                        if isinstance(ref, dict) and set(ref) == {"file"}:
                            out.append({"node": node, "label": labels[node], "file": f"store:{ref['file']}"})
                return out

            finished, takes = wanted[g]
            done = stored(finished)
            folder, _, stamp = g.rpartition("renders/")
            renders.append({
                "path": f"store:graphs/{g}",
                "graph": folder.rstrip("/") or None,
                "time": stamp.removesuffix(".json"),
                "finished": bool(done),
                "videos": done or stored(takes),
            })
        return sorted(renders, key=lambda r: r["time"], reverse=True)

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

    def put_file(self, path: str, ref: str) -> None:
        """A file at ref under the root. With files_url (H3's container, beside its mounted bucket) one S3 PUT, its read
        timeout 10 s, as a stalled request waits it out before its retry: 1.4 s to the mount's 4.7 for 1.5 MB, worst
        11.8 to 27 (findings §32). Only files: the mount caches a miss for 60 s and doesn't see an S3 write past it, so
        the marker, which a reader looks up and the executor's refs() misses before the node runs, is through fs."""
        if not self.files_url:
            self.fs.makedirs(f"{self.root}/{ref.rsplit('/', 1)[0]}", exist_ok=True)
            self.fs.put_file(path, f"{self.root}/{ref}")
            return
        if self._s3 is None:  # its endpoint and region from AWS_ENDPOINT_URL and AWS_DEFAULT_REGION
            import boto3
            from botocore.config import Config

            config = Config(connect_timeout=10, read_timeout=10, retries={"max_attempts": 3, "mode": "standard"})
            self._s3 = boto3.client("s3", config=config)
        bucket, _, prefix = self.files_url.removeprefix("s3://").partition("/")
        with open(path, "rb") as body:
            self._s3.put_object(Bucket=bucket, Key=f"{prefix}/{ref}", Body=body)

    def put_named(self, path: str, name: str | None = None) -> str:
        """A file put in the store by its content and name, files/<hex>/<name>, unless there already; its name there,
        sha256:<hex>/<name>, which keys as the file did and which any host reads it by (an upload, a library entry's
        input)."""
        value = named_value(path, name)
        named = SHA256.fullmatch(value)
        ref = f"files/{named[1]}/{named[2]}"
        if not self.fs.exists(f"{self.root}/{ref}"):
            self.put_file(path, ref)
        return value

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
                    self.put_file(path, ref)
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
            for path in entry.pop("files"):  # what the node read, so a pick of the entry has it on any host
                self.put_named(path)
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
        self.put_file(value, ref)
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
