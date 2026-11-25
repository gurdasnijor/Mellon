"""Ours: a node that nodes.json places elsewhere ("modal:<app>/<class>") runs there, not here. Its class's run_graph is
spawned with the node, the nodes it links from, all of them stored, and their keys, which every host shares; the
call's id is written beside the node's key, nodes/<key>.call, and the outputs are read from the store when the call
ends. A re-post, after a reload or a restart, finds the outputs or that call and waits on it, and never spawns twice.
A call that failed is forgotten, so the next post spawns another."""

import os
from copy import deepcopy
from pathlib import Path

from mellon.store import store

WAIT = 10  # seconds between looks at whether the server was asked to stop


def upstream(nodes: dict, node_id: str) -> list[str]:
    """The node and every node it links from, each after the ones it links from."""
    order: list[str] = []

    def visit(i):
        if i in order:
            return
        for p in nodes[i]["params"].values():
            if p.get("sourceId"):
                visit(p["sourceId"])
        order.append(i)

    visit(node_id)
    return order


def stored_inputs(value, key: str, name: str, work_dir: Path, index):
    """A param's value with each file on this machine it names (as store.content finds them) put in the store under
    the node's key and named by its ref, so the host it runs on reads it."""
    if isinstance(value, list):
        return [stored_inputs(v, key, name, work_dir, index) for v in value]
    if not isinstance(value, str) or not value or "://" in value:
        return value
    path = Path(value) if Path(value).is_absolute() else work_dir / value
    if not path.is_file():
        return value
    ref = f"nodes/{key}/inputs/{name}/{next(index)}{path.suffix}"
    store.fs.makedirs(os.path.dirname(f"{store.root}/{ref}"), exist_ok=True)
    store.fs.put_file(str(path), f"{store.root}/{ref}")
    return {"file": ref}


def run(place: str, node_id: str, nodes: dict, keys: dict, revision: str, work_dir, stopped) -> None:
    """Runs a placed node where it is placed and waits for its outputs in the store. Raises if the call failed, or
    if the server was asked to stop: the call goes on, and its outputs are there for the next post."""
    import modal

    scheme, _, target = place.partition(":")
    if scheme != "modal" or "/" not in target:
        raise ValueError(f"Unknown placement {place!r}: modal:<app>/<class>")
    key = keys[node_id]
    marker = f"{store.root}/nodes/{key}.call"
    if store.fs.exists(marker):
        call = modal.FunctionCall.from_id(store.fs.cat_file(marker).decode())
    else:
        order = upstream(nodes, node_id)
        missing = [i for i in order[:-1] if not store.exists(keys[i])]
        if missing:
            raise RuntimeError(f"Run {', '.join(missing)} first: {place} reads them from the store")
        node = deepcopy(nodes[node_id])
        index = iter(range(1 << 30))
        for name, p in node["params"].items():
            if not p.get("sourceId") and "value" in p:
                p["value"] = stored_inputs(p["value"], key, name, Path(work_dir), index)
        graph = {
            "nodes": {**{i: nodes[i] for i in order}, node_id: node},
            "paths": [order],
            "keys": {i: keys[i] for i in order},
        }
        app, cls = target.split("/", 1)
        call = modal.Cls.from_name(app, cls)().run_graph.spawn(graph, node_id, revision)
        store.fs.pipe_file(marker, call.object_id.encode())
    while True:
        if stopped():
            raise InterruptedError(f"Stopped waiting on {call.object_id}: its outputs land in the store")
        try:
            call.get(timeout=WAIT)
            return
        except TimeoutError:
            continue
        except Exception:
            store.fs.rm_file(marker)
            raise

