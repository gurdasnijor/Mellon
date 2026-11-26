import logging
import asyncio
from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
import aiofiles
logging.getLogger('asyncio').setLevel(logging.WARNING)
from functools import partial
from importlib import import_module
import json
import nanoid
from utils.paths import list_files
from pathlib import Path
import traceback
import time
from copy import deepcopy
import threading
from concurrent.futures import Future, ThreadPoolExecutor
logger = logging.getLogger('mellon')

from mellon.config import CONFIG
from mellon.modelstore import modelstore
from mellon.store import store, revisions, types, graph_keys, placement, graph_path, Stale, library_entry, render_path
from mellon import remote, telemetry
from modules import MODULE_MAP
telemetry.configure()  # ours: before the app is made, so aiohttp's instrumentation sees it
from utils.huggingface import get_local_models, delete_model, search_hub, download_hub_model, get_local_model_ids
from utils.torch_utils import reset_memory_stats, get_memory_stats

async def revalidated(request, response):
    """Ours: the client's files are asked for again on each load, a 304 when unchanged, as index.html already is: their
    names have no hash, so a browser that kept them would run an old client after a deploy."""
    if request.path.startswith(('/assets/', '/user/')):
        response.headers['Cache-Control'] = 'no-cache'


class WebServer:
    def __init__(
            self,
            modules: dict = {},
            host: str = '127.0.0.1',
            port: int = 8088,
            secure: bool = False,
            certfile: str = None,
            keyfile: str = None,
            cors: bool = False,
            cors_routes: list = [],
            client_max_size: int = 1024**4,
            work_dir: str = 'data',
            data_dir: str = 'data'
        ):
        self.instance = nanoid.generate(size=10)

        self.modules = modules
        self.ws_sessions = {}
        self.pending_ws_requests = {}

        self.interrupt_flag = False
        self.node_cache = {}
        self.tasks = {}  # ours: each graph's state for GET /graph/{task_id}, kept while the server runs (unbounded)

        self.queued_tasks = {}
        self.current_task = {}
        
        self.main_queue = asyncio.Queue()
        self.background_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

        self.main_worker_task = None
        self.background_worker_task = None
        self.runner = None
        self.site = None

        self.host = host
        self.port = port
        self.ssl_context = None
        if secure and certfile and keyfile:
            import ssl
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        self.client_max_size = client_max_size
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.app = web.Application(client_max_size=self.client_max_size)
        self.app.on_response_prepare.append(revalidated)  # ours

        # set up the routes
        self.app.add_routes([
            web.static('/assets', 'web/assets', append_version=True),
            web.get('/', self.index),
            web.get('/favicon.ico', self.favicon),
            web.get('/ws', self.websocket),
            web.get(r'/nodes{id:/?([\w\d_-]+/[\w\d_-]+)?}', self.nodes),
            web.post('/fields/action', self.field_action),
            web.get('/cache/{node}/{field}', self.cache),
            web.get('/cache/{node}/{field}/{index}', self.cache),
            web.delete('/cache', self.delete_cache),
            web.get('/listdir', self.listdir),
            web.get('/listgraphs', self.listgraphs),
            web.put('/graphs/{path:.+}', self.save_graph),
            web.get('/renders', self.renders),
            web.get('/file', self.fileGet),
            web.post('/file', self.filePost),
            web.get('/preview', self.preview),
            web.post('/graph', self.graph),
            web.get('/graph/{task_id}', self.graph_status),
            web.post('/stored', self.stored),  # ours
            web.get('/queue', self.get_queue),
            web.delete('/queue/{task_id}', self.delete_task),
            web.get('/stop', self.stop_execution),
            web.get('/local_models', self.local_models),
            web.get('/hf_cache', self.hf_cache),
            web.delete('/hf_cache/{hash}', self.hf_cache_delete),
            web.get('/hf_hub', self.hf_hub),
            web.get('/hf_download', self.hf_download),
            web.get('/static/{module}/{file}', self.user_assets),
            web.get('/stream', self.stream)
        ])

        # serve the user assets
        try:
            self.app.add_routes([web.static('/user', 'web/user', append_version=True)])  # ours: add_routes takes a list; a bare StaticDef raised TypeError, which the except hid, so /user was never served
        except Exception as e:
            pass

        # set up the cors routes
        if cors:
            cors = cors_setup(self.app, defaults={
                cors_route: ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")
                for cors_route in cors_routes
            })
            for route in list(self.app.router.routes()):
                cors.add(route)
    
    async def run(self):
        # Get the current event loop
        self.loop = asyncio.get_event_loop()
        
        # Start both workers
        self.main_worker_task = self.loop.create_task(self._main_worker())
        self.background_worker_task = self.loop.create_task(self._background_worker())
        
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host=self.host, port=self.port, ssl_context=self.ssl_context)
        await self.site.start()
    
    async def cleanup(self):
        # Signal shutdown to workers
        self._shutdown_event.set()
        
        # Stop the web server from accepting new connections.
        if self.site:
            await self.site.stop()

        # Close all active websocket connections with timeout
        if self.ws_sessions:
            close_coroutines = []
            for sid, ws in list(self.ws_sessions.items()):
                try:
                    if not ws.closed:
                        close_coroutines.append(ws.close())
                except Exception as e:
                    logger.debug(f"Error preparing to close websocket {sid}: {e}")
            
            if close_coroutines:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*close_coroutines, return_exceptions=True),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Websocket connections did not close within timeout, forcing shutdown")
            
            # Clear the sessions dict
            self.ws_sessions.clear()

        # Cancel and clean up the background workers.
        if self.main_worker_task:
            self.main_worker_task.cancel()
        if self.background_worker_task:
            self.background_worker_task.cancel()
        
        # Add sentinel values to queues to wake up workers
        try:
            self.main_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        try:
            self.background_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        
        # Wait for the workers to finish with a timeout
        tasks_to_wait = [task for task in [self.main_worker_task, self.background_worker_task] if task]
        if tasks_to_wait:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_wait, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning("Worker tasks did not finish within timeout, forcing shutdown")

        # Cleanup the runner.
        if self.runner:
            await self.runner.cleanup()

    """
    ╭───────────────╮
          Queue    
    ╰───────────────╯
    """
    def _get_queue(self):
        #task_list_sorted = {k: v for k, v in sorted(self.queued_tasks.items(), key=lambda x: x[1]['queued_at'], reverse=True)}
        # filter out keys that are not needed for the client
        queued_tasks = {
            k: {
                'name': v['name'],
                'sid': v['sid'],
                'queued_at': v['queued_at'],
                'task_id': k,
            }
            for k, v in self.queued_tasks.items()
        }

        current_task = None
        if self.current_task:
            current_task = {
                "task_id": self.current_task["task_id"],
                "name": self.current_task["name"],
                "sid": self.current_task["sid"],
                "started_at": self.current_task["started_at"],
                "progress": self.current_task["progress"],
            }
        
        return queued_tasks, current_task

    async def queue_task(self, task, args, future, sid, name=None):
        task_id = nanoid.generate(size=12)
        task_name = name or f'Unnamed task ({task.__name__})'

        self.queued_tasks[task_id] = {
            'task': task,
            'args': args,
            'future': future,
            "sid": sid,
            "queued_at": time.time(),
            "name": task_name,
        }
        await self.main_queue.put((task, args, future, task_id))

        task_list, current_task = self._get_queue()

        self.queue_message({
            "type": "task_queued",
            "task_id": task_id,
            "sid": sid,
            "queued": task_list,
            "current": current_task,
        })
        return task_id  # ours: POST /graph answered task_id null without it

    async def graph_status(self, request):
        """Ours: {state: running | done | failed, error, nodes: {id: {key, stored, ran?, outputs?}}}, every node of the graph."""
        task_id = request.match_info["task_id"]
        if task_id in self.queued_tasks or (task_id not in self.tasks and (self.current_task or {}).get("task_id") == task_id):
            return web.json_response({"state": "queued"})
        task = self.tasks.get(task_id)
        return web.json_response(task) if task else web.json_response({"error": True, "message": "No such graph."}, status=404)

    async def get_queue(self, _):
        """
        HTTP endpoint to return the tasks queue and the current task.
        """
        task_list, current_task = self._get_queue()
        return web.json_response({
            "queued": task_list,
            "current": current_task,
        })
    
    async def delete_task(self, request):
        """
        HTTP endpoint to delete a task from the queue.
        """
        task_id = request.match_info.get('task_id')
        if task_id in self.queued_tasks:
            task = self.queued_tasks.pop(task_id)
            logger.info(f"Task {task_id} {task['name']} deleted from queue.")
            task_list, current_task = self._get_queue()
            self.queue_message({
                "type": "task_cancelled",
                "task_id": task_id,
                "queued": task_list,
                "current": current_task,
            })
            return web.json_response({"error": False, "task_id": task_id, "queued": task_list, "current": current_task})
        elif self.current_task and self.current_task["task_id"] == task_id:
            return web.json_response({"error": True, "message": f"Task is already running and cannot be cancelled.", "task_id": task_id}, status=400)
        
        return web.json_response({"error": True, "message": f"Task not found in queue.", "task_id": task_id}, status=404)
    
    async def _main_worker(self):
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Use wait_for with a timeout to check shutdown periodically
                    queue_item = await asyncio.wait_for(self.main_queue.get(), timeout=1.0)
                    
                    # Check for sentinel value (None) indicating shutdown
                    if queue_item is None:
                        self.main_queue.task_done()
                        break
                        
                    task, args, future, task_id = queue_item

                    if task_id not in self.queued_tasks:
                        logger.debug(f"Task {task_id} was cancelled, skipping.")
                        if future:
                            future.set_exception(asyncio.CancelledError(f"Task {task_id} was cancelled"))
                        self.main_queue.task_done()
                        continue

                    current_task = self.queued_tasks.pop(task_id)

                    self.current_task = {
                        "task_id": task_id,
                        "task": task,
                        "name": current_task["name"],
                        "sid": current_task["sid"],
                        "started_at": time.time(),
                        "progress": 0,
                        "args": args,
                    }
                    task_list, current_task = self._get_queue()
                    self.queue_message({
                        "type": "task_started",
                        "task_id": task_id,
                        "queued": task_list,
                        "current": current_task,
                    })
                    
                    try:
                        if isinstance(args, tuple):
                            result = await self.loop.run_in_executor(None, partial(task, *args))
                        elif isinstance(args, dict):
                            result = await self.loop.run_in_executor(None, partial(task, **args))
                        else:
                            result = await self.loop.run_in_executor(None, partial(task, args))
                        
                        if future:
                            future.set_result(result)
                    except Exception as e:
                        #logger.error(f"Error processing main task: {e}")
                        logger.error(f"Error occurred in {traceback.format_exc()}")
                        if future:
                            future.set_exception(e)
                    finally:
                        if self.current_task:
                            task_list, _ = self._get_queue()
                            self.queue_message({
                                "type": "task_completed",
                                "task_id": task_id,
                                "completed_at": time.time(),
                                "queued": task_list,
                                "current": None,
                                "sid": self.current_task["sid"],
                                "args": args,
                            })
                            self.current_task = None
                        self.main_queue.task_done()
                        self.interrupt_flag = False
                        
                except asyncio.TimeoutError:
                    # Timeout is expected, just continue to check shutdown
                    continue

        except Exception as e:
            logger.error(f"Main worker error: {e}")
        finally:
            logger.debug("Main worker shutting down")


    async def _background_worker(self):
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Use wait_for with a timeout to check shutdown periodically
                    queue_item = await asyncio.wait_for(self.background_queue.get(), timeout=1.0)
                    
                    # Check for sentinel value (None) indicating shutdown
                    if queue_item is None:
                        self.background_queue.task_done()
                        break
                        
                    task, args = queue_item
                    
                    try:
                        if isinstance(args, tuple):
                            await task(*args)
                        elif isinstance(args, dict):
                            await task(**args)
                        else:
                            await task(args)
                    except Exception as e:
                        logger.error(f"Error processing background task: {e}")
                        #logger.error(f"Error occurred in {traceback.format_exc()}")
                    finally:
                        self.background_queue.task_done()
                        
                except asyncio.TimeoutError:
                    # Timeout is expected, just continue to check shutdown
                    continue

        except Exception as e:
            logger.error(f"Background worker error: {e}")
        finally:
            logger.debug("Background worker shutting down")


    """
    ╭─────────────────────╮
       Basic HTTP Routes   
    ╰─────────────────────╯
    """

    async def index(self, _):
        response = web.FileResponse('web/index.html')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def favicon(self, _):
        return web.FileResponse('web/favicon.ico')

    async def user_assets(self, request):
        module = request.match_info.get('module')
        file = request.match_info.get('file')
        fileName = f"custom/{module}/web/{file}"
        
        if not Path(fileName).exists():
            return web.HTTPNotFound(text='File not found')
        
        response = web.FileResponse(fileName)
        #response.headers["Content-Type"] = "application/javascript"
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response


    """
    ╭─────────────────────────╮
       Nodes & Fields Routes
    ╰─────────────────────────╯
    """
    async def nodes(self, request):
        id = request.match_info.get('id', '').strip('/')
        modules = self.modules

        if id:
            m, n = id.split('/')
            if m not in modules:
                return web.json_response({"error": f"The module {m} was not found. Try refreshing the page and restarting the server."}, status=404)
            if n not in modules[m]:
                return web.json_response({"error": f"The node {n} was not found in the module {m}. Try refreshing the page and restarting the server."}, status=404)
            modules = {m: {n: modules[m][n]}}

        output = {}
        for module, actions in modules.items():
            for action, values in actions.items():
                params = deepcopy(values.get('params', {}))
                #spawn_fields = []
                for p in params:
                    if 'postProcess' in params[p]:
                        del params[p]["postProcess"]
                    #if 'spawn' in params[p] and params[p]['spawn']:
                    #    spawn_fields.append(p)
                # spawn fields are identified by the '>>>' suffix in the field key
                #for p in spawn_fields:
                #    params[f"{p}>>>0"] = params[p]
                #    del params[p]

                output[f"{module}.{action}"] = {
                    'module': module,
                    'action': action,
                    'type': values.get('type', 'custom'),
                    'label': values.get('label', f"{module}: {action}"),
                    'category': values.get('category', 'default'),
                    'description': values.get('description', ''),
                    'resizable': values.get('resizable', False),
                    'skipParamsCheck': values.get('skipParamsCheck', False),
                    'style': values.get('style', ''),
                    'params': params,
                    'time': [0,0,0],
                    'memory': [0,0,0],
                    'cache': False,
                }

        return web.json_response({
            'instance': self.instance,
            'nodes': output
        })

    async def field_action(self, request):
        data = await request.json()
        node = data.get('node')
        sid = data.get('sid')
        fn = data.get('fn')
        values = data.get('values')
        key = data.get('fieldKey', None)
        queue = data.get('queue', False)

        if node not in self.node_cache:
            module = data.get('module')
            action = data.get('action')
            work_module = import_module(f"{module}.main")
            work_action = getattr(work_module, action)
            work_action = work_action(node_id=node)
            self.node_cache[node] = work_action
        
        self.node_cache[node]._sid = sid # always update the sid as it may change over time

        fn = getattr(self.node_cache[node], fn)
        ref = {
            "node": node,
            "key": key,
            "queue": queue,
        }

        if queue:
            task_id = await self.queue_task(fn, (values, ref), None, sid, name=f"Field action")
        else:
            # Run field action in executor to avoid blocking the event loop
            if not getattr(self, 'loop', None):
                self.loop = asyncio.get_event_loop()
            try:
                await self.loop.run_in_executor(None, partial(fn, values, ref))
            except Exception as e:
                logger.error(f"Error executing field action synchronously: {e}")
                return web.json_response({
                    "error": True,
                    "message": f"Field action error: {e}",
                    "sid": sid,
                    "ref": ref,
                }, status=500)
            task_id = None

        return web.json_response({
            "error": False,
            "message": f"Field action `{fn}` for node `{node}` queued for processing",
            "sid": sid,
            "task_id": task_id,
            "ref": ref,
        })

    """
    ╭─────────────────────╮
       Node Cache Routes
    ╰─────────────────────╯
    """

    async def cache(self, request):
        node = request.match_info.get('node')
        field = request.match_info.get('field')
        index = request.match_info.get('index', None)

        if node not in self.node_cache:
            return web.HTTPNotFound(text=f"Node {node} not found in cache.")

        # get the actual value from the node cache
        if field in self.node_cache[node].output:
            data = self.node_cache[node].output[field]
        elif field in self.node_cache[node].params:
            data = self.node_cache[node].params[field]
        else:
            return web.HTTPNotFound(text=f"Field {field} not found in node {node} cache.")
        
        if data is None:
            return web.HTTPNotFound(text=f"Field {field} is empty in node {node} cache.")
        
        if isinstance(data, list):
            index = max(0, min(len(data) - 1, int(index))) if index else 0
            data = data[index]

        # check the registry for the type of the field
        module = self.node_cache[node].module_name
        action = self.node_cache[node].class_name
        type = getattr(self.node_cache[node], '_fields', self.modules[module][action]['params'])[field].get('type')  # ours

        # ours: a stored file plays from the bucket (a presigned URL answers ranges); anything else is read from the store
        if any(t in ('video', 'audio', 'file', '3d') for t in (type if isinstance(type, list) else [type])) and (url := store.url(data)):
            raise web.HTTPFound(url)
        data = store.resolve(data)

        filename = request.query.get('filename', f"{field}")

        charset = None
        if type == 'image' or any(t == 'image' for t in type):
            format = request.query.get('format', 'WEBP').upper()
            quality = request.query.get('quality', 100)
            out = to_bytes(type, data, {'format': format, 'quality': quality})
            content_type = f'image/{format.lower()}'
            filename = f"{filename}.{format.lower()}"
        elif type == 'text' or any(t.startswith('str') for t in type):
            out = str(data).encode('utf-8')
            content_type = f'text/plain'
            charset = 'utf-8'
            filename = f"{filename}.txt"
        else:
            resp = web.FileResponse(data)
            resp.headers['Content-Disposition'] = f'inline; filename="{filename}"'
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp
        
        return web.Response(
            body=out,
            content_type=content_type,
            charset=charset,
            headers={
                'Content-Disposition': f'inline; filename="{filename}"',
                'Content-Length': str(len(out)),
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )

    async def delete_cache(self, request):
        data = await request.json()
        nodes = data.get('nodes', [])

        if isinstance(nodes, str):
            nodes = list(self.node_cache.keys()) if nodes == '*' else [nodes]

        # this might take a while because it could be freeing up VRAM
        for node in nodes:
            if node in self.node_cache:
                del self.node_cache[node]
        
        logger.debug(f"Removed {len(nodes)} nodes from cache.")
        
        return web.json_response({"error": False, "nodes": nodes})
    

    """
    ╭───────────────────╮
       File Management
    ╰───────────────────╯
    """

    async def listdir(self, request):
        req_basepath = self.data_dir if request.query.get('basepath') == 'data' else self.work_dir
        req_path = request.query.get('path', req_basepath)
        if req_path.startswith('store:'):  # ours: store:<folder> lists that folder's library at its current revision
            folder = req_path.removeprefix('store:')
            current = revisions.get(folder)
            entries = [e for e in store.library(folder) if current is None or e['revision'] == current]
            files = [{'is_dir': False, 'is_hidden': False, 'name': e['name'] or e['key'][:12], 'path': f"store:nodes/{e['key']}",
                      'modified': e['modified'], 'size': None, 'ext': None, 'type': 'other', 'key': e['key'], 'params': e['params']} for e in entries]
            return web.json_response({'files': files, 'path': req_path, 'abs_path': req_path})
        req_type = request.query.get('type', None)
        if req_type:
            req_type = [t.strip() for t in req_type.lower().split(',')]

        full_path = Path(req_path)
        if not full_path.is_absolute():
            full_path = Path(req_basepath) / full_path

        file_types = {
            'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'ico', 'webp'],
            'audio': ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma', 'm4b', 'm4p', 'm4r'],
            'video': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'mpeg', 'mpg', 'm4v', 'webm'],
            'text': ['txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'log', 'html', 'css', 'js', 'ts', 'py', 'rb', 'php', 'sql', 'sh', 'bash'],
            'archive': ['zip', 'rar', 'tar', 'gz', 'bz2', '7z'],
            '3d': ['glb', 'gltf', 'stl', 'obj', 'fbx', 'dae', 'ply', '3ds', 'max', 'blend'],
        }

        if not str(full_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        contents = {
            'files': [],
            'path': '',
            'abs_path': '',
        }

        try:
            if full_path.exists():
                contents['path'] = str(full_path.relative_to(self.work_dir))
                contents['abs_path'] = str(full_path)

                for item in full_path.iterdir():
                    suffix = item.suffix.lstrip('.').lower()
                    # if any of the requested types don't match the file type, skip it
                    if not item.is_dir() and req_type and not any(suffix in exts for ftype, exts in file_types.items() if ftype in req_type):
                        continue

                    file = {
                        'is_dir': item.is_dir(),
                        'is_hidden': item.name.startswith('.'), # TODO: Windows: bool(os.stat(item).st_mode & stat.FILE_ATTRIBUTE_HIDDEN),
                        'name': item.name,
                        'path': str(item.relative_to(self.work_dir)),
                        #'abs_path': str(item),
                        'modified': item.stat().st_mtime,
                        'size': None,
                        'ext': None,
                        'type': None,
                    }
                    if not item.is_dir():
                        file['size'] = item.stat().st_size
                        file['ext'] = suffix
                        file['type'] = next((ftype for ftype, exts in file_types.items() if suffix in exts), 'other')
                    
                    contents['files'].append(file)

                return web.json_response(contents)
            else:
                return web.json_response({"error": f"The path {req_path} does not exist."}, status=404)

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def listgraphs(self, request):
        path = Path(self.data_dir) / 'graphs'
        saved = store.graphs()  # ours: the store's saved graphs, listed as the folder "saved"
        if not path.exists() and not saved:
            return web.json_response({"error": True, "message": "No graph directory found."}, status=404)
        
        graphs = list_files(str(path), recursive=True, extensions=['json']) if path.exists() else []
        graphs += [{"rel_directory": str(Path("saved") / Path(g).parent), "name": Path(g).name, "path": f"store:graphs/{g}"} for g in saved]

        # build a nested tree of directories and files
        tree: list[dict] = []

        for g in graphs:
            rel_dir = g.get('rel_directory') or '.'
            # normalize and split directory parts
            parts = [] if rel_dir in (None, '.', '') else [p for p in rel_dir.strip('/').split('/') if p]

            parent_children = tree
            current_path_parts: list[str] = []
            # ensure directory nodes exist for each part
            for part in parts:
                current_path_parts.append(part)
                node = next((n for n in parent_children if n.get('isDir') and n.get('name') == part), None)
                if not node:
                    node = {
                        "isDir": True,
                        "name": part,
                        "path": f"{str(path)}/{'/'.join(current_path_parts)}",
                        "children": []
                    }
                    parent_children.append(node)
                parent_children = node['children']

            raw_name = g.get('name') or Path(g.get('path', '')).name
            file_item = {
                "isDir": False,
                "name": Path(raw_name).stem,
                "path": g.get('path')
            }
            parent_children.append(file_item)

        # recursively sort directories (dirs first, then files) by name
        def sort_children(children: list[dict]):
            dirs = [c for c in children if c['isDir']]
            files = [c for c in children if not c['isDir']]
            dirs.sort(key=lambda x: x['name'].lower())
            files.sort(key=lambda x: x['name'].lower())
            for d in dirs:
                sort_children(d['children'])
            # mutate list in-place to preserve references
            children[:] = dirs + files

        sort_children(tree)

        return web.json_response(tree)

    async def renders(self, request):
        """ours: the renders kept as saved graphs, newest first, each with its videos in the store (the Gallery)."""
        return web.json_response(await asyncio.to_thread(store.renders) if store.fs else [])

    async def fileGet(self, request):
        file = request.query.get('file')

        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)

        if file.startswith('store:'):  # ours: a saved graph in the store, with its ETag for a save to send back
            ref = file.removeprefix('store:')
            if ref.startswith('nodes/'):
                return self.stored_file(ref)
            found = store.graph(ref) if ref.startswith('graphs/') else None
            if not found:
                return web.json_response({"error": f"The file {file} does not exist."}, status=404)
            return web.Response(body=found[0], content_type='application/json', headers={'ETag': found[1], 'Cache-Control': 'no-cache'})
        
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)
        
        return web.FileResponse(file_path)

    async def save_graph(self, request):
        """ours: PUT /graphs/<workspace>/<name>.json saves the canvas's graph to the store. With If-Match, only over the
        graph as it was opened; without, only where none is: anything else is 412, as a save from a stale tab."""
        ref = graph_path(request.match_info['path'])
        if not ref:
            return web.json_response({"error": "A saved graph's path is <workspace>/<name>.json."}, status=400)
        if not store.fs:
            return web.json_response({"error": "No store to save to: set MELLON_STORE."}, status=404)
        data = await request.read()
        try:
            graph = json.loads(data)
            assert isinstance(graph, dict) and 'nodes' in graph and 'edges' in graph
        except (ValueError, AssertionError):
            return web.json_response({"error": "Not a graph: the canvas's nodes, edges and viewport."}, status=400)
        try:
            etag = store.save_graph(ref, data, request.headers.get('If-Match'))
        except Stale:
            return web.json_response({"error": f"{request.match_info['path']} has changed since it was opened, or already exists."}, status=412)
        return web.json_response({"path": f"store:{ref}", "etag": etag}, headers={'ETag': etag})

    async def filePost(self, request):
        data = await request.post()
        file = data.get('file')
        type = data.get('type', 'images')
        type = type if type in ['images', 'audio', 'videos', 'text', '3d'] else 'images'
        file_path = Path(self.data_dir) / type / file.filename

        if file_path.exists():
            file_path = file_path.with_name(f"{file_path.stem}_{nanoid.generate(size=6)}{file_path.suffix}")
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(file.file.read())
            
            return web.json_response({"error": False, "path": str(file_path.relative_to(self.work_dir))})
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def preview(self, request):
        from utils.image import cover
        from PIL import Image
        from io import BytesIO

        Image.MAX_IMAGE_PIXELS = None

        file = request.query.get('file')
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)
        
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not str(file_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)
        
        if not file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.webp']:
            return web.json_response({"error": f"The file {file} is not an image."}, status=400)

        width = int(request.query.get('width', 0))
        height = int(request.query.get('height', 0))
        format = request.query.get('format', 'jpeg')
        format = format.lower()
        quality = int(request.query.get('quality', 95))

        image = Image.open(file_path)

        if width > 0 or height > 0:
            width = min(2048, width) if width > 0 else min(2048, height)
            height = min(2048, height) if height > 0 else min(2048, width)
        else:
            width = min(2048, image.width)
            height = min(2048, image.height)

        if width != image.width or height != image.height:
            image = cover(image, width, height, resample='BICUBIC')
        
        if image.mode != 'RGB' and format in ['jpeg', 'jpg', 'bmp', 'ico']:
            image = image.convert('RGB')

        bytes = BytesIO()
        image.save(bytes, format=format.upper(), quality=quality)
        bytes = bytes.getvalue()
        return web.Response(
            body=bytes,
            content_type=f'image/{format.lower()}',
            headers={
                'Content-Disposition': f'inline; filename="{file_path.name}"',
                'Content-Length': str(len(bytes)),
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    
    def stored_file(self, ref):
        """ours: a node's stored file, store:nodes/<key>/…: from the bucket by a presigned URL, or from a folder store."""
        if not (store.fs and '..' not in ref.split('/') and store.fs.exists(f"{store.root}/{ref}")):
            return web.json_response({"error": f"The file store:{ref} does not exist."}, status=404)
        if url := store.url({"file": ref}):
            raise web.HTTPFound(url)
        return web.FileResponse(store.resolve({"file": ref}))

    async def stream(self, request):
        file = request.query.get('file')
        if not file:
            return web.json_response({"error": "Incorrect request, `file` is required."}, status=400)
        if file.startswith('store:nodes/'):  # ours
            return self.stored_file(file.removeprefix('store:'))

        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = Path(self.work_dir) / file_path

        if not str(file_path).startswith(self.work_dir):
            return web.json_response({"error": f"Cannot access paths outside of {self.work_dir}."}, status=403)

        if not file_path.exists():
            return web.json_response({"error": f"The file {file} does not exist."}, status=404)
        
        resp = web.FileResponse(file_path)
        resp.headers['Content-Disposition'] = f'inline; filename="{file_path.name}"'
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        
        return resp


    async def local_models(self, request):
        refresh = request.query.get('refresh', False)
        path_match = request.query.get('match', "")

        if refresh:
            modelstore.update_local()

        files = modelstore.get_local_ids(name=path_match)

        return web.json_response(files)

    """
    ╭────────────────────────╮
       Main graph execution
    ╰────────────────────────╯
    """

    async def graph(self, request):
        graph = await request.json()
        sid = graph.get("sid")
        #if not sid:
        #    return web.json_response({"error": True, "message": "Missing session id"}, status=400)
        
        task_id = await self.queue_task(self.execute_graph, (graph,), None, sid, name=f"Graph execution")
        return web.json_response({
            "error": False,
            "message": "Graph queued for processing",
            "sid": sid,
            "task_id": task_id,
        })
    
    def check(self, nodes):
        """ours: every node's key, whether or not this server runs it (a file by its content), and whether it is stored,
        from every node's stored outputs read in one pass, which refs() then answers from"""
        with telemetry.timed(telemetry.store_s, op="keys"):
            keys = graph_keys(nodes, revisions, CONFIG.paths['work_dir'], types)
        found = store.read(keys.values())
        return keys, {i: found[k] is not None for i, k in keys.items()}

    async def stored(self, request):
        """ours: POST an exported graph ({nodes}); answers {id: stored} by the check a run starts with, running nothing,
        so the canvas shows what the store holds when a graph opens or an edit moves a key. With {sid, paths, show},
        the nodes in show that are stored then draw their results on that canvas, unless a task runs (its own do)"""
        graph = await request.json()
        keys, stored = await asyncio.to_thread(self.check, graph['nodes'])
        if graph.get('sid') and graph.get('show') and not self.current_task and not self.queued_tasks:
            self.showing = asyncio.create_task(asyncio.to_thread(self.show, graph, keys, stored))
        return web.json_response(stored)

    def show(self, graph, keys, stored):
        """ours: each stored node in show, in the paths' order, goes through execute_node as a run that finds it stored
        does: nothing runs, its outputs are its stored refs and its results go to the canvas (update_value, /cache)"""
        for id in dict.fromkeys(i for path in graph['paths'] for i in path):
            if stored.get(id) and id in graph['show']:
                try:
                    self.execute_node(id, graph['nodes'][id], graph['sid'], quiet=True, keys=keys)
                except Exception as e:  # a node it links from never ran here: its results wait for a run
                    logger.debug(f"Results of {id} not shown: {type(e).__name__}: {e}")

    def execute_graph(self, graph):
        with telemetry.timed(telemetry.graph_s, outcome="failed") as run:  # ours: the run's seconds
            self._execute_graph(graph)
            run["outcome"] = "done"

    def _execute_graph(self, graph):
        sid = graph['sid']
        nodes = graph['nodes']
        paths = graph['paths']
        with telemetry.timed(telemetry.store_s, op="check"):  # ours: the run's check, its keys and whether each is stored
            keys, stored = self.check(nodes)
            status = self.tasks[self.current_task['task_id']] = {"state": "running", "error": None, "nodes": {i: {"key": k, "stored": stored[i]} for i, k in keys.items()}}
        # ours: a post that runs a node placed elsewhere (a render) is kept as a saved graph, the canvas as it was run,
        # beside the composition it was opened from, at its first spawn (a skipped render keeps nothing); its keys find its takes
        self.keep_render = (lambda: store.save_graph(render_path(graph.get('saved'), time.time()), json.dumps({**graph['canvas'], 'keys': keys}).encode(), None)) if graph.get('canvas') and store.fs else None  # and its keys, which find its videos (/renders)
        try:
            self._execute_paths(sid, paths, nodes, keys, status)
        except Exception as e:
            status.update(state="failed", error=f"{type(e).__name__}: {e}")
            raise
        status["state"] = "done"

    class NotRun(Exception):
        """Ours: a node not run, as a node it links from failed or the run was stopped."""

    keeping = threading.Lock()  # ours: keep_render's, taken at a spawn, which a render's own thread makes

    def _execute_paths(self, sid, paths, nodes, keys, status):
        """Ours: each node once. A placed node not stored runs on a thread of its own as soon as the nodes it links
        from are done, at most remote.IN_FLIGHT spawned at once, so a graph's renders are in flight together; every
        other node runs here, in the paths' order, once the nodes it links from are done. A piece carried on from the
        one before links from it, so it waits for it. What the nodes it links from did decides, in one walk, what a
        node does: if one failed, it isn't run, and nor is any node after it; if a non-spawn input links a skipped
        one, it is skipped too (not run, spawned or stored); a spawn input leaves skipped ones out. Every render is
        waited for before the first error is raised, so one that fails leaves the others' takes stored and shown."""
        graph_execution_time = time.time()
        order = list(dict.fromkeys(i for path in paths for i in path))
        links = {i: [(p['sourceId'], bool(p.get('spawn'))) for p in nodes[i]['params'].values() if p.get('sourceId') in order] for i in order}
        placed = [i for i in order if not status['nodes'].get(i, {}).get('stored') and placement.get(nodes[i]['params'].get('repo_id', {}).get('value'))]
        done = {i: Future() for i in order if i not in placed}  # each node's status once done, or why it isn't
        finished, progress, in_flight = [0], threading.Lock(), threading.Semaphore(remote.IN_FLIGHT)

        def execute(id):
            if failed := list(dict.fromkeys(s for s, _ in links[id] if done[s].exception() is not None)):  # waits for them
                raise self.NotRun(f"Not run: {', '.join(failed)}, which it links from, did not run")
            skipped = {s for s, _ in links[id] if (done[s].result() or {}).get('skipped')}
            repo_id = nodes[id]['params'].get('repo_id', {}).get('value')  # each node's seconds, by folder and outcome
            if any(s in skipped and not spawn for s, spawn in links[id]):
                status['nodes'][id] = {"key": keys[id], "stored": False, "skipped": True}  # not run, spawned or stored
            else:
                with telemetry.timed(telemetry.node_s, folder=repo_id or f"{nodes[id]['module']}.{nodes[id]['action']}", outcome="failed") as timing:
                    if id in placed:
                        with in_flight:
                            if self.interrupt_flag:
                                raise self.NotRun("Not run: stopped before it was spawned")
                            status['nodes'][id] = self.execute_node(id, nodes[id], sid, keys=keys, nodes=nodes, skipped=skipped)
                    else:
                        status['nodes'][id] = self.execute_node(id, nodes[id], sid, keys=keys, nodes=nodes, skipped=skipped)
                    ran = (status['nodes'][id] or {}).get('ran', True)
                    timing['outcome'] = "skipped" if (status['nodes'][id] or {}).get('skipped') else ("remote" if placement.get(repo_id) else "ran") if ran else "stored"
            if (status['nodes'][id] or {}).get('skipped'):
                self.queue_message({"type": "skipped", "node": id}, sid)
            if self.current_task:  # broadcast the task progress
                with progress:
                    finished[0] += 1
                    self.current_task['progress'] = int(100 * finished[0] / len(order))
                self.queue_message({
                    "type": "task_progress",
                    "task_id": self.current_task["task_id"],
                    "progress": self.current_task['progress'],
                })
            return status['nodes'][id]

        with ThreadPoolExecutor(max(len(placed), 1), thread_name_prefix="placed") as pool:
            for i in placed:  # in order, so each node it links from has its future when its thread looks
                done[i] = pool.submit(execute, i)
            for id in order:
                if id in placed:
                    continue
                try:
                    if self.interrupt_flag:
                        raise self.NotRun("Not run: stopped")
                    done[id].set_result(execute(id))
                except Exception as e:
                    done[id].set_exception(e)
        failed = {i: done[i].exception() for i in order if done[i].exception() is not None}
        for i, e in failed.items():
            logger.error(f"Node {i} ({nodes[i]['module']}.{nodes[i]['action']}): {type(e).__name__}: {e}")
            if not isinstance(e, self.NotRun):  # ours: the canvas shows which node failed and why (NodeBase wraps the why)
                self.queue_message({"type": "failed", "node": i, "error": str(e.__context__ or e)}, sid)

        if self.interrupt_flag:
            self.interrupt_flag = False
            self.queue_message({
                "type": "graph_stopped",
                "sid": sid,
                "message": "Execution interrupted by the user."
            }, sid)
            return
        if errors := [e for e in failed.values() if not isinstance(e, self.NotRun)]:
            raise errors[0]

        # the graph has completed
        self.queue_message({
            "type": "graph_completed",
            "sid": sid,
            "executionTime": time.time() - graph_execution_time,
        }, sid)

    async def stop_execution(self, _):
        # check if there is a current task or any queued task
        if not self.current_task and not self.queued_tasks:
            return web.json_response({"error": True, "message": "Nothing to do. No task is currently running or queued."})

        if self.interrupt_flag:
            return web.json_response({"error": True, "message": "Execution is already set for interruption."})

        self.interrupt_flag = True

        # set the interrupt flag for all the nodes in the cache
        for node in list(self.node_cache.values()):  # ours: a copy, as a placed node's thread may add one meanwhile
            node._interrupt = True
        
        return web.json_response({"error": False, "message": "Execution set for interruption."})

    def execute_node(self, id, node, sid, quiet=False, keys=None, nodes=None, skipped=()):
        module = node['module']
        action = node['action']
        params = node['params']

        if module not in self.modules:
            raise ValueError(f"Invalid module: {module}")

        if action not in self.modules[module]:
            raise ValueError(f"Invalid action: {action}")

        # get the arguments values
        args = {}
        ui_fields = {}

        for p in params:
            data_source_id = params[p].get('sourceId')
            data_param_key = params[p].get('sourceKey')

            # the field is a UI element, used mostly to display the data in the UI
            if 'display' in params[p] and params[p]['display'] in ['ui_group', 'ui_text', 'ui_image', 'ui_imagecompare', 'ui_areaselect', 'ui_audio', 'ui_video', 'ui_3d', 'ui_label', 'ui_button']:
                ui_fields[p] = data_param_key if data_param_key else None

            # the field is an input that gets its value from an output of another node
            elif data_source_id and data_param_key:
                # spawn field handling
                #if '>>>' in p or self.modules[module][action]['params'][p].get('spawn'):
                if params[p].get('spawn'):
                    spawn_key = p.split('>>>')[0]
                    if not spawn_key in args:
                        args[spawn_key] = []
                    if data_source_id in skipped:  # ours
                        continue
                    args[spawn_key].append(self.node_cache[data_source_id].output[data_param_key])
                else:
                    args[p] = self.node_cache[data_source_id].output[data_param_key]
            # the field is a static value
            else:
                args[p] = params[p].get('value')

        if not quiet:
            reset_memory_stats()
            start_time = time.time()

            # tell the client that the node is running
            self.queue_message({
                "type": "progress",
                "node": id,
                "progress": -1, # -1 sets the progress to indeterminate
            }, sid)

        # if the node is not in the cache, initialize it
        if id not in self.node_cache:
            work_module = import_module(f"{module}.main")
            work_action = getattr(work_module, action)
            self.node_cache[id] = work_action(id)

        if not callable(self.node_cache[id]):
            raise TypeError(f"The class `{module}.{action}` is not callable. Make sure the class has a `__call__` method or extends `NodeBase`.")

        # set the session id, it can be used to send messages from the node back to the client
        self.node_cache[id]._sid = sid

        # ours: a Dynamic Block node's fields are its class's and its folder's config's (previews, /cache, the store)
        node = self.node_cache[id]
        fields = node._fields = {**self.modules[module][action]['params'], **(node._get_custom_config(args['repo_id']).node_params['custom']['params'] if hasattr(node, '_get_custom_config') else {})}

        # ours: a node stored under its key is not run, here or on any machine sharing the store, but shows its previews
        key = keys[id] if keys else None
        refs = store.refs(key) if key else None
        place = placement.get(params.get('repo_id', {}).get('value')) if key and nodes and refs is None else None
        if place:  # ours: a node placed elsewhere (nodes.json) runs there, and its outputs are read from the store
            with self.keeping:  # renders spawn from threads of their own: the first to spawn keeps the render, once
                keep, self.keep_render = getattr(self, 'keep_render', None), None
            if keep:
                keep()
            remote.run(place, id, nodes, keys, revisions[params['repo_id']['value']], CONFIG.paths['work_dir'], lambda: self.interrupt_flag)
            refs = store.refs(key)
            if refs is None:
                raise RuntimeError(f"{place} ran node {id} but stored no outputs")
        if refs is not None:
            node.output = refs  # read from the store only where it is used: by a node that runs here, or /cache
            node.params = {}  # its outputs are another key's: NodeBase's memo, by node id, must not answer a later run with them
            self.queue_message({"type": "executed", "node": id, "name": f"{module}.{action}", "stored": True}, sid)
        else:
            args = {p: store.resolve(v) for p, v in args.items()}  # ours: each {"file": ref} as a path on this machine

        # *** execute the node ***
        try:
            if refs is None:  # ours
                self.node_cache[id](**args)
        except Exception as e:
            logger.error(f"Error executing node {id} ({module}.{action})")
            logger.error(traceback.format_exc())
            raise e

        if not quiet and refs is None:  # ours
            execution_time = time.time() - start_time
            self.node_cache[id]._execution_time['last'] = execution_time
            self.node_cache[id]._execution_time['min'] = min(self.node_cache[id]._execution_time['min'], execution_time) if self.node_cache[id]._execution_time['min'] is not None else execution_time
            self.node_cache[id]._execution_time['max'] = max(self.node_cache[id]._execution_time['max'], execution_time) if self.node_cache[id]._execution_time['max'] is not None else execution_time

            memory_stats = get_memory_stats()
            if memory_stats:
                self.node_cache[id]._memory_usage['last'] = memory_stats['peak']
                self.node_cache[id]._memory_usage['min'] = min(self.node_cache[id]._memory_usage['min'], memory_stats['peak']) if self.node_cache[id]._memory_usage['min'] is not None else memory_stats['peak']
                self.node_cache[id]._memory_usage['max'] = max(self.node_cache[id]._memory_usage['max'], memory_stats['peak']) if self.node_cache[id]._memory_usage['max'] is not None else memory_stats['peak']

            # the node has completed
            self.queue_message({
                "type": "executed",
                "node": id,
                "name": f"{module}.{action}",
                "hasChanged": self.node_cache[id]._has_changed,
                "executionTime": self.node_cache[id]._execution_time,
                "memoryUsage": self.node_cache[id]._memory_usage
            }, sid)

        for ui_key, data_key in ui_fields.items():           
            message = None

            # skip for button and group fields
            if fields[ui_key].get('display') in ['ui_button', 'ui_group']:
                continue

            else:
                # if the data key is an output, get the value from the output otherwise from the params
                if data_key in self.node_cache[id].output:
                    source_value = self.node_cache[id].output[data_key]
                elif data_key in self.node_cache[id].params:
                    source_value = self.node_cache[id].params[data_key]
                else:
                    continue
                        
            data_type = fields[data_key].get('type') # data type of the source field
            data_format = fields[ui_key].get('type', 'text') # format of the returned value: text, raw, url
            fieldOptions = fields[ui_key].get('fieldOptions', {})

            source_value = source_value if isinstance(source_value, list) else [source_value]
            source_value = source_value if data_format == 'url' else store.resolve(source_value)  # ours: /cache reads a url's
            if data_format == 'url':
                if data_type == 'image':
                    data_value = [f"/cache/{id}/{data_key}/{i}?format={fieldOptions.get('format', 'WEBP')}&quality={fieldOptions.get('quality', 100)}&t={time.time()}"
                                  for i in range(len(source_value)) if source_value[i] is not None]
                else:
                    data_value = [f"/cache/{id}/{data_key}/{i}?t={time.time()}"
                                  for i in range(len(source_value)) if source_value[i] is not None]
            elif data_format == 'raw':
                data_value = [to_bytes(data_type, item, fieldOptions) for item in source_value if item is not None]
            else:
                data_value  = [to_base64(data_type, item, fieldOptions) for item in source_value if item is not None]

            message = {
                'client_id': sid,
                'type': 'update_value',
                'node': id,
                'key': ui_key,
                'data_type': data_type,
                'value': data_value
            }

            if message:
                self.queue_message(message, sid)

        # ours: a node that ran here and set none of its declared outputs has nothing to do: it is skipped, not stored
        declared = [n for n, f in fields.items() if f.get('display') == 'output' and n != 'doc']
        if refs is None and declared and all(self.node_cache[id].output.get(n) is None for n in declared):
            return {"key": key, "stored": False, "ran": True, "skipped": True}
        # ours: store the outputs under the node's key; a Dynamic Block node declares its outputs in its folder's config
        if not key:
            return None
        if refs is not None:
            return {"key": key, "stored": True, "ran": bool(place), "outputs": refs}
        output = self.node_cache[id].output
        repo_id = params.get('repo_id', {}).get('value')
        stored = store.put(key, output, fields, library_entry(params, revisions.get(repo_id), CONFIG.paths['work_dir'], output))  # ours: a source node is in its folder's library
        return {"key": key, "stored": stored is not None, "ran": True, "outputs": stored}


    def trigger_node(self, source_id, output, sid):
        if not self.current_task:
            return

        graph = self.current_task['args'][0]
        nodes = graph['nodes']

        for id in nodes:
            params = nodes[id]['params']

            for p in params:
                data_source_id = params[p].get('sourceId')
                data_param_key = params[p].get('sourceKey')
                if data_source_id == source_id and data_param_key == output:
                    self.execute_node(id, nodes[id], sid, quiet=True)


    """
    ╭────────────────╮
       Hugging Face
    ╰────────────────╯
    """

    async def hf_cache(self, request):
        refresh = request.query.get('refresh', False)
        id = request.match_info.get('id', None)
        class_name = request.query.get('className', None)
        compact = request.query.get('compact', False)
        return_type = "compact" if compact else "full"

        if refresh:
            modelstore.update_hf()
        
        models = modelstore.get_hf_models(id, class_name, return_type)

        return web.json_response(models)

    async def hf_cache_delete(self, request):
        hashes = request.match_info.get('hash').split(',')
        if not hashes:
            return web.json_response({"error": "Incorrect request, `hash` is required."}, status=400)
        
        result = delete_model(*hashes)
        return web.json_response({"error": not result})

    # TODO: not yet implemented
    async def hf_hub(self, request):
        query = request.query.get('q', '')

        future = asyncio.Future()
        await self.main_queue.put((search_hub, query, future))
        
        try:
            result = await future
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error in hf_hub endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)

    # TODO: not yet implemented
    async def hf_download(self, request):
        repo_id = request.query.get('repo_id')

        if not repo_id:
            return web.json_response({"error": "Incorrect request, `repo_id` is required."}, status=400)
        
        def progress_cb(progress):
            print(f"Download progress: {progress * 100:.1f}%")
        
        future = asyncio.Future()
        await self.main_queue.put((download_hub_model, (repo_id, progress_cb), future))
        
        try:
            result = await future
            return web.json_response({"error": False, "result": result})
        except Exception as e:
            logger.error(f"Error in hf_download endpoint: {e}")
            return web.json_response({"error": str(e)}, status=500)


    """
    ╭───────────────╮
        Websocket    
    ╰───────────────╯
    """    
    async def websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.query.get('sid')
        if not sid:
            sid = nanoid.generate(size=10)
        if sid in self.ws_sessions:
            # close the connection and remove the old session
            logger.debug(f"Websocket session {sid} already exists, closing the old session.")
            #await self.ws_sessions[sid].close()
            sid = nanoid.generate(size=10)
            #del self.ws_sessions[sid]
        
        self.ws_sessions[sid] = ws
        logger.debug(f"Websocket connection opened: {sid}")

        # Update the session id in all cached nodes
        for node in list(self.node_cache):  # ours: a copy, as a placed node's thread may add one meanwhile
            # check if the node has a _sid attribute
            if hasattr(self.node_cache[node], '_sid') and self.node_cache[node]._sid != sid:
                self.node_cache[node]._sid = sid

        # send the welcome message together with the ids of the cached nodes
        await self.broadcast({"type": "welcome", "instance": self.instance, "sid": sid, "cachedNodes": list(self.node_cache.keys())}, sid)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data['type'] == 'close':
                        await ws.close()
                        break
                    elif data['type'] == 'ping':
                        await self.broadcast({"type": "pong"}, sid)
                    elif data['type'] == 'signal_value':
                        request_id = data.get('request_id')
                        if not request_id:
                            logger.warning("[Websocket] signal_value received without request_id")
                            continue
                        # Resolve the pending request future if present
                        try:
                            promised_sid = data.get('sid', None)
                            future = self.pending_ws_requests.pop(request_id, None)
                            if future is None:
                                logger.debug(f"[Websocket] signal_value received for unknown request_id: {request_id}")
                                continue
                            if not future.done():
                                result = data.get('value')
                                if promised_sid != sid:
                                    result = { '__MELLON_ERROR': 'sid_mismatch' }
                                future.set_result(result)
                        except Exception as e:
                            logger.error(f"[Websocket] Error resolving signal_value for request_id {request_id}: {e}")

                    else:
                        logger.error(f"[Websocket] Invalid message type: {data['type']}")
                elif msg.type == WSMsgType.CLOSE:
                    await self.broadcast({"type": "close"}, sid)
                    break
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"[Websocket] Error: {ws.exception()}")
        except Exception as e:
            logger.error(f"[Websocket] Error: {e}")
        finally:
            if sid in self.ws_sessions:
                del self.ws_sessions[sid]
            logger.debug(f"Websocket connection closed: {sid}")
        
        return ws

    async def broadcast(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        sessions = []

        if sid:
            sessions = [sid] if not isinstance(sid, list) else sid
        else:
            sessions = list(self.ws_sessions.keys())

        if exclude:
            exclude = [exclude] if not isinstance(exclude, list) else exclude
            sessions = [s for s in sessions if s not in exclude]

        for session in sessions:
            try:
                if session in self.ws_sessions and not self.ws_sessions[session].closed:
                    if isinstance(message, dict):
                        await self.ws_sessions[session].send_json(message)
                    else:
                        await self.ws_sessions[session].send_bytes(message)
            except Exception as e:
                logger.error(f"[Websocket] Error broadcasting message: {e}")
                pass
    
    def queue_message(self, message: dict | bytes, sid: list[str] | str = None, exclude: list[str] | str = None):
        if self.loop.is_running() and not self._shutdown_event.is_set():
            asyncio.run_coroutine_threadsafe(
                self.background_queue.put((self.broadcast, (message, sid, exclude))),
                self.loop
            )
    
    def get_signal_value(self, node: str, field: str, sid: str, timeout: int = 2):
        try:
            if not sid or sid not in self.ws_sessions or self.ws_sessions[sid].closed:
                return { '__MELLON_ERROR': 'invalid_sid' }

            if not getattr(self, 'loop', None) or not self.loop.is_running():
                return { '__MELLON_ERROR': 'server_not_running' }

            try:
                running_loop = asyncio.get_running_loop()
                if running_loop is self.loop:
                    logger.warning("[Server] get_signal_value called from event loop thread; returning None to avoid deadlock.")
                    return { '__MELLON_ERROR': 'called_from_event_loop' }
            except RuntimeError:
                # No running loop in this thread; safe to proceed
                pass

            # Create a future bound to the server loop and register it
            request_id = nanoid.generate(size=12)
            future = self.loop.create_future()
            self.pending_ws_requests[request_id] = future

            # Send the request to the target client
            self.queue_message({
                "type": "get_signal_value",
                "request_id": request_id,
                "node": node,
                "field": field,
                "sid": sid
            }, sid)

            # Await the future result from outside the event loop thread
            # Use run_coroutine_threadsafe to wait with a timeout safely
            wrapped = asyncio.wait_for(future, timeout=timeout)
            cfut = asyncio.run_coroutine_threadsafe(wrapped, self.loop)
            try:
                result = cfut.result(timeout=timeout + 0.5)
            except Exception:
                result = { '__MELLON_ERROR': 'timeout' }
            finally:
                # Cleanup any leftover pending entry
                self.pending_ws_requests.pop(request_id, None)

            return result
        except Exception as e:
            logger.error(f"[Server] get_signal_value error: {e}")
            return { '__MELLON_ERROR': 'exception' }
    

def to_base64(type, value, options={}):
    import io
    import base64
    
    out = value

    if type == 'image':
        format = options.get('format', 'WEBP').upper()
        quality = options.get('quality')
        if format == 'WEBP' and not quality:
            quality = 100
        elif format == 'JPEG' and not quality:
            quality = 75
        elif format == 'PNG' and not quality:
            quality = None
        mime_type = f"image/{format.lower()}"

        byte_arr = io.BytesIO()
        value.save(byte_arr, format=format, quality=int(quality))
        # TODO: check shutil.copyfile
        header = f"data:{mime_type};base64,"
        out = header + base64.b64encode(byte_arr.getvalue()).decode('utf-8')
    
    return out

def to_bytes(data_type, value, options={}):
    import io
    from PIL import Image
    
    out = value

    if isinstance(value, Image.Image):
        format = options.get('format', 'WEBP').upper()
        quality = options.get('quality')
        if format == 'WEBP' and not quality:
            quality = 100
        elif format == 'JPEG' and not quality:
            quality = 75
        elif format == 'PNG' and not quality:
            quality = None

        byte_arr = io.BytesIO()
        value.save(byte_arr, format=format, quality=int(quality))
        out = byte_arr.getvalue()
    elif isinstance(value, str):
        out = value.encode('utf-8')
    
    return out

server = WebServer(
    MODULE_MAP,
    host=CONFIG.server['host'],
    port=CONFIG.server['port'],
    secure=CONFIG.server['secure'],
    certfile=CONFIG.server['certfile'],
    keyfile=CONFIG.server['keyfile'],
    cors=CONFIG.server['cors'],
    cors_routes=CONFIG.server['cors_routes'],
    client_max_size=CONFIG.server['client_max_size'],
    work_dir=CONFIG.paths['work_dir'],
    data_dir=CONFIG.paths['data']
)
