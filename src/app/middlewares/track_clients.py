from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import uuid
from app.services.manager import client_manager

async def track_clients_middleware(request: Request, call_next):
    client_id = request.headers.get('cookie', str(uuid.uuid4()))

    if request.url.path == "/video_feed":
        client_manager.add_client(client_id)

    response = await call_next(request)

    if request.url.path == "/video_feed":
        if isinstance(response, StreamingResponse):
            response.background = BackgroundTask(
                client_manager.remove_client, client_id
            )

    return response