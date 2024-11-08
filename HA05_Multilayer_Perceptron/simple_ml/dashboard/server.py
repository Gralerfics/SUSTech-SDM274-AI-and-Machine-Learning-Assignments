"""
    Run by `python -m simple_ml.dashboard.server` outside the package.
"""

from sanic import Sanic, response
from sanic import HTTPResponse, Request, Websocket

from .vc import ViewClientsPool
from .core import DashboardCore


""" Properties """
APP_NAME = 'simple_ml_dashboard'
APP_HOST = 'localhost'
APP_PORT = 2333

app = Sanic(APP_NAME)

clients = ViewClientsPool()
core = DashboardCore(clients)


""" CORS Middleware """
@app.middleware('request')
async def cors_middle_req(request: Request):
    if request.method.lower() == 'options':
        allow_headers = [
            'Authorization',
            'content-type'
        ]
        headers = {
            'Access-Control-Allow-Methods': ', '.join(['GET', 'POST', 'OPTIONS']),
            'Access-Control-Max-Age': '86400',
            'Access-Control-Allow-Headers': ', '.join(allow_headers),
        }
        return HTTPResponse('', headers=headers)

@app.middleware('response')
def cors_middle_res(request: Request, response: HTTPResponse):
    allow_origin = '*'
    response.headers.update(
        {
            'Access-Control-Allow-Origin': allow_origin,
        }
    )


""" Routes """
@app.route('/')
def index(request: Request):
    return response.html(f'<p>{APP_NAME}</p>')

@app.websocket('/notify') # new client
async def notify(request: Request, ws: Websocket):
    task_id = request.args.get('task_id', None)
    view_id = request.args.get('view_id', None)
    if task_id is None or view_id is None:
        return response.json({
            'status': 'error',
            'message': 'task_id and view_id are required.'
        })
    
    clients.add(ws, task_id, view_id)
    
    try:
        async for msg in ws: # TODO: 异步监听？
            print(f'Received: {msg}')
    except Exception as e:
        print(f'WebSocket error: {e}')

@app.get('/api/get_state')
def api_get_state(request: Request):
    return response.json({
        'status': 'ok',
        'state': core.get_state()
    })

@app.post('/api/launch_task')
def api_launch_task(request: Request):
    data = request.json

    for key in ['model', 'loss', 'dataset', 'optimizer']:
        if key not in data.keys():
            return response.json({
                'status': 'error',
                'message': f'{key} is required.'
            })
    
    if core.launch_task(data):
        return response.json({
            'status': 'ok',
            'state': core.get_state()
        })
    else:
        return response.json({
            'status': 'error',
            'message': 'Failed to launch task.'
        })


""" Main """
if __name__ == '__main__':
    app.run(host = APP_HOST, port = APP_PORT)

