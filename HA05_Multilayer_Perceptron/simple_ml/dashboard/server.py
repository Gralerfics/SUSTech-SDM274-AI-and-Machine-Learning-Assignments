"""
    Run by `python -m simple_ml.dashboard.server` outside the package.
"""

import json

from sanic import Sanic, response
from sanic import HTTPResponse, Request, Websocket

from .clients import ClientsPool
from .core import DashboardCore


""" Properties """
APP_NAME = 'simple_ml_dashboard'
APP_HOST = 'localhost'
APP_PORT = 2333

app = Sanic(APP_NAME)

clients = ClientsPool()
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
    clients.add(ws)
    await core.async_ws_listener(ws)

@app.get('/api/get_state')
def api_get_state(request: Request):
    return response.json({
        'status': 'ok',
        'state': core.get_state()
    })

@app.post('/api/kfold_cv')
def api_kfold_cv(request: Request):
    pass

@app.post('/api/launch')
def api_launch(request: Request):
    data = request.json
    """
    {
        'dataset': {
            'type': 'builtin',
            'name': '<function_name>',
            'params': {...: ...},
            'proc': '<preprocess_function_name>',
            'test_ratio': 0.2,
            'batch_size': ...
        },
        'model': [
            {'type': 'Linear', 'params': {...: ...}},
            {'type': 'Sigmoid'},
            ...
        ],
        'loss': {
            'type': 'MSELoss',
            'params': {...: ...}
        },
        'optimizer': {
            'type': 'GD',
            'params': {...: ...}
        }
    }
    """

    for key in ['model', 'loss', 'dataset', 'optimizer']:
        # TODO: full validation
        if key not in data.keys():
            return response.json({
                'status': 'error',
                'message': f'{key} is required.'
            })
    
    if core.launch(data):
        return response.json({
            'status': 'ok',
            'state': core.get_state()
        })
    else:
        return response.json({
            'status': 'error',
            'message': 'Failed to launch task.'
        })

@app.get('/api/reset')
def api_stop(request: Request):
    core.reset()
    return response.json({
        'status': 'ok',
        'state': core.get_state()
    })

@app.get('/api/pause')
def api_pause(request: Request):
    core.pause()
    return response.json({
        'status': 'ok',
        'state': core.get_state()
    })

@app.get('/api/resume')
def api_resume(request: Request):
    core.resume()
    return response.json({
        'status': 'ok',
        'state': core.get_state()
    })


""" Main """
if __name__ == '__main__':
    app.run(host = APP_HOST, port = APP_PORT)

