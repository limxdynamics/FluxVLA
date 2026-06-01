"""Serving subpackage for remote VLA inference.

All imports are lazy -- msgpack, zmq, and other serving dependencies are
only loaded when a symbol from this package is actually accessed.  This
avoids forcing ``pip install pyzmq msgpack`` on users who only do local
inference.
"""


def __getattr__(name):
    _public = {
        'FORMAT_MSGPACK',
        'FORMAT_PROTOBUF',
        'MsgSerializer',
        'ObsSerializer',
        'ObsSerializerProto',
        'decode_predict_request',
        'decode_predict_response',
        'detect_format',
        'encode_predict_request',
        'encode_predict_response',
    }
    _server = {'PolicyServer', 'create_server', 'serialize_actions'}

    if name in _public:
        from . import serializers
        return getattr(serializers, name)
    if name in _server:
        from . import zmq_server
        return getattr(zmq_server, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'FORMAT_MSGPACK',
    'FORMAT_PROTOBUF',
    'MsgSerializer',
    'ObsSerializer',
    'ObsSerializerProto',
    'PolicyServer',
    'create_server',
    'decode_predict_request',
    'decode_predict_response',
    'detect_format',
    'encode_predict_request',
    'encode_predict_response',
    'serialize_actions',
]
