"""Wire-format serializers for ZMQ transport.

Two formats are supported for the predict_action hot path:

- **msgpack** (default): flexible dict-based, zero schema
- **protobuf**: schema-driven, optimal for cross-language clients (C++/Rust)

Format detection: first byte ``0x01`` = protobuf, otherwise msgpack.
"""
from __future__ import annotations
import io
from typing import Any, Literal

import cv2
import msgpack
import numpy as np

FORMAT_MSGPACK: int = 0x00
FORMAT_PROTOBUF: int = 0x01

JPEG_QUALITY = 95
JPEG_KEYS = frozenset({
    'cam_high',
    'cam_left_wrist',
    'cam_right_wrist',
    'agentview_image',
    'robot0_eye_in_hand_image',
})


def detect_format(raw: bytes) -> int:
    if raw and raw[0] == FORMAT_PROTOBUF:
        return FORMAT_PROTOBUF
    return FORMAT_MSGPACK


def encode_obs_fields(obs: dict, compress: bool = True):
    """Encode observation dict into (images, arrays, strings) dicts.

    Args:
        obs: Raw observation dict.
        compress: JPEG-compress RGB images in JPEG_KEYS.

    Returns:
        Tuple of (images_bytes_dict, arrays_bytes_dict, strings_dict).
    """
    images, arrays, strings = {}, {}, {}
    for k, v in obs.items():
        if (compress and isinstance(v, np.ndarray) and v.ndim == 3
                and v.dtype == np.uint8 and k in JPEG_KEYS):
            _, jpg = cv2.imencode('.jpg', v,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            images[k] = jpg.tobytes()
        elif isinstance(v, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, v, allow_pickle=False)
            arrays[k] = buf.getvalue()
        elif isinstance(v, str):
            strings[k] = v
    return images, arrays, strings


def decode_image(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def decode_array(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


class MsgSerializer:
    """Msgpack serializer with built-in numpy array support."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if '__ndarray__' in obj:
            return decode_array(obj['data'])
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {'__ndarray__': True, 'data': buf.getvalue()}
        raise TypeError(f'Cannot serialize {type(obj)}')


class ObsSerializer:
    """Serialize raw observation dicts via msgpack."""

    @staticmethod
    def to_bytes(obs: dict, compress: bool = True) -> bytes:
        images, arrays, strings = encode_obs_fields(obs, compress)
        encoded = {}
        for k, v in images.items():
            encoded[k] = {'__jpeg__': True, 'data': v}
        for k, v in arrays.items():
            encoded[k] = {'__ndarray__': True, 'data': v}
        for k, v in strings.items():
            encoded[k] = v
        return msgpack.packb(encoded)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        raw = msgpack.unpackb(data, raw=False)
        obs = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                if '__jpeg__' in v:
                    obs[k] = decode_image(v['data'])
                elif '__ndarray__' in v:
                    obs[k] = decode_array(v['data'])
                else:
                    obs[k] = v
            elif isinstance(v, bytes):
                obs[k] = v.decode()
            else:
                obs[k] = v
        return obs


class ObsSerializerProto:
    """Serialize raw observation dicts via protobuf Observation message."""

    @staticmethod
    def obs_to_proto(obs: dict, compress: bool = True):
        from .proto import vla_service_pb2 as pb

        images, arrays, strings = encode_obs_fields(obs, compress)
        msg = pb.Observation()
        for k, v in images.items():
            msg.images[k] = v
        for k, v in arrays.items():
            msg.arrays[k] = v
        for k, v in strings.items():
            msg.strings[k] = v
        return msg

    @staticmethod
    def obs_from_proto(msg) -> dict:
        obs: dict = {}
        for k, v in msg.images.items():
            obs[k] = decode_image(v)
        for k, v in msg.arrays.items():
            obs[k] = decode_array(v)
        for k, v in msg.strings.items():
            obs[k] = v
        return obs


def encode_predict_request(
    obs: dict,
    unnorm_key: str,
    fmt: Literal['msgpack', 'protobuf'] = 'msgpack',
    compress: bool = True,
) -> bytes:
    """Encode a predict_action request."""
    if fmt == 'protobuf':
        from .proto import vla_service_pb2 as pb

        req = pb.PredictActionRequest()
        req.obs.CopyFrom(
            ObsSerializerProto.obs_to_proto(obs, compress=compress))
        req.unnorm_key = unnorm_key
        return bytes([FORMAT_PROTOBUF]) + req.SerializeToString()

    payload = ObsSerializer.to_bytes(obs, compress=compress)
    return msgpack.packb({
        'endpoint': 'predict_action',
        'data': {
            'obs_data': payload,
            'unnorm_key': unnorm_key
        },
    })


def decode_predict_request(raw: bytes) -> tuple[int, dict, str]:
    """Decode a predict_action request.

    Returns:
        (format_tag, obs_dict, unnorm_key)
    """
    fmt = detect_format(raw)
    if fmt == FORMAT_PROTOBUF:
        from .proto import vla_service_pb2 as pb

        req = pb.PredictActionRequest()
        req.ParseFromString(raw[1:])
        obs = ObsSerializerProto.obs_from_proto(req.obs)
        return FORMAT_PROTOBUF, obs, req.unnorm_key

    parsed = msgpack.unpackb(raw, raw=False)
    data = parsed.get('data', {})
    obs = ObsSerializer.from_bytes(data['obs_data'])
    return FORMAT_MSGPACK, obs, data.get('unnorm_key', '')


def encode_predict_response(
    action_data: bytes,
    infer_time: float,
    fmt: int = FORMAT_MSGPACK,
    error: str = '',
) -> bytes:
    """Encode a predict_action response."""
    if fmt == FORMAT_PROTOBUF:
        from .proto import vla_service_pb2 as pb

        resp = pb.PredictActionResponse()
        if error:
            resp.error = error
        else:
            resp.action_data = action_data
            resp.infer_time = infer_time
        return bytes([FORMAT_PROTOBUF]) + resp.SerializeToString()

    if error:
        return msgpack.packb({'error': error})
    return msgpack.packb({
        'action_data': action_data,
        'infer_time': infer_time
    })


def decode_predict_response(
    raw: bytes,
    fmt: int = FORMAT_MSGPACK,
) -> dict:
    """Decode a predict_action response."""
    if fmt == FORMAT_PROTOBUF:
        from .proto import vla_service_pb2 as pb

        resp = pb.PredictActionResponse()
        resp.ParseFromString(raw[1:])
        if resp.error:
            return {'error': resp.error}
        return {
            'action_data': resp.action_data,
            'infer_time': resp.infer_time,
        }

    return msgpack.unpackb(raw, raw=False)
