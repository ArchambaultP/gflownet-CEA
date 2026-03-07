"""Binary protocol for communicating with worker subprocesses."""
import pickle
import struct
import select


def send(pipe, obj):
    """Send a pickled object with length prefix."""
    data = pickle.dumps(obj)
    pipe.write(struct.pack('<I', len(data)))
    pipe.write(data)
    pipe.flush()


def recv(pipe):
    """Receive a length-prefixed pickled object."""
    raw = pipe.read(4)
    if not raw:
        raise EOFError
    size = struct.unpack('<I', raw)[0]
    data = pipe.read(size)
    return pickle.loads(data)


def recv_timeout(pipe, timeout=15):
    """Receive with timeout. Raises TimeoutError if no data arrives."""
    ready, _, _ = select.select([pipe], [], [], timeout)
    if not ready:
        raise TimeoutError("Worker did not respond in time")
    return recv(pipe)