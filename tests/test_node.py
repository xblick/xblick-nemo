"""The node as its own process, without a camera: its options are parameters, and the objects to find (parameter
object) change while it runs. Needs a CUDA device and weights/ (tools/setup.sh): the node loads NeMO at its start."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest
import torch
from xblick.parameters import ParameterClient
from xblick.session import open_session, replies

from xblick_nemo import objects
from xblick_nemo.nemo import CROP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_the_objects_to_find_are_a_live_parameter(tmp_path):
    """A list with an object that is not in the objects directory is refused with the reason, a list of objects there
    is taken (the objects key says so), and an object added over the objects key shows in the parameter."""
    for name in ("red-tape", "blue-tape"):
        objects.save(tmp_path, name, torch.randn(50, 8), torch.rand(50, 3) * 2 - 1, torch.rand(3, 3, CROP, CROP),
                     {"source": "test"})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    command = [shutil.which("xblick-nemo", path=os.path.dirname(sys.executable)), "--objects-dir", str(tmp_path),
               "--object", "red-tape", "--connect", f"127.0.0.1:{port}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    client = ParameterClient(session, "nemo")

    def ask(payload=None) -> dict:
        got = [r for r in replies(session, "nemo/objects", 2.0, **({"payload": payload} if payload else {}))
               if r.ok is not None]
        return json.loads(got[0].ok.payload.to_bytes()) if got else {}

    try:
        deadline = time.monotonic() + 180.0
        while not any("nemo/objects" in line for line in output):
            assert process.poll() is None and time.monotonic() < deadline, "".join(output)
            time.sleep(0.5)
        descriptors, values = client.all()
        by_name = {d.name: (d, v.value()) for d, v in zip(descriptors, values, strict=True)}
        assert by_name["object"][1] == ["red-tape"] and not by_name["object"][0].read_only
        assert not by_name["min_inliers"][0].read_only and by_name["autocast"][0].read_only
        assert by_name["library"][1] == ["blue-tape", "red-tape"] and by_name["library"][0].read_only
        refused = client.set({"object": ["red-tape", "no-such-tape"]})[0]
        assert not refused.successful and "no-such-tape" in refused.reason
        assert client.set({"object": ["blue-tape"]})[0].successful
        deadline = time.monotonic() + 30.0
        while set(ask().get("objects", {})) != {"blue-tape"}:
            assert time.monotonic() < deadline, "".join(output)
            time.sleep(0.3)
        assert ask(json.dumps({"add": "red-tape"}))
        assert sorted(client.get(["object"])[0].value()) == ["blue-tape", "red-tape"]
    finally:
        client.close()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        session.close()
    assert process.returncode == 0, "".join(output)
