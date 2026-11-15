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


def test_queries_are_answered_at_once_while_the_model_works(tmp_path):
    """While NeMO's search of a frame takes 4 s (tests/slow_step.py), the queryables answer within 0.5 s and nothing
    else (no "Timeout" reply: the query is not kept open); a camera started and an object added then are taken
    up by the frame loop after the step."""
    import numpy as np  # noqa: PLC0415
    import xblick as db  # noqa: PLC0415
    from xblick.session import timed_out  # noqa: PLC0415

    for name in ("red-tape", "blue-tape"):
        objects.save(tmp_path, name, torch.randn(50, 768),  # NeMO's feature size: frames decoded
                     torch.rand(50, 3) * 2 - 1, torch.rand(3, 3, CROP, CROP),
                     {"source": "test"})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    cam = "test/nemo_slow"
    info = db.CameraInfo.pinhole(320, 240, 300.0, 300.0, 159.5, 119.5, frame_id="cam")
    color = db.CameraPublisher(session, f"{cam}/color", encoding="rgb8", frame_id="cam", camera_info=info,
                               source="test")
    depth = db.CameraPublisher(session, f"{cam}/depth", encoding="16UC1", frame_id="cam", camera_info=info, codec="png",
                               source="test")
    stop = threading.Event()

    def publish():
        rgb, mm, stamp = np.full((240, 320, 3), 128, np.uint8), np.full((240, 320), 1000, np.uint16), 1_000_000_000
        while not stop.is_set():
            color.publish(rgb, stamp_ns=stamp)
            depth.publish(mm, stamp_ns=stamp)
            stamp += 100_000_000
            time.sleep(0.1)

    command = [sys.executable, os.path.join(os.path.dirname(__file__), "slow_step.py"), "4.0", "--objects-dir",
               str(tmp_path), "--object", "red-tape", "--camera", cam, "--detect-every", "0", "--max-hz", "0",
               "--connect", f"127.0.0.1:{port}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    threading.Thread(target=publish, daemon=True).start()

    def alive() -> bool:
        if process.poll() is None:
            return True
        time.sleep(0.5)  # its last lines, the error
        return False

    def ask(key, payload=None):
        began = time.monotonic()
        got = replies(session, key, 3.0, **({"payload": payload} if payload else {}))
        took = time.monotonic() - began
        assert not any(r.err is not None and timed_out(r) for r in got), "".join(output)
        assert len(got) == 1 and got[0].ok is not None, ([r.err.payload.to_string() for r in got if r.err], output)
        return json.loads(got[0].ok.payload.to_bytes()), took

    try:
        deadline = time.monotonic() + 180.0
        while sum("slow step" in line for line in output) < 1:
            assert alive() and time.monotonic() < deadline, "".join(output)
            time.sleep(0.05)
        began = time.monotonic()
        answers = [ask("nemo/cameras"), ask("nemo/objects"), ask("nemo/objects", json.dumps({"add": "blue-tape"})),
                   ask("nemo/cameras", json.dumps({"start": "test/nemo_other"}))]
        assert time.monotonic() - began < 3.0, "the queries outlasted the step: no check"
        assert all(took < 0.5 for _, took in answers), [took for _, took in answers]
        assert answers[0][0] == {"cameras": [cam]}
        assert answers[2][0]["objects"] == {"red-tape": 1, "blue-tape": 2}
        assert answers[3][0] == {"cameras": [cam, "test/nemo_other"]}
        while not any("reading test/nemo_other" in line for line in output):  # the loop, after the step
            assert alive() and time.monotonic() < deadline, "".join(output)
            time.sleep(0.1)
        assert ask("nemo/cameras", json.dumps({"stop": "test/nemo_other"}))[0] == {"cameras": [cam]}
    finally:
        stop.set()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        color.close()
        depth.close()
        session.close()
    assert process.returncode == 0, "".join(output)
