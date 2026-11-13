# Performance

This page states how long the node needs per frame, which design choices make it fast, and what is still open. The
measurements are from the simulation, status of 2026-11-01.

## Results

| What | Time |
| --- | --- |
| one tracked object, per frame | 10 ms: decoder 6 ms, pose 4 ms |
| NeMO's own code, same queries, same accuracy | 43 to 93 ms |
| mesh pipeline (RecGen mesh, FoundationPose registration), per query | 1.2 s |
| search of the whole frame, per object, 640 x 480, bfloat16 | 61 ms |
| search of the whole frame, per object, 640 x 480, float32 | 160 ms |
| the live node with seven objects | the camera's 10 frames a second |

## What makes the node fast

### 1. RANSAC on the GPU

NeMO's own pose step solves PnP with RANSAC on the CPU. RANSAC fits a model to many small random samples and keeps
the fit that most data points agree with. On a CPU, a loop tests one hypothesis after the other.

The node fits a similarity transform to 3D point pairs. It evaluates all hypotheses at once on the GPU, as
batched tensor operations, and then fits again on the inliers. The whole pose step takes 4 ms.
[design.md](design.md#the-metric-pose) explains the fit.

NeMO's own code needs 43 to 93 ms per query. The node needs 10 ms at the same accuracy.

### 2. The decoder computes in bfloat16

bfloat16 is a floating-point format with 16 bits. It keeps the value range of float32 and has fewer digits of
precision. The GPU computes it faster and moves half the memory.

A search of the whole frame takes 61 ms per object in bfloat16 and 160 ms in float32. The poses are as accurate as
with float32 on the benchmark. `--autocast none` selects float32.

### 3. One batch for all tracked objects

The node cuts one square crop per tracked object and decodes all crops of a frame in one batch. The GPU then does
the work of all objects in one pass through the decoder. Seven objects keep the camera's 10 frames a second.

### 4. The whole-frame search is rare

NeMO's detection decodes the whole frame in patches of 128 pixels. This takes 61 ms per object, which is hundreds of
milliseconds with several objects. The node therefore searches only for lost objects, and at most every 0.5 s per
object and camera (`--detect-every`). A tracked object needs only its crop.

### 5. The node takes only the frames it can process

The node processes about 10 frames a second. `--max-hz` limits the colour frames to 20 a second per camera, and the
node always takes the newest one. It decodes the depth only for the frames that it takes.

## What did not help

We have no record of a change that we tried and then removed.

## Status and limits

- The decoder does not use TensorRT. TensorRT is NVIDIA's compiler that turns a network into an optimised engine
  for one GPU model.
- The decoder does not use CUDA graphs. A CUDA graph records a fixed sequence of GPU operations once and replays it
  with one call, which saves the cost of launching each operation.
- The decoder takes 6 ms, so both would save little.
- We have not measured the node on a robot's GPU beside other nodes.
