# Accuracy

This page has the benchmark of the node against a mesh pipeline and the results of the live node. All results are
from the simulation, status of 2026-11-01.

## The benchmark

Setup:

- A simulated scene with seven objects of our own object set. None of them is in YCB.
- The camera is the Roboception pair with the depth of durchblick-fast-foundationstereo.
- The crops use masks of the true silhouette, so the benchmark measures the pose alone.
- The templates are 4 frames of a first recording. The queries are every other frame in which the object is visible.
- We align a method's object frame to the true frame once per object. Per query we compute ADD-S and ADD in
  millimetres.
- The reference is our mesh pipeline on the same inputs. RecGen makes a mesh from the same 4 templates, and
  FoundationPose registers it for each query.

ADD is the mean distance between the points of the object at the estimated pose and the same points at the true
pose. ADD-S uses the nearest point in place of the same point, which suits symmetric objects.

| | NeMO (this node) | RecGen mesh + FoundationPose |
| --- | --- | --- |
| queries from the templates' side, median ADD-S / ADD | 1.7 / 1.8 mm | 2.5 / 3.3 mm |
| objects turned by 150 deg, median ADD-S / ADD | 1.9 / 2.3 mm | 2.6 / 3.5 mm |
| worst object (ADD, turned) | 3.7 mm (hand-wash tub) | 13.8 mm (hand-wash tub) |
| time per query | 10 ms | 1.2 s |
| YCB objects (4), median ADD-S | 1.7 mm | 2.0 mm |

The turned objects show sides that no template saw. The hand-wash tub is nearly symmetric, which explains the
13.8 mm of the mesh pipeline. NeMO's own code reaches the same accuracy as the node on the same queries.

## Masks

NeMO's masks reach an IoU of 0.59 to 0.92 against the true silhouettes, with a median of about 0.9. SAM 3.1 with
text prompts such as "spray bottle" and "tube" found five of the seven objects in hardly any frame (IoU 0).

## The node, live

Setup: the turned scene at 10 frames a second. The node detects and tracks on its own.

- The node poses all seven objects at once at the camera's 10 frames a second.
- The median ADD-S is 1.4 to 3.6 mm per object.
- Every pose is below 10 % of the object's diameter.
- The node finds each object in 71 to 100 % of the frames. The front row leaves the image while the head sweeps.

## As the segmenter of durchblick-foundationpose

Setup: the same live scene, the node with `--segments`, FoundationPose with the objects' true meshes from that
object set.

FoundationPose reaches an ADD-S of 0.9 to 1.1 mm for four objects. Where a true mesh exists, FoundationPose on
NeMO's masks is therefore more accurate than NeMO's own pose.

FoundationPose never accepted the whisk and the mounting adhesive. Their NeMO masks overlap the truth least, at
about 0.6. They most likely stay below FoundationPose's `--accept-iou` of 0.75. We did not check this for each
registration.

## Status and limits

- We tested the node in the simulation only. We have not tested it on a robot.
- The benchmark uses true silhouettes for the crops. The live results use the node's own masks.
