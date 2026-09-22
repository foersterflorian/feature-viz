# Asset provenance

Every file in this directory that did not originate in this project is listed
here with its source, licence and checksum, so that the origin can be
re-verified without trusting this file.

---

## `street-scene.jpg`

The demonstration still. Chosen because YOLO's COCO classes are everyday
objects, and because a street scene yields both one large foreground object and
many small ones — which is what makes the depth gradient in the feature maps
legible (see `DECISIONS.md` §8).

| | |
|---|---|
| **Title** | San Francisco Cable Cars. (40956932812).jpg |
| **Author** | Bernard Spragg. NZ, Christchurch, New Zealand |
| **Licence** | **CC0 1.0 Universal** (public domain dedication) |
| **Licence text** | <https://creativecommons.org/publicdomain/zero/1.0/> |
| **Source page** | <https://commons.wikimedia.org/wiki/File:San_Francisco_Cable_Cars._(40956932812).jpg> |
| **Direct file** | <https://upload.wikimedia.org/wikipedia/commons/b/bb/San_Francisco_Cable_Cars._%2840956932812%29.jpg> |
| **Created** | 2017-10-17 |
| **Retrieved** | 2026-09-22 |
| **Original size** | 4490 x 2928 px, 9 057 537 bytes |
| **SHA-1** | `c849b740e39e24e85ec363f4c3a7c803c8f46875` |
| **SHA-256** | `7e3261d2eb5297b3a85df3fca86f441b758b0f6909f4550c44af11f0f5bd1754` |
| **Restrictions on Commons** | none stated |

The file is stored unmodified. Its SHA-1 was compared against the `sha1` that
the Commons API reports for the file and matched, so the local copy is
byte-identical to what Commons serves.

To re-verify:

```bash
sha256sum assets/street-scene.jpg
curl -s "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=sha1|extmetadata&titles=File:San%20Francisco%20Cable%20Cars.%20(40956932812).jpg"
```

The machine-readable form of the same record, as returned by the Commons API,
is in `street-scene.json`.

**On CC0 and depicted persons.** CC0 waives the photographer's copyright. It
does not address the personality rights of people in the photograph. This image
was chosen partly because it contains no prominent, identifiable face: one
person is visible behind the cable car's window, small and not clearly
recognisable. For any use beyond this demonstrator, that judgement should be
made again.

---

## `sample.mp4`

Generated from `street-scene.jpg` by this project; not third-party material. It
inherits the CC0 status of its source.

A ten-second 1280x720 H.264 clip built as a slow horizontal pan across the
still, with a slight brightness drift so that consecutive frames differ.
Encoding goes through ffmpeg over a pipe — see the generator's docstring for
why OpenCV cannot do it here. The reason a clip exists rather than the still
itself: a single image yields exactly one
frame from `model.predict`, and the FPS counter drawn onto the canvas is
smoothed with `0.9 * fps + 0.1 * ...` from a starting value of 0.0 — on one
frame it would show a meaningless number. The pan also keeps the EMA of the
normalisation scale in motion, which is closer to the live case.

The generator is `tools/make_sample.py`, so the clip can be rebuilt from the
still at any time.
