import base64
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from aegis.schemas import Finding, VisionInspectionResult

# Split into independent calls on purpose. A single mega-prompt asking one model call
# to both compare two photos AND judge synthetic-generation likelihood lets the
# comparison conclusion leak into the generation judgment even when explicitly told
# not to -- observed in practice: a model justified a LOW artifact_score by citing how
# well the two photos' background matched, backwards reasoning for a question that
# should be judged from one photo's own pixels alone. Explicit "don't let X influence Y"
# instructions are inherently weak within one continuous completion, since later tokens
# are conditioned on everything written before them. Running these as genuinely separate
# API calls -- each with no visibility into the other's prompt or reasoning -- removes
# the leakage structurally instead of hoping instructions hold, and lets each call focus
# fully on one narrower task instead of splitting attention across several at once.
#
# Deliberately contains no reference to the Red Team's chosen attack technique -- a real
# defender never knows the attacker's method in advance, and leaking it here would
# inflate detection metrics in a way that wouldn't hold up in production.

# Shared persona for all three calls: an adversarial investigator, not a neutral critic.
# Added after observing the model default to a helpful, charitable reading of ambiguous
# evidence -- explaining away a mismatch instead of flagging it -- which is the wrong prior
# for a role whose entire job is catching the submissions specifically designed to survive
# a charitable read. The last paragraph exists to stop that hostility from just becoming a
# fixed "always suspicious" bias: the persona is graded on being RIGHT about fraud, not on
# maximizing rejections, so a genuinely clean photo still has to score low.
_INVESTIGATOR_PERSONA = (
    "You are an adversarial fraud investigator working a payment dispute, not a neutral photo "
    "critic doing the customer a courtesy inspection. The merchant loses real money the moment "
    "you sign off, and a competent fraudster is counting on you to be polite, quick to move on, "
    "and willing to resolve ambiguity in their favor. Do not extend that courtesy: default to "
    "assuming the claim is fraudulent until the evidence in front of you rules it out, hunt "
    "actively for the one inconsistency the customer is hoping you'll skim past, and treat "
    "\"there's an innocent explanation for that\" as worthless unless the pixels actually support "
    "it -- possible is not the same as evidenced.\n\n"
    "That hostility serves accuracy, not a foregone conclusion. You are graded on catching real "
    "fraud, not on manufacturing it: if a photo genuinely survives this level of scrutiny, say so "
    "and score it low. Inventing a finding you can't back with pixel-level evidence is exactly "
    "the kind of sloppy, credulous work a careless analyst produces -- being careless in the "
    "customer's favor is no better than being careless against them."
)

_ARTIFACT_SYSTEM_PROMPT = (
    f"{_INVESTIGATOR_PERSONA}\n\n"
    "You are given ONE customer-submitted photo of claimed product damage. Your ONLY job is to "
    "judge whether THIS PHOTO ON ITS OWN shows signs of being AI-generated, digitally edited, or "
    "otherwise manipulated. You have no other photo to compare it against and should not reason "
    "about consistency, cross-angle matching, or anything relational -- that question is handled "
    "elsewhere, by a different investigator. Interrogate this image's own pixels: lighting/shadow "
    "physics (a single plausible light source vs. impossible shadows), sensor-noise/grain realism, "
    "and edge/texture artifacts at the damage boundary (blending or warping). Do not give this "
    "photo credit for looking convincing at a glance -- that is precisely what a good forgery is "
    "designed to do. That said, most genuine, unedited photos should still score well under 0.3: "
    "do not score high just because the damage itself looks plausible, and do not manufacture an "
    "artifact that isn't really there just to justify suspicion -- whether the story is believable "
    "is a different question from whether these specific pixels show signs of manipulation."
)

_IDENTITY_SYSTEM_PROMPT = (
    f"{_INVESTIGATOR_PERSONA}\n\n"
    "You are given TWO customer-submitted photos of claimed product damage, taken from different "
    "angles. Your ONLY job is to determine whether these two photos show the EXACT SAME physical "
    "unit of the product -- not just the same product type or model, but the literal same "
    "individual item, matched down to every visible detail. Assume they are two different units "
    "photographed to look like one until every checked feature proves otherwise. Do NOT assess "
    "whether the damage position is plausible across the angle change, and do NOT judge whether "
    "either image looks AI-generated; both are assessed separately, by different investigators, "
    "and are not your concern here.\n\n"
    "Go through every visible surface systematically, nook and cranny -- do not summarize "
    "broadly, and do not let a good overall match talk you out of checking the small things. For "
    "each surface/feature you can see in both photos, explicitly state whether it matches: exact "
    "color, finish, and sheen; logo and label placement, orientation, and font; material texture "
    "and grain; any pre-existing scratches, scuffs, dents, stickers, or dust unrelated to the "
    "claimed damage; cable routing, screw or seam positions; and any visible serial numbers or "
    "unique markings. A sophisticated forgery can nail the overall gestalt while getting ONE "
    "specific micro-detail wrong -- your job is to find that detail if it exists, by actually "
    "checking each feature rather than forming one overall impression and stopping there."
)

_COMPARISON_SYSTEM_PROMPT = (
    f"{_INVESTIGATOR_PERSONA}\n\n"
    "You are given TWO customer-submitted photos of claimed product damage, taken from different "
    "angles, plus the customer's stated reason for the dispute. Do NOT perform a general "
    "object-identity/detail match between the two photos (assessed separately, in exhaustive "
    "detail, by a different investigator) and do NOT judge whether either image looks AI-generated "
    "(also assessed separately) -- your only job here is whether the SAME physical damage is being "
    "shown from two angles, or whether the two photos actually show a DIFFERENT damage pattern "
    "dressed up as the same claim. Assume it's the latter until the geometry in front of you "
    "proves otherwise -- judge only from the evidence, and never resolve a shape mismatch in the "
    "customer's favor just because a same-crack explanation is imaginable.\n\n"
    "A single physical crack or tear has a fixed 3D shape. Viewed from a different camera angle, "
    "its apparent size can foreshorten and its apparent position can shift slightly from "
    "parallax -- but its fundamental topology cannot change: whether it is one continuous "
    "opening or several separate ones, roughly what fraction of the surface it spans, and which "
    "direction it runs relative to fixed landmarks (a logo, a hinge, the cup rim, a button) must "
    "stay recognizable as the same feature. A small curved nick near an edge in one photo and a "
    "long diagonal split through the center in the other are NOT plausibly the same crack viewed "
    "from two angles -- that is a strong inconsistency, not a minor one worth explaining away.\n\n"
    "Work through this explicitly, in prose, before scoring:\n"
    "STEP 1 -- In image 1, describe exactly where the damage sits relative to at least two fixed "
    "reference points on the object, and describe its precise shape, extent, and topology (one "
    "opening or several; roughly how much of the surface it covers).\n"
    "STEP 2 -- Do the same for image 2, independently, relative to the same reference points.\n"
    "STEP 3 -- Compare the two descriptions from steps 1-2 directly: is this a plausible same "
    "physical crack under a viewpoint change, or does the fundamental shape/extent/topology "
    "differ enough that these look like two different damage patterns? Also note whether the "
    "damage matches the customer's claimed defect.\n\n"
    "Separately: near-identical noise/grain between the two supposedly independent shots is "
    "itself suspicious and should lower the consistency score, even though overall object "
    "identity is not your concern here."
)


# Extensive testing (multiple runs, two model tiers, maximally rigorous step-by-step
# prompting -- see _COMPARISON_SYSTEM_PROMPT above) confirmed that asking a VLM to
# directly JUDGE cross-angle geometric/topological consistency ("is this the same
# crack shape from two angles?") is unreliable: it repeatedly missed a mismatch a
# human caught instantly. The two checks below work around that by not asking the
# model to judge consistency at all -- only to MEASURE (localization) or to compare
# an isolated, pre-scaled crop instead of the whole scene (shape match). The actual
# consistency judgment for the localization data is arithmetic in Python, not model
# reasoning, so it can't be talked out of flagging a real mismatch by an otherwise
# convincing photo.

_LOCALIZATION_SYSTEM_PROMPT = (
    f"{_INVESTIGATOR_PERSONA}\n\n"
    "You are given TWO customer-submitted photos of claimed product damage, taken from different "
    "angles. Your ONLY job here is precise measurement, not judgment -- you are not deciding "
    "whether anything is consistent or suspicious; a separate deterministic calculation will do "
    "that from the numbers you provide, so report your most accurate estimate rather than "
    "hedging toward the middle or toward a size that would look right.\n\n"
    "Pick exactly two fixed, rigid reference points that are clearly visible in BOTH photos (e.g. "
    "a logo corner, a hinge, a seam, a button) -- name them identically in both images so they can "
    "be matched programmatically. For each photo independently, report their normalized (x, y) "
    "position (0,0 = top-left corner, 1,1 = bottom-right corner, as fractions of that photo's own "
    "width and height). Then report a tight bounding box around the claimed damage in that photo, "
    "same normalized coordinates. Be as precise as you can -- do not pad the box generously 'to "
    "be safe', and do not silently guess if the damage is partly out of frame; a loose or careless "
    "box defeats the entire point of this measurement."
)

_SHAPE_MATCH_SYSTEM_PROMPT = (
    f"{_INVESTIGATOR_PERSONA}\n\n"
    "You are given ONE composite image: two close-up crops of claimed product damage from two "
    "different customer photos, placed side by side and already scaled so neither crop is "
    "artificially larger than the other -- you don't need to mentally correct for zoom "
    "differences. Your ONLY job is to judge whether the left and right crops show the same "
    "physical damage -- same shape, same extent, same topology -- allowing for the natural "
    "change in apparent angle/foreshortening between two photos of a genuine 3D object, but not "
    "for a fundamentally different shape. A small curved nick on one side and a long diagonal "
    "split on the other are NOT the same damage, regardless of how convincing each individually "
    "looks in isolation."
)


def _run_artifact_check(vision_provider, image_b64: str, claimed_reason: str) -> dict:
    prompt = f"Customer's claimed reason: {claimed_reason}\n\n{_ARTIFACT_SYSTEM_PROMPT}"
    raw = vision_provider.inspect_images([image_b64], prompt, schema_hint="vision_artifact")
    return json.loads(raw)


def _run_identity_check(vision_provider, images_b64: list[str]) -> dict:
    raw = vision_provider.inspect_images(images_b64, _IDENTITY_SYSTEM_PROMPT, schema_hint="vision_identity")
    return json.loads(raw)


def _run_comparison_check(vision_provider, images_b64: list[str], claimed_reason: str) -> dict:
    prompt = f"Customer's claimed reason: {claimed_reason}\n\n{_COMPARISON_SYSTEM_PROMPT}"
    raw = vision_provider.inspect_images(images_b64, prompt, schema_hint="vision_comparison")
    return json.loads(raw)


def _run_localization_check(vision_provider, images_b64: list[str]) -> dict:
    raw = vision_provider.inspect_images(images_b64, _LOCALIZATION_SYSTEM_PROMPT, schema_hint="vision_localization")
    return json.loads(raw)


def _run_shape_match_check(vision_provider, composite_b64: str, claimed_reason: str) -> dict:
    prompt = f"Customer's claimed reason: {claimed_reason}\n\n{_SHAPE_MATCH_SYSTEM_PROMPT}"
    raw = vision_provider.inspect_images([composite_b64], prompt, schema_hint="vision_shape_match")
    return json.loads(raw)


def _project_onto_landmark_axis(landmark_a: dict, landmark_b: dict, point: dict) -> tuple[float, float, float]:
    """Expresses `point` in a coordinate frame defined by the two landmarks: how far
    along the line from A to B it sits (as a fraction of the A-B distance), and how far
    off that line perpendicular to it (also as a fraction of the A-B distance). This
    frame is invariant to translation, rotation, and uniform scale -- exactly the
    degrees of freedom a different camera angle introduces -- so a genuinely same crack
    should land at roughly the same (parallel, perpendicular) coordinates in both
    photos even though the raw pixel coordinates differ completely. Returns the A-B
    distance too, needed separately to scale-normalize the damage's own extent."""
    dx, dy = landmark_b["x"] - landmark_a["x"], landmark_b["y"] - landmark_a["y"]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 0.0, 0.0
    unit_x, unit_y = dx / length, dy / length
    px, py = point["x"] - landmark_a["x"], point["y"] - landmark_a["y"]
    parallel = (px * unit_x + py * unit_y) / length
    perpendicular = (px * -unit_y + py * unit_x) / length
    return parallel, perpendicular, length


def _bbox_center(bbox: dict) -> dict:
    return {"x": (bbox["x_min"] + bbox["x_max"]) / 2, "y": (bbox["y_min"] + bbox["y_max"]) / 2}


def _bbox_diagonal(bbox: dict) -> float:
    return math.hypot(bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"])


# Generous by design: these coordinates come from a VLM eyeballing pixel fractions,
# not precise pixel measurement, so some drift is expected even for a genuinely
# consistent crack under a real viewpoint change. These tolerances are a first pass --
# calibrated to the real Sony ULT WEAR example when this check was built, not proven
# against a large labeled batch -- and worth revisiting once more real examples exist.
_POSITION_DRIFT_TOLERANCE = 0.35
_EXTENT_DRIFT_TOLERANCE = 0.45


def _geometric_consistency_score(localization: dict) -> tuple[float, str]:
    """The actual consistency judgment for the localization data -- arithmetic, not
    model reasoning, so unlike the holistic prompt it cannot be talked out of flagging
    a real position/extent mismatch by an otherwise-convincing photo."""
    image_1, image_2 = localization["image_1"], localization["image_2"]
    center_1, center_2 = _bbox_center(image_1["damage_bbox"]), _bbox_center(image_2["damage_bbox"])
    parallel_1, perpendicular_1, scale_1 = _project_onto_landmark_axis(
        image_1["landmark_a"], image_1["landmark_b"], center_1
    )
    parallel_2, perpendicular_2, scale_2 = _project_onto_landmark_axis(
        image_2["landmark_a"], image_2["landmark_b"], center_2
    )
    extent_1 = _bbox_diagonal(image_1["damage_bbox"]) / scale_1 if scale_1 > 1e-6 else 0.0
    extent_2 = _bbox_diagonal(image_2["damage_bbox"]) / scale_2 if scale_2 > 1e-6 else 0.0

    position_drift = math.hypot(parallel_1 - parallel_2, perpendicular_1 - perpendicular_2)
    extent_drift = abs(extent_1 - extent_2)

    score = max(0.0, 1.0 - min(1.0, position_drift / _POSITION_DRIFT_TOLERANCE + extent_drift / _EXTENT_DRIFT_TOLERANCE))
    detail = (
        f"Landmark-relative damage position: {parallel_1:.2f}/{perpendicular_1:.2f} in image 1 vs "
        f"{parallel_2:.2f}/{perpendicular_2:.2f} in image 2 (drift {position_drift:.2f}); relative "
        f"size {extent_1:.2f} vs {extent_2:.2f} (drift {extent_drift:.2f})."
    )
    return score, detail


# Extra margin around the reported bbox so a genuine damage-adjacent landmark (an
# edge, a seam) often stays in frame for context, without pulling in so much of the
# rest of the product that the crop stops isolating the comparison.
_CROP_PADDING_FRACTION = 0.35
_COMPOSITE_TARGET_HEIGHT = 420
_COMPOSITE_DIVIDER_WIDTH = 6


def _decode_png(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _crop_to_bbox(image: Image.Image, bbox: dict) -> Image.Image:
    width, height = image.size
    pad_x = (bbox["x_max"] - bbox["x_min"]) * _CROP_PADDING_FRACTION
    pad_y = (bbox["y_max"] - bbox["y_min"]) * _CROP_PADDING_FRACTION
    left = max(0.0, bbox["x_min"] - pad_x) * width
    top = max(0.0, bbox["y_min"] - pad_y) * height
    right = min(1.0, bbox["x_max"] + pad_x) * width
    bottom = min(1.0, bbox["y_max"] + pad_y) * height
    return image.crop((int(left), int(top), int(max(right, left + 1)), int(max(bottom, top + 1))))


def _build_focused_composite(image_1_b64: str, bbox_1: dict, image_2_b64: str, bbox_2: dict) -> str:
    """Crops both photos down to just the claimed-damage region (option 2) and pastes
    them side by side, each independently rescaled to the same height (option 3) --
    so the comparison call sees only the damage itself, at a comparable apparent size,
    instead of having to isolate it from the rest of the product and correct for zoom
    differences at the same time. Deliberately independent resizing rather than a
    zoom-matched rescale: it preserves each crop's own aspect ratio, which itself
    carries shape information (a long diagonal split crops much wider than a small
    curved nick) -- exactly the difference a same-shape judgment needs to see."""

    def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
        ratio = height / image.height
        return image.resize((max(1, round(image.width * ratio)), height))

    crop_1 = _resize_to_height(_crop_to_bbox(_decode_png(image_1_b64), bbox_1), _COMPOSITE_TARGET_HEIGHT)
    crop_2 = _resize_to_height(_crop_to_bbox(_decode_png(image_2_b64), bbox_2), _COMPOSITE_TARGET_HEIGHT)
    composite = Image.new(
        "RGB", (crop_1.width + _COMPOSITE_DIVIDER_WIDTH + crop_2.width, _COMPOSITE_TARGET_HEIGHT), (255, 255, 255)
    )
    composite.paste(crop_1, (0, 0))
    composite.paste(crop_2, (crop_1.width + _COMPOSITE_DIVIDER_WIDTH, 0))
    return _encode_png(composite)


def inspect(vision_provider, images_b64: list[str], claimed_reason: str) -> VisionInspectionResult:
    """Pipelines the inspection into independent calls instead of one combined prompt: a
    synthetic-generation check per image, a dedicated exhaustive same-object-identity
    check (drives detail_consistency_score), a narrower holistic comparison check
    (semantic_match + one vote on cross-angle plausibility), a pure-measurement
    localization check, and -- once localization returns -- a focused shape-match check
    on a cropped composite built from it. The first four run concurrently since none
    depends on another's output; the shape-match call necessarily waits on localization
    since it needs the bounding boxes to build its input image. See the module
    docstring above _LOCALIZATION_SYSTEM_PROMPT for why the last two checks exist."""
    with ThreadPoolExecutor(max_workers=len(images_b64) + 3) as pool:
        artifact_futures = [
            pool.submit(_run_artifact_check, vision_provider, image_b64, claimed_reason)
            for image_b64 in images_b64
        ]
        identity_future = pool.submit(_run_identity_check, vision_provider, images_b64)
        comparison_future = pool.submit(_run_comparison_check, vision_provider, images_b64, claimed_reason)
        localization_future = pool.submit(_run_localization_check, vision_provider, images_b64)
        artifact_results = [future.result() for future in artifact_futures]
        identity_result = identity_future.result()
        comparison_result = comparison_future.result()
        localization_result = localization_future.result()

    geometric_score, geometric_detail = _geometric_consistency_score(localization_result)

    composite_b64 = _build_focused_composite(
        images_b64[0],
        localization_result["image_1"]["damage_bbox"],
        images_b64[1],
        localization_result["image_2"]["damage_bbox"],
    )
    shape_match_result = _run_shape_match_check(vision_provider, composite_b64, claimed_reason)
    shape_match_score = shape_match_result["shape_match_score"]

    holistic_score = comparison_result["angle_consistency_score"]
    # min, not average, mirroring the artifact_score max above: if ANY of the three
    # independent consistency methods lands low, that shouldn't get diluted by the
    # other two -- especially since the whole point of adding the geometric/shape-match
    # checks was that the holistic method alone can be talked into a false high score.
    angle_consistency_score = min(holistic_score, geometric_score, shape_match_score)
    consistency_disagreement = max(holistic_score, geometric_score, shape_match_score) - angle_consistency_score

    # max, not average: if EITHER photo independently shows strong signs of being
    # synthetic, that shouldn't get diluted by the other photo looking clean.
    artifact_score = max(result["artifact_score"] for result in artifact_results)
    # Each artifact check runs on exactly one photo with no view of the other, so two
    # findings of the same type (e.g. "sensor_noise" from image 1 and image 2) are
    # easy to mistake for a duplicate rather than two independent per-photo judgments
    # -- label which photo each came from to make that unambiguous in the merged list.
    artifact_findings = [
        Finding(
            type=finding["type"],
            confidence=finding["confidence"],
            description=f"Image {index + 1}: {finding['description']}",
        )
        for index, result in enumerate(artifact_results)
        for finding in result.get("findings", [])
    ]
    identity_findings = [Finding(**finding) for finding in identity_result.get("findings", [])]
    comparison_findings = [Finding(**finding) for finding in comparison_result.get("findings", [])]
    shape_match_findings = [Finding(**finding) for finding in shape_match_result.get("findings", [])]
    # Surfaces the three sub-scores that combined into angle_consistency_score even
    # though they're not separate top-level metrics elsewhere -- lets a viewer see
    # *why* the combined score landed where it did instead of just the one number.
    consistency_breakdown_findings = [
        Finding(
            type="consistency_submethod_holistic",
            confidence=holistic_score,
            description=f"Full-image reasoning check scored cross-angle consistency at {holistic_score:.2f}.",
        ),
        Finding(
            type="consistency_submethod_geometric",
            confidence=geometric_score,
            description=f"Deterministic landmark-relative geometry check scored consistency at "
            f"{geometric_score:.2f}. {geometric_detail}",
        ),
        Finding(
            type="consistency_submethod_shape_match",
            confidence=shape_match_score,
            description=f"Focused cropped-and-scaled shape comparison scored consistency at {shape_match_score:.2f}.",
        ),
    ]

    return VisionInspectionResult(
        artifact_score=artifact_score,
        angle_consistency_score=angle_consistency_score,
        geometric_consistency_score=geometric_score,
        shape_match_score=shape_match_score,
        consistency_disagreement=round(consistency_disagreement, 3),
        detail_consistency_score=identity_result["detail_consistency_score"],
        semantic_match=comparison_result["semantic_match"],
        findings=artifact_findings
        + identity_findings
        + comparison_findings
        + shape_match_findings
        + consistency_breakdown_findings,
    )
