"""Deterministic, non-LLM metadata forensics on raw uploaded image bytes.

Real insurance-industry fraud detection leans heavily on exactly this: camera EXIF
(make/model/timestamp) and, increasingly, C2PA content-credential manifests that
AI generation tools embed. Unlike the vision model's judgment, this signal can't
hallucinate -- it's a plain byte-level check. The tradeoff is that absence proves
nothing: ordinary processing (screenshots, many messaging apps, some upload
pipelines) strips EXIF from genuine photos too, so this is scored as one modest
input among several, never a standalone verdict.

Deliberately only wired into the Try It Yourself tab (see app.py), not the shared
Red Team / Batch Evaluation pipeline: every image on that path -- fraud attempts
and the legit_samples.json fixtures alike -- is synthetically rendered by this same
codebase, so none of it has real-world metadata to check. Applying this signal
there would flag everything identically for a meaningless reason and break the
batch-eval false-positive metric.
"""

import io
import struct

from PIL import Image
from PIL.ExifTags import TAGS

from aegis.schemas import ImageMetadataReport

_JPEG_APP11 = 0xEB  # reserved specifically for JUMBF (C2PA) metadata in JPEG, ISO 19566-5
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_C2PA_CHUNK = b"caBX"  # the standard PNG ancillary chunk C2PA manifests are stored in

# IPTC Photo Metadata's DigitalSourceType is a plain, human-readable XMP field (no
# crypto, unlike C2PA) that Google/OpenAI/Adobe and others set to declare AI-involved
# content -- this is the practical "was this made/edited by a generative AI tool"
# marker actually embedded by tools like Gemini's image generation, distinct from the
# separate SynthID invisible watermark (which requires Google's own detector to read
# and isn't something this codebase can decode).
_XMP_PACKET_START = b"<?xpacket begin="
_XMP_PACKET_END = b"<?xpacket end="
_AI_DIGITAL_SOURCE_MARKERS = (
    b"trainedalgorithmicmedia",
    b"compositewithtrainedalgorithmicmedia",
    b"algorithmicmedia",
)


def _extract_exif(raw_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        exif = image.getexif()
    except Exception:
        return {}
    if not exif:
        return {}
    return {TAGS.get(tag_id, tag_id): value for tag_id, value in exif.items()}


def _has_c2pa_marker_jpeg(raw_bytes: bytes) -> bool:
    if not raw_bytes.startswith(b"\xff\xd8"):
        return False
    i = 2
    while i + 4 <= len(raw_bytes):
        if raw_bytes[i] != 0xFF:
            break
        marker = raw_bytes[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of entropy-coded scan data -- no more marker segments follow
            break
        if marker == _JPEG_APP11:
            return True
        length = struct.unpack(">H", raw_bytes[i + 2 : i + 4])[0]
        i += 2 + length
    return False


def _has_c2pa_marker_png(raw_bytes: bytes) -> bool:
    if not raw_bytes.startswith(_PNG_SIGNATURE):
        return False
    i = len(_PNG_SIGNATURE)
    while i + 8 <= len(raw_bytes):
        length = struct.unpack(">I", raw_bytes[i : i + 4])[0]
        chunk_type = raw_bytes[i + 4 : i + 8]
        if chunk_type == _PNG_C2PA_CHUNK:
            return True
        i += 8 + length + 4  # length field + type + data + CRC
    return False


def _has_c2pa_marker(raw_bytes: bytes) -> bool:
    """Parses actual JPEG marker / PNG chunk structure rather than scanning for the
    literal bytes "c2pa"/"jumb" anywhere in the file -- a naive substring scan
    produces false positives from ordinary compressed image data (verified against
    a real photo: the 4-byte sequence "c2PA" turned up by chance inside the JPEG's
    entropy-coded scan data, with no manifest structure anywhere near it)."""
    try:
        return _has_c2pa_marker_jpeg(raw_bytes) or _has_c2pa_marker_png(raw_bytes)
    except (struct.error, IndexError):
        return False


def _has_ai_digital_source_marker(raw_bytes: bytes) -> bool:
    """Looks for an IPTC DigitalSourceType declaring AI involvement, but ONLY inside
    an actual XMP packet (anchored on its fixed <?xpacket begin=...?> / <?xpacket
    end=...?> markers), not anywhere in the raw file. XMP's packet wrapper is
    effectively impossible to produce by coincidence in compressed image data --
    unlike the 4-byte "c2pa"/"jumb" strings that caused a real false positive earlier
    -- so anchoring on it first keeps this check from repeating that mistake."""
    start = raw_bytes.find(_XMP_PACKET_START)
    if start == -1:
        return False
    end = raw_bytes.find(_XMP_PACKET_END, start)
    if end == -1:
        return False
    closing = raw_bytes.find(b"?>", end)
    if closing == -1:
        return False
    packet = raw_bytes[start : closing + 2].lower()
    return b"digitalsourcetype" in packet and any(marker in packet for marker in _AI_DIGITAL_SOURCE_MARKERS)


def inspect_metadata(raw_bytes: bytes) -> ImageMetadataReport:
    exif = _extract_exif(raw_bytes)
    make = exif.get("Make")
    model = exif.get("Model")
    timestamp = exif.get("DateTimeOriginal") or exif.get("DateTime")
    has_camera_exif = bool(make or model or timestamp)
    has_c2pa = _has_c2pa_marker(raw_bytes)
    has_ai_digital_source = _has_ai_digital_source_marker(raw_bytes)

    notes = []
    if not has_camera_exif:
        notes.append(
            "No camera make/model/timestamp found in EXIF -- suggestive of a screenshot, download, "
            "or AI-generated image, but ordinary processing can also strip this from a genuine "
            "photo, so this alone is not conclusive."
        )
    if has_c2pa:
        notes.append(
            "A C2PA content-credential marker was found in the file -- this is a strong signal it "
            "declares AI generation or editing provenance."
        )
    if has_ai_digital_source:
        notes.append(
            "An IPTC DigitalSourceType marker declaring AI-generated/edited content was found in "
            "the file's XMP metadata."
        )

    return ImageMetadataReport(
        has_camera_exif=has_camera_exif,
        camera_make=str(make) if make else None,
        camera_model=str(model) if model else None,
        capture_timestamp=str(timestamp) if timestamp else None,
        has_c2pa_marker=has_c2pa,
        has_ai_digital_source_marker=has_ai_digital_source,
        notes=notes,
    )


def combine_metadata_reports(reports: list[ImageMetadataReport]) -> ImageMetadataReport:
    """Two images -> one representative report. Both photos need real camera EXIF to
    count as having it; either one carrying a C2PA or AI digital-source marker is
    enough to flag."""
    if not reports:
        return ImageMetadataReport(has_camera_exif=False)
    return ImageMetadataReport(
        has_camera_exif=all(r.has_camera_exif for r in reports),
        camera_make=next((r.camera_make for r in reports if r.camera_make), None),
        camera_model=next((r.camera_model for r in reports if r.camera_model), None),
        capture_timestamp=next((r.capture_timestamp for r in reports if r.capture_timestamp), None),
        has_c2pa_marker=any(r.has_c2pa_marker for r in reports),
        has_ai_digital_source_marker=any(r.has_ai_digital_source_marker for r in reports),
        notes=[note for r in reports for note in r.notes],
    )


def metadata_risk_score(report: ImageMetadataReport) -> float:
    """A modest, honestly-calibrated nudge -- never a standalone verdict, per the
    module docstring. A self-declared AI marker (C2PA or IPTC DigitalSourceType) is
    much stronger evidence than EXIF absence."""
    if report.has_c2pa_marker or report.has_ai_digital_source_marker:
        return 0.9
    if not report.has_camera_exif:
        return 0.4
    return 0.05
