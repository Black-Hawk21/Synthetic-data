import io
import struct

from PIL import Image

from aegis.blueteam.metadata_forensics import combine_metadata_reports, inspect_metadata, metadata_risk_score
from aegis.schemas import ImageMetadataReport


def _jpeg_bytes(with_camera_exif: bool) -> bytes:
    image = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    if with_camera_exif:
        exif = image.getexif()
        exif[271] = "Apple"  # Make
        exif[272] = "iPhone 14 Pro"  # Model
        image.save(buf, format="JPEG", exif=exif)
    else:
        image.save(buf, format="JPEG")
    return buf.getvalue()


def _jpeg_bytes_with_app11_marker() -> bytes:
    """A real JPEG with an actual APP11 (0xFFEB) marker segment spliced in right
    after SOI -- the standard, structurally-correct way JUMBF/C2PA data is embedded,
    as opposed to the literal bytes "c2pa" merely appearing somewhere in the file."""
    plain = _jpeg_bytes(with_camera_exif=False)
    payload = b"JP\x00\x00c2pa fake manifest payload"
    app11_segment = b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload
    return plain[:2] + app11_segment + plain[2:]


def _png_bytes_with_cabx_chunk() -> bytes:
    image = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    plain = buf.getvalue()
    data = b"fake c2pa manifest bytes"
    chunk = struct.pack(">I", len(data)) + b"caBX" + data + struct.pack(">I", 0)
    # Splice the chunk in right after the PNG signature (before IHDR) -- position
    # relative to other chunks doesn't matter for this scan, only that it parses.
    return plain[:8] + chunk + plain[8:]


def test_image_with_no_exif_is_flagged_but_not_conclusively():
    report = inspect_metadata(_jpeg_bytes(with_camera_exif=False))
    assert report.has_camera_exif is False
    assert report.camera_make is None
    assert report.notes  # explains the caveat, doesn't just silently flag


def test_image_with_camera_exif_is_recognized():
    report = inspect_metadata(_jpeg_bytes(with_camera_exif=True))
    assert report.has_camera_exif is True
    assert report.camera_make == "Apple"
    assert report.camera_model == "iPhone 14 Pro"
    assert not report.notes


def test_c2pa_bytes_appearing_outside_a_real_marker_segment_are_not_flagged():
    # Regression test for a real false positive: the literal bytes "c2pa" turned up
    # by chance inside a genuine JPEG's compressed scan data, with no actual JUMBF
    # marker structure anywhere in the file. A naive substring scan flagged it;
    # proper marker-based parsing must not.
    plain_jpeg = _jpeg_bytes(with_camera_exif=False) + b"...c2pa...jumb..." + b"\x00" * 50
    assert inspect_metadata(plain_jpeg).has_c2pa_marker is False


def test_real_jpeg_app11_marker_is_detected():
    report = inspect_metadata(_jpeg_bytes_with_app11_marker())
    assert report.has_c2pa_marker is True


def test_real_png_cabx_chunk_is_detected():
    report = inspect_metadata(_png_bytes_with_cabx_chunk())
    assert report.has_c2pa_marker is True


def _jpeg_bytes_with_xmp_digital_source_type(value: bytes = b"trainedAlgorithmicMedia") -> bytes:
    """A real JPEG with an actual XMP packet (in an APP1 segment) declaring an IPTC
    DigitalSourceType -- the standard, structurally-correct embedding, as opposed to
    the literal marker string merely appearing somewhere in the file."""
    plain = _jpeg_bytes(with_camera_exif=False)
    xmp_xml = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF>'
        b'<rdf:Description Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/'
        b'digitalsourcetype/' + value + b'"/>'
        b"</rdf:RDF></x:xmpmeta>"
        b'<?xpacket end="w"?>'
    )
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_xml
    app1_segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return plain[:2] + app1_segment + plain[2:]


def test_ai_digital_source_bytes_outside_a_real_xmp_packet_are_not_flagged():
    # Same class of regression as the C2PA false positive: the marker string alone,
    # with no real XMP packet structure around it, must not be flagged.
    plain_jpeg = _jpeg_bytes(with_camera_exif=False) + b"...trainedAlgorithmicMedia..." + b"\x00" * 50
    assert inspect_metadata(plain_jpeg).has_ai_digital_source_marker is False


def test_real_xmp_digital_source_type_marker_is_detected():
    report = inspect_metadata(_jpeg_bytes_with_xmp_digital_source_type())
    assert report.has_ai_digital_source_marker is True


def test_xmp_packet_without_the_ai_marker_is_not_flagged():
    # An XMP packet is common in ordinary photos (most phones/editors write one) --
    # only the specific AI-declaring DigitalSourceType value should trigger this.
    plain = _jpeg_bytes(with_camera_exif=False)
    xmp_xml = (
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF>'
        b'<rdf:Description photoshop:Credit="Some Photographer"/>'
        b"</rdf:RDF></x:xmpmeta>"
        b'<?xpacket end="w"?>'
    )
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_xml
    app1_segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    jpeg_with_plain_xmp = plain[:2] + app1_segment + plain[2:]
    assert inspect_metadata(jpeg_with_plain_xmp).has_ai_digital_source_marker is False


def test_combine_requires_both_images_to_have_exif():
    combined = combine_metadata_reports(
        [
            ImageMetadataReport(has_camera_exif=True, camera_make="Apple"),
            ImageMetadataReport(has_camera_exif=False),
        ]
    )
    assert combined.has_camera_exif is False
    assert combined.camera_make == "Apple"


def test_combine_flags_c2pa_if_either_image_has_it():
    combined = combine_metadata_reports(
        [ImageMetadataReport(has_camera_exif=True), ImageMetadataReport(has_camera_exif=True, has_c2pa_marker=True)]
    )
    assert combined.has_c2pa_marker is True


def test_combine_flags_ai_digital_source_marker_if_either_image_has_it():
    combined = combine_metadata_reports(
        [
            ImageMetadataReport(has_camera_exif=True),
            ImageMetadataReport(has_camera_exif=True, has_ai_digital_source_marker=True),
        ]
    )
    assert combined.has_ai_digital_source_marker is True


def test_risk_score_ranks_ai_markers_above_missing_exif_above_clean():
    c2pa = metadata_risk_score(ImageMetadataReport(has_camera_exif=True, has_c2pa_marker=True))
    digital_source = metadata_risk_score(ImageMetadataReport(has_camera_exif=True, has_ai_digital_source_marker=True))
    missing = metadata_risk_score(ImageMetadataReport(has_camera_exif=False))
    clean = metadata_risk_score(ImageMetadataReport(has_camera_exif=True))
    assert c2pa > missing > clean
    assert digital_source > missing > clean
