from typing import Protocol, runtime_checkable


@runtime_checkable
class TextProvider(Protocol):
    def generate_text(self, system: str, user: str) -> str:
        """Return freeform text (e.g. a social-engineering chat transcript)."""
        ...

    def generate_json(self, system: str, user: str, schema_hint: str) -> str:
        """Return a JSON string. schema_hint names which canned shape the mock
        provider should produce (e.g. "sanitizer", "vision", "supervisor");
        real providers fold it into the prompt as a "respond with JSON for X" instruction."""
        ...


@runtime_checkable
class ImageProvider(Protocol):
    def generate_image(self, prompt: str, reference_b64: str | None = None) -> str:
        """Return a base64-encoded PNG. If reference_b64 is given, condition on it (img2img)."""
        ...


@runtime_checkable
class VisionProvider(Protocol):
    def inspect_images(self, images_b64: list[str], prompt: str, schema_hint: str = "vision") -> str:
        """Return the vision model's raw text response (expected to be JSON) for a set of images."""
        ...
