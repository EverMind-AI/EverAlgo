"""Raw input data contracts — pre-boundary inputs to EverCore.

Schema TBD — fields finalized in T1 review.
"""

from pydantic import BaseModel, Field


class RawData(BaseModel):
    """Generic raw structured payload (Jira / Email / Confluence / Agent trace ...).

    Stub — schema fields TBD (T1).
    """

    id: str = Field(default="", description="TBD (T1 review)")
    source_type: str = Field(default="", description="TBD (T1 review)")


class RawFile(BaseModel):
    """Multimodal raw file (image / audio / document / video / url).

    Stub — schema fields TBD (T1).
    """

    uri: str = Field(default="", description="File URI / path / URL")
    mime: str = Field(default="", description="MIME type, e.g. application/pdf")
