"""Shared paragraph-ID construction for preprocessing and live retrieval."""


def make_paragraph_id(
    paper_id: str,
    section_index: int,
    paragraph_index: int,
) -> str:
    return (
        f"{paper_id}_s{section_index:03d}_p{paragraph_index:03d}"
    )
