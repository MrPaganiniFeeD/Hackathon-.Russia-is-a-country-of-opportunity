from dataclasses import dataclass, field


def not_null(value):
    if value is None:
        raise ValueError("Field cannot be null")
    return value


@dataclass(frozen=True, slots=True)
class Violation:
    text: str = field(default_factory=lambda: not_null(""))
    fine: int = field(default_factory=lambda: not_null(0))
    duration: int = field(default_factory=lambda: not_null(0))
