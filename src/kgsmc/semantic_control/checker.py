from __future__ import annotations

from dataclasses import asdict, dataclass

from kgsmc.semantic_control.resources import SemanticResources


@dataclass(frozen=True)
class SemanticCheckResult:
    status: str
    domain_ok: bool | None
    range_ok: bool | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class SemanticChecker:
    def __init__(self, resources: SemanticResources):
        self.resources = resources

    @staticmethod
    def compatible(
        actual: frozenset[str],
        expected: frozenset[str],
    ) -> bool | None:
        if not actual or not expected:
            return None

        return bool(actual & expected)

    def check_triple(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> SemanticCheckResult:
        constraint = self.resources.constraint(relation)

        if constraint is None:
            return SemanticCheckResult(
                status="UNKNOWN",
                domain_ok=None,
                range_ok=None,
                reason="relation constraint not found",
            )

        head_types = self.resources.expanded_types(head)
        tail_types = self.resources.expanded_types(tail)

        domain_ok = self.compatible(
            head_types,
            constraint.domain,
        )
        range_ok = self.compatible(
            tail_types,
            constraint.range,
        )

        if domain_ok is False or range_ok is False:
            return SemanticCheckResult(
                status="INVALID",
                domain_ok=domain_ok,
                range_ok=range_ok,
                reason="domain or range mismatch",
            )

        if domain_ok is None or range_ok is None:
            return SemanticCheckResult(
                status="UNKNOWN",
                domain_ok=domain_ok,
                range_ok=range_ok,
                reason="missing type or constraint information",
            )

        return SemanticCheckResult(
            status="VALID",
            domain_ok=True,
            range_ok=True,
            reason="domain and range match",
        )