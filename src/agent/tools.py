"""Mock RAG and web search tools with fault injection."""
from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from agent.state import Document


class ToolError(Exception):
    """Base class for tool failures."""


class TransientToolError(ToolError):
    """Worth retrying (timeouts, rate limits, flaky network)."""


class PermanentToolError(ToolError):
    """Retrying will not help (bad request, auth, bug)."""


class ToolTimeoutError(TransientToolError):
    pass


class SearchTool(Protocol):
    name: str

    def search(self, query: str, top_k: int) -> list[Document]: ...


@dataclass
class FaultInjector:
    """Simulates failures and latency so the orchestration paths can be exercised."""

    failure_rate: float = 0.0   # probability of a failure per call
    latency_s: float = 0.0
    fail_times: int = 0         # fail the first N calls, then succeed
    always_fail: bool = False
    permanent: bool = False     # raise PermanentToolError instead of TransientToolError
    seed: Optional[int] = None
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def before_call(self, tool_name: str) -> None:
        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        fail = (
            self.always_fail
            or self.calls <= self.fail_times
            or (self.failure_rate > 0 and self._rng.random() < self.failure_rate)
        )
        if fail:
            error_cls = PermanentToolError if self.permanent else TransientToolError
            raise error_cls(f"{tool_name}: injected failure (call {self.calls})")


STOPWORDS = frozenset(
    """a about an and any are as at be by can do does for from have how i in is it its me my of on
    or our should that the their them they this those to was we what when where which who why will
    with you your please tell also""".split()
)


def tokenize(text: str) -> set[str]:
    """Lowercased content words with a naive plural strip."""
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


INTERNAL_CORPUS: list[dict] = [
    {
        "id": "kb-001",
        "title": "Vacation Policy",
        "source": "kb://hr/vacation-policy",
        "text": "Full-time employees accrue 25 vacation days per year. The carry-over limit is 5 "
        "unused days into the next year, and requests longer than 10 consecutive days need "
        "manager approval at least 30 days in advance.",
    },
    {
        "id": "kb-002",
        "title": "Expense Reimbursement Policy",
        "source": "kb://finance/expense-policy",
        "text": "Expenses must be submitted within 30 days with a receipt. Meals are reimbursed up "
        "to 60 USD per day while travelling, and any single expense above 500 USD needs "
        "pre-approval from the budget owner.",
    },
    {
        "id": "kb-003",
        "title": "Deployment Runbook",
        "source": "kb://engineering/deployment-runbook",
        "text": "Production deployments go through the CI pipeline: merge to main, wait for the "
        "staging canary to stay green for 15 minutes, then promote. Rollback is a one-click "
        "revert in the deploy dashboard. Deployments are frozen on Fridays after 14:00.",
    },
    {
        "id": "kb-004",
        "title": "Incident Response Guide",
        "source": "kb://engineering/incident-response",
        "text": "For a production incident, page the on-call engineer, open an incident channel, "
        "and assign an incident commander. Severity 1 incidents require a status update every "
        "30 minutes and a postmortem within 5 business days.",
    },
    {
        "id": "kb-005",
        "title": "Onboarding Checklist",
        "source": "kb://hr/onboarding",
        "text": "New hires receive a laptop on day one, complete security training in the first "
        "week, and are paired with an onboarding buddy for their first 30 days.",
    },
    {
        "id": "kb-006",
        "title": "Password and Access Security Policy",
        "source": "kb://security/access-policy",
        "text": "All accounts require multi-factor authentication. Passwords must be at least 14 "
        "characters and stored in the company password manager. Access to production data is "
        "granted per request and reviewed quarterly.",
    },
    {
        "id": "kb-007",
        "title": "Remote Work Guidelines",
        "source": "kb://hr/remote-work",
        "text": "Employees may work remotely up to 3 days per week. Working from another country "
        "for more than 20 days per year requires approval from HR and the tax team.",
    },
    {
        "id": "kb-008",
        "title": "Data Retention Policy",
        "source": "kb://legal/data-retention",
        "text": "Customer data is retained for 24 months after contract end and then deleted. "
        "Application logs are kept for 90 days, audit logs for 7 years.",
    },
]


class MockRAGTool:
    """Keyword-overlap retrieval over a small in-memory corpus."""

    name = "rag"

    def __init__(
        self,
        corpus: Optional[list[dict]] = None,
        faults: Optional[FaultInjector] = None,
        min_score: float = 0.2,
    ) -> None:
        self.corpus = corpus if corpus is not None else INTERNAL_CORPUS
        self.faults = faults or FaultInjector()
        self.min_score = min_score

    def search(self, query: str, top_k: int = 3) -> list[Document]:
        self.faults.before_call(self.name)
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        results: list[Document] = []
        for doc in self.corpus:
            doc_tokens = tokenize(doc["title"] + " " + doc["text"])
            score = len(query_tokens & doc_tokens) / len(query_tokens)
            if score >= self.min_score:
                results.append(
                    Document(
                        id=doc["id"],
                        title=doc["title"],
                        snippet=doc["text"],
                        source=doc["source"],
                        origin="rag",
                        score=round(score, 3),
                    )
                )
        results.sort(key=lambda d: d["score"], reverse=True)
        return results[:top_k]


class MockWebSearchTool:
    """Returns deterministic canned results derived from the query."""

    name = "web"

    def __init__(self, faults: Optional[FaultInjector] = None, score: float = 0.7) -> None:
        self.faults = faults or FaultInjector()
        self.score = score

    def search(self, query: str, top_k: int = 3) -> list[Document]:
        self.faults.before_call(self.name)
        if not query.strip():
            raise PermanentToolError("web: empty query")
        digest = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        slug = "-".join(sorted(tokenize(query))[:6]) or "result"
        templates = [
            ("Overview", "Public sources give an overview of '{q}', covering the main facts and context."),
            ("Recent coverage", "Recent coverage of '{q}' highlights the latest developments and figures."),
        ]
        return [
            Document(
                id=f"web-{digest}-{i}",
                title=f"{label}: {query[:60]}",
                snippet=text.format(q=query),
                source=f"https://example.com/{slug}/{i}",
                origin="web",
                score=self.score,
            )
            for i, (label, text) in enumerate(templates[:top_k], start=1)
        ]
