"""项目 Pydantic 数据合同。"""

from app.schemas.budget_optimize_schema import (
    BudgetAdjustmentPlan,
    BudgetCheckResult,
    BudgetCombinationOption,
    KnownCostBreakdown,
    RemainingBudgetAllocation,
    SelectionResult,
)
from app.schemas.cancel_schema import CancelResult
from app.schemas.candidate_rank_schema import (
    CandidateRankResult,
    RankedFlightCandidate,
    RankedHotelCandidate,
)
from app.schemas.commit_draft_schema import (
    CommitDraftResult,
    CommittedApprovalRecord,
    CommittedTripDraft,
    CommittedTripVersion,
)
from app.schemas.decision_schema import (
    DecisionActionOption,
    DecisionGateResult,
    DecisionHistoryItem,
    DecisionInterruptPayload,
    DecisionResponse,
)
from app.schemas.evidence_grade_schema import (
    EvidenceCoverage,
    EvidenceGradeResult,
    EvidenceIssue,
)
from app.schemas.planning_context_schema import BudgetPlan, PlanningContext
from app.schemas.preference_schema import PreferenceSignal
from app.schemas.proposal_schema import (
    ProposalDraft,
    ProposalGenerationResult,
    TripProposal,
)
from app.schemas.request_constraint_schema import (
    FieldResolution,
    HardConstraint,
    NamedConstraint,
    PartialDate,
    RequirementIssue,
    SpendPreference,
    UnmappedRequirement,
)
from app.schemas.revision_schema import (
    HardConstraintSelector,
    NamedConstraintSelector,
    PreferenceSelector,
    RevisionApplyResult,
    RevisionFieldOperation,
    RevisionPatch,
    RevisionPlan,
    SpendPreferenceSelector,
)
from app.schemas.rerank_schema import RerankedEvidence, RerankResult
from app.schemas.retrieval_plan_schema import RetrievalPlan, RetrievalTask
from app.schemas.retrieval_result_schema import HybridRetrievalResult, RetrievedChunk
from app.schemas.trip_request_schema import TripRequest, TripRequestDraft
from app.schemas.verifier_schema import (
    ProposalVerificationResult,
    VerificationCheckResult,
    VerificationIssue,
)


__all__ = [
    "DecisionActionOption",
    "DecisionInterruptPayload",
    "DecisionResponse",
    "DecisionGateResult",
    "DecisionHistoryItem",
    "PreferenceSignal",
    "HardConstraint",
    "SpendPreference",
    "NamedConstraint",
    "UnmappedRequirement",
    "RequirementIssue",
    "FieldResolution",
    "PartialDate",
    "RevisionFieldOperation",
    "PreferenceSelector",
    "SpendPreferenceSelector",
    "HardConstraintSelector",
    "NamedConstraintSelector",
    "RevisionPatch",
    "RevisionPlan",
    "RevisionApplyResult",
    "TripRequestDraft",
    "TripRequest",
    "BudgetPlan",
    "PlanningContext",
    "RankedFlightCandidate",
    "RankedHotelCandidate",
    "CandidateRankResult",
    "KnownCostBreakdown",
    "RemainingBudgetAllocation",
    "BudgetCombinationOption",
    "SelectionResult",
    "BudgetCheckResult",
    "BudgetAdjustmentPlan",
    "RetrievalTask",
    "RetrievalPlan",
    "RetrievedChunk",
    "HybridRetrievalResult",
    "RerankedEvidence",
    "RerankResult",
    "EvidenceIssue",
    "EvidenceCoverage",
    "EvidenceGradeResult",
    "ProposalDraft",
    "TripProposal",
    "ProposalGenerationResult",
    "VerificationIssue",
    "VerificationCheckResult",
    "ProposalVerificationResult",
    "CommittedTripDraft",
    "CommittedTripVersion",
    "CommittedApprovalRecord",
    "CommitDraftResult",
    "CancelResult",
]
