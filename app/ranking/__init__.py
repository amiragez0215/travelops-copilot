"""
候选评分与排序领域模块。

这里保存确定性业务算法，不直接读写 TravelState。
Node 负责流程边界，CandidateRanker 负责可独立测试的评分逻辑。
"""

from app.ranking.candidate_ranker import (
    CandidateRankConfig,
    CandidateRanker,
)

__all__ = [
    "CandidateRankConfig",
    "CandidateRanker",
]
