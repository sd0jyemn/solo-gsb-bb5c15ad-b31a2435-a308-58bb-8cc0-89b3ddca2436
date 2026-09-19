"""交接计划符合性审计。

处理流程：
1. 调用现有单链校验器（``validate_events``）得到原校验结果并原样保留；
2. 仅从其中 ``status=ok`` 的 seal / reseal / handover 事件提取实际到任记录
   （保管人 + 到任时间），保留全链事件下标；违规事件与 unseal 事件不参与匹配；
3. 在计划节点序列与实际到任记录序列之间做**全局最优匹配**（动态规划，
   非逐项贪心）：仅保管人相同可匹配，且匹配对在两个序列中顺序不交叉。
   优化目标按优先级为——
   a. 匹配数最大；
   b. 实际到任时间偏离对应窗口的总秒数最小（落在窗口内偏离为 0，
      早于起点 / 晚于终点按到窗口的距离计）；
   c. 仍并列时，匹配事件下标序列（按计划节点顺序）字典序最小；
4. 输出逐计划节点的 matched / missing、匹配事件下标与 early / on_time / late，
   未匹配的实际到任记录列为 unexpected，并汇总各状态数量。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from .models import (
    EventType,
    HandoverPlanAuditRequest,
    HandoverPlanAuditResponse,
    HandoverPlanNode,
    HandoverPlanNodeResult,
    HandoverPlanSummary,
    UnexpectedArrival,
)
from .validator import validate_events

#: 产生实际到任记录的事件类型（custodian 在该事件发生时到任）
_ARRIVAL_TYPES = (EventType.SEAL, EventType.RESEAL, EventType.HANDOVER)


class HandoverPlanAuditError(ValueError):
    """审计入参的语义错误（在路由层转换为 HTTP 422）。"""


@dataclass(frozen=True)
class _ArrivalRecord:
    """一条实际到任记录：来自 status=ok 的 seal / reseal / handover 事件。"""

    event_index: int  # 全链事件下标（从 0 开始）
    type: EventType
    custodian: str
    timestamp: datetime


@dataclass(frozen=True)
class _PairNode:
    """匹配对链表节点：计划节点下标 -> 实际到任记录下标（按计划节点顺序链接）。"""

    plan_pos: int
    record_pos: int
    next: Optional["_PairNode"]


@dataclass(frozen=True)
class _Candidate:
    """子问题的最优匹配方案。

    比较优先级：count 大者优；并列时 deviation 小者优；仍并列时匹配记录下标
    序列（与事件下标序列同序，记录下标即字典序比较依据）小者优。
    """

    count: int
    deviation: timedelta
    pairs: Optional[_PairNode]  # 按计划节点顺序的匹配对链表


def _window_deviation(
    window_start: datetime, window_end: datetime, timestamp: datetime
) -> timedelta:
    """实际到任时间偏离窗口 [window_start, window_end] 的时长（窗口内为 0）。"""
    if timestamp < window_start:
        return window_start - timestamp
    if timestamp > window_end:
        return timestamp - window_end
    return timedelta(0)


def _timing(window_start: datetime, window_end: datetime, timestamp: datetime) -> str:
    """实际到任时间相对窗口的早晚（端点落在边界上算 on_time）。"""
    if timestamp < window_start:
        return "early"
    if timestamp > window_end:
        return "late"
    return "on_time"


def _sequence_less(a: Optional[_PairNode], b: Optional[_PairNode]) -> bool:
    """两条匹配序列按记录下标字典序比较，a 严格小于 b 时返回 True。"""
    while a is not None and b is not None:
        if a.record_pos != b.record_pos:
            return a.record_pos < b.record_pos
        a = a.next
        b = b.next
    return a is None and b is not None  # 前缀序列字典序更小


def _better(a: _Candidate, b: _Candidate) -> _Candidate:
    """返回更优的匹配方案（比较优先级见 :class:`_Candidate`）。"""
    if a.count != b.count:
        return a if a.count > b.count else b
    if a.deviation != b.deviation:
        return a if a.deviation < b.deviation else b
    return b if _sequence_less(b.pairs, a.pairs) else a


def _optimal_matching(
    plans: List[HandoverPlanNode], records: List[_ArrivalRecord]
) -> Optional[_PairNode]:
    """全局最优匹配（动态规划，非逐项贪心），返回按计划节点顺序的匹配对链表。

    自底向上填表，仅保留相邻两行：``cur[j]`` 为子问题 ``(plans[i:], records[j:])``
    的最优方案。每个状态的转移为三者取最优（见 :func:`_better`）——
    计划节点记为 missing、实际记录记为 unexpected、保管人相同时将二者匹配。
    """
    m, n = len(plans), len(records)
    zero = _Candidate(0, timedelta(0), None)
    next_row = [zero] * (n + 1)  # plans[m:] 为空时的子问题解
    for i in range(m - 1, -1, -1):
        cur = [zero] * (n + 1)  # records[n:] 为空时无可匹配
        for j in range(n - 1, -1, -1):
            best = _better(next_row[j], cur[j + 1])  # missing 或 unexpected
            if plans[i].custodian == records[j].custodian:
                sub = next_row[j + 1]
                best = _better(
                    best,
                    _Candidate(
                        count=sub.count + 1,
                        deviation=sub.deviation
                        + _window_deviation(
                            plans[i].window_start,
                            plans[i].window_end,
                            records[j].timestamp,
                        ),
                        pairs=_PairNode(i, j, sub.pairs),
                    ),
                )
            cur[j] = best
        next_row = cur
    return next_row[0].pairs


def _ensure_comparable(timestamps: List[datetime]) -> None:
    """混用带时区与不带时区的时间无法比较，统一报 422。"""
    if not timestamps:
        return
    aware = timestamps[0].tzinfo is not None and timestamps[0].utcoffset() is not None
    for value in timestamps[1:]:
        value_aware = value.tzinfo is not None and value.utcoffset() is not None
        if value_aware != aware:
            raise HandoverPlanAuditError(
                "时间戳时区信息不一致（不能混用带时区与不带时区的时间）"
            )


def audit_handover_plan(request: HandoverPlanAuditRequest) -> HandoverPlanAuditResponse:
    """校验事件链，并将计划节点与实际到任记录做全局最优匹配。"""
    validation = validate_events(request.events)

    # 仅从 status=ok 的 seal/reseal/handover 提取实际到任记录（保留全链下标）；
    # 违规事件与 unseal 事件不参与匹配
    records: List[_ArrivalRecord] = []
    for event, result in zip(request.events, validation.events):
        if result.status == "ok" and event.type in _ARRIVAL_TYPES:
            records.append(
                _ArrivalRecord(
                    event_index=result.index,
                    type=event.type,
                    custodian=event.custodian,  # type: ignore[arg-type]
                    timestamp=event.timestamp,
                )
            )

    # 窗口与到任时间不得混用带时区与不带时区的时间
    timestamps = [record.timestamp for record in records]
    for node in request.plan:
        timestamps.extend([node.window_start, node.window_end])
    _ensure_comparable(timestamps)

    # 全局最优匹配：plan_pos -> record_pos
    plan_to_record: Dict[int, int] = {}
    matched_records: Set[int] = set()
    pair = _optimal_matching(request.plan, records)
    while pair is not None:
        plan_to_record[pair.plan_pos] = pair.record_pos
        matched_records.add(pair.record_pos)
        pair = pair.next

    plan_results: List[HandoverPlanNodeResult] = []
    early = on_time = late = 0
    total_deviation = timedelta(0)
    for plan_index, node in enumerate(request.plan):
        record_pos = plan_to_record.get(plan_index)
        if record_pos is None:
            plan_results.append(
                HandoverPlanNodeResult(
                    plan_index=plan_index,
                    custodian=node.custodian,
                    window_start=node.window_start,
                    window_end=node.window_end,
                    status="missing",
                )
            )
            continue
        record = records[record_pos]
        deviation = _window_deviation(
            node.window_start, node.window_end, record.timestamp
        )
        timing = _timing(node.window_start, node.window_end, record.timestamp)
        if timing == "early":
            early += 1
        elif timing == "late":
            late += 1
        else:
            on_time += 1
        total_deviation += deviation
        plan_results.append(
            HandoverPlanNodeResult(
                plan_index=plan_index,
                custodian=node.custodian,
                window_start=node.window_start,
                window_end=node.window_end,
                status="matched",
                event_index=record.event_index,
                timestamp=record.timestamp,
                timing=timing,
                deviation_seconds=deviation.total_seconds(),
            )
        )

    unexpected = [
        UnexpectedArrival(
            event_index=record.event_index,
            type=record.type,
            custodian=record.custodian,
            timestamp=record.timestamp,
        )
        for record_pos, record in enumerate(records)
        if record_pos not in matched_records
    ]

    matched = len(plan_to_record)
    return HandoverPlanAuditResponse(
        validation=validation,
        plan_results=plan_results,
        unexpected=unexpected,
        summary=HandoverPlanSummary(
            matched=matched,
            missing=len(request.plan) - matched,
            unexpected=len(unexpected),
            early=early,
            on_time=on_time,
            late=late,
            total_deviation_seconds=total_deviation.total_seconds(),
        ),
    )
