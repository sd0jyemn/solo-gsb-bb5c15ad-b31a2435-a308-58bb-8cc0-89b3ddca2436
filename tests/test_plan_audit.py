"""交接计划符合性审计接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models import (
    Event,
    EventType,
    HandoverPlanAuditRequest,
    HandoverPlanNode,
)
from app.plan_audit import HandoverPlanAuditError, audit_handover_plan

client = TestClient(app)

DAY = "2026-09-18"


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:00+00:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def node(custodian: str, start_h: int, start_m: int, end_h: int, end_m: int) -> HandoverPlanNode:
    return HandoverPlanNode(
        custodian=custodian, window_start=dt(start_h, start_m), window_end=dt(end_h, end_m)
    )


def make_request(events, plan) -> HandoverPlanAuditRequest:
    return HandoverPlanAuditRequest(events=list(events), plan=list(plan))


def plan_tuples(resp):
    """(status, event_index, timing, deviation_seconds) 逐节点元组。"""
    return [
        (r.status, r.event_index, r.timing, r.deviation_seconds) for r in resp.plan_results
    ]


# ---------------------------------------------------------------------------
# 基本匹配与早晚判定
# ---------------------------------------------------------------------------


def test_happy_path_matches_with_timing():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 11, 30, custodian="Carol"),
        ev(EventType.UNSEAL, 12),
        ev(EventType.RESEAL, 12, 20, seal_id="S2", custodian="Dave"),
    ]
    plan = [
        node("Bob", 10, 0, 10, 30),    # 10:10 落在窗口内 -> on_time
        node("Carol", 11, 0, 11, 20),  # 11:30 晚于窗口终点 10 分钟 -> late
        node("Dave", 12, 10, 12, 40),  # 12:20 落在窗口内 -> on_time
    ]
    resp = audit_handover_plan(make_request(events, plan))

    assert resp.validation.valid is True
    assert plan_tuples(resp) == [
        ("matched", 1, "on_time", 0.0),
        ("matched", 2, "late", 600.0),
        ("matched", 4, "on_time", 0.0),
    ]
    # Alice 的初始封装（事件 0）不在计划内 -> unexpected
    assert [(u.event_index, u.custodian) for u in resp.unexpected] == [(0, "Alice")]
    assert resp.unexpected[0].type is EventType.SEAL
    assert resp.summary.matched == 3
    assert resp.summary.missing == 0
    assert resp.summary.unexpected == 1
    assert resp.summary.on_time == 2
    assert resp.summary.late == 1
    assert resp.summary.early == 0
    assert resp.summary.total_deviation_seconds == 600.0


def test_early_match_and_window_boundaries_are_on_time():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 11, custodian="Carol"),
        ev(EventType.HANDOVER, 12, custodian="Dave"),
    ]
    plan = [
        node("Bob", 10, 30, 11, 0),    # 10:00 早于窗口起点 30 分钟 -> early
        node("Carol", 11, 0, 11, 30),  # 11:00 恰在窗口起点 -> on_time
        node("Dave", 11, 30, 12, 0),   # 12:00 恰在窗口终点 -> on_time
    ]
    resp = audit_handover_plan(make_request(events, plan))
    assert plan_tuples(resp) == [
        ("matched", 1, "early", 1800.0),
        ("matched", 2, "on_time", 0.0),
        ("matched", 3, "on_time", 0.0),
    ]
    assert resp.summary.early == 1
    assert resp.summary.on_time == 2
    assert resp.summary.total_deviation_seconds == 1800.0


def test_missing_and_unexpected_and_summary():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 11, custodian="Carol"),
    ]
    plan = [
        node("Bob", 10, 0, 10, 30),
        node("Eve", 12, 0, 12, 30),  # 无人到任 -> missing
    ]
    resp = audit_handover_plan(make_request(events, plan))
    assert plan_tuples(resp) == [
        ("matched", 1, "on_time", 0.0),
        ("missing", None, None, None),
    ]
    # Alice（seal）与 Carol（handover）均未列入计划
    assert [(u.event_index, u.custodian) for u in resp.unexpected] == [
        (0, "Alice"),
        (2, "Carol"),
    ]
    assert resp.summary.matched == 1
    assert resp.summary.missing == 1
    assert resp.summary.unexpected == 2


def test_missing_result_echoes_plan_window():
    resp = audit_handover_plan(
        make_request([], [node("Bob", 10, 0, 11, 0)])
    )
    result = resp.plan_results[0]
    assert result.plan_index == 0
    assert result.custodian == "Bob"
    assert result.window_start == dt(10)
    assert result.window_end == dt(11)
    assert result.status == "missing"
    assert result.event_index is None
    assert result.timestamp is None
    assert result.timing is None
    assert result.deviation_seconds is None


# ---------------------------------------------------------------------------
# 仅 status=ok 的 seal/reseal/handover 参与匹配
# ---------------------------------------------------------------------------


def test_violation_events_never_participate_in_matching():
    events = [
        ev(EventType.HANDOVER, 8, custodian="Bob"),   # 违规：未封装交接
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Carol"),
        ev(EventType.RESEAL, 11, seal_id="S1", custodian="Dan"),  # 违规：已封装
    ]
    plan = [node("Bob", 8, 0, 8, 30), node("Dan", 11, 0, 11, 30)]
    resp = audit_handover_plan(make_request(events, plan))

    # 违规事件不产生实际到任记录：Bob 与 Dan 的计划节点均为 missing
    assert plan_tuples(resp) == [
        ("missing", None, None, None),
        ("missing", None, None, None),
    ]
    # 违规事件也不会出现在 unexpected 中；合法记录 Alice/Carol 未列入计划
    assert [(u.event_index, u.custodian) for u in resp.unexpected] == [
        (1, "Alice"),
        (2, "Carol"),
    ]
    assert resp.validation.valid is False
    assert [v.index for v in resp.validation.violations] == [0, 3]


def test_unseal_is_not_an_arrival_record():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.UNSEAL, 10),
        ev(EventType.RESEAL, 11, seal_id="S2", custodian="Alice"),
    ]
    resp = audit_handover_plan(make_request(events, [node("Alice", 8, 30, 9, 30)]))
    assert plan_tuples(resp) == [("matched", 0, "on_time", 0.0)]
    # unseal 不产生记录；reseal（下标 2）是记录且未列入计划
    assert [(u.event_index, u.type) for u in resp.unexpected] == [(2, EventType.RESEAL)]


def test_chain_violations_still_return_results_with_validation():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.SEAL, 10, seal_id="S2", custodian="Bob"),  # 违规：已封装
        ev(EventType.HANDOVER, 11, custodian="Carol"),
    ]
    resp = audit_handover_plan(
        make_request(events, [node("Carol", 10, 30, 11, 30)])
    )
    assert resp.validation.valid is False
    assert plan_tuples(resp) == [("matched", 2, "on_time", 0.0)]
    # 违规的 seal（下标 1）不是实际到任记录，不出现在 unexpected
    assert [(u.event_index, u.custodian) for u in resp.unexpected] == [(0, "Alice")]


# ---------------------------------------------------------------------------
# 全局最优匹配（非逐项贪心）
# ---------------------------------------------------------------------------


def test_global_optimum_beats_first_fit_greedy():
    # 两个 Bob 记录；逐项贪心会把计划节点配给第一个 Bob（偏离 45 分钟），
    # 全局最优应选窗口内的第二个 Bob（偏离 0）
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, 0, custodian="Bob"),
        ev(EventType.HANDOVER, 10, 50, custodian="Bob"),
    ]
    resp = audit_handover_plan(make_request(events, [node("Bob", 10, 45, 11, 0)]))
    assert plan_tuples(resp) == [("matched", 2, "on_time", 0.0)]
    assert [u.event_index for u in resp.unexpected] == [0, 1]


def test_match_count_maximized_before_deviation():
    # 匹配两个（其中一个偏离 2 小时）优于只匹配一个（偏离 0）
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 14, custodian="Carol"),
    ]
    plan = [node("Bob", 9, 30, 10, 30), node("Carol", 11, 0, 12, 0)]
    resp = audit_handover_plan(make_request(events, plan))
    assert plan_tuples(resp) == [
        ("matched", 1, "on_time", 0.0),
        ("matched", 2, "late", 7200.0),
    ]
    assert resp.summary.matched == 2
    assert resp.summary.total_deviation_seconds == 7200.0


def test_order_must_not_cross():
    # 计划顺序 Bob->Carol，实际顺序 Carol->Bob：二者不可兼得（交叉），
    # 匹配数并列（各 1）且偏离并列（均 0）时取下标序列字典序最小者，
    # 即 Carol 的计划节点匹配事件 0，Bob 的计划节点 missing
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Carol"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
    ]
    plan = [node("Bob", 9, 30, 10, 30), node("Carol", 8, 30, 9, 30)]
    resp = audit_handover_plan(make_request(events, plan))
    assert plan_tuples(resp) == [
        ("missing", None, None, None),
        ("matched", 0, "on_time", 0.0),
    ]
    assert [u.event_index for u in resp.unexpected] == [1]
    assert resp.summary.matched == 1
    assert resp.summary.missing == 1
    assert resp.summary.unexpected == 1


def test_lexicographic_tie_break_prefers_smaller_event_index():
    # 两个 Bob 记录都在窗口内（偏离同为 0）：选事件下标更小者
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 11, custodian="Bob"),
    ]
    resp = audit_handover_plan(make_request(events, [node("Bob", 9, 30, 11, 30)]))
    assert plan_tuples(resp) == [("matched", 1, "on_time", 0.0)]
    assert [u.event_index for u in resp.unexpected] == [0, 2]


def test_deviation_beats_lexicographic_when_tied_on_count():
    # 下标更小但偏离 7200 秒 vs 下标更大但偏离 60 秒：偏离优先
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 11, 59, custodian="Bob"),
    ]
    resp = audit_handover_plan(make_request(events, [node("Bob", 12, 0, 13, 0)]))
    assert plan_tuples(resp) == [("matched", 2, "early", 60.0)]
    assert resp.summary.total_deviation_seconds == 60.0


def test_multiple_plan_nodes_same_custodian_matched_in_order():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
        ev(EventType.HANDOVER, 12, custodian="Bob"),
    ]
    plan = [node("Bob", 9, 30, 10, 30), node("Bob", 11, 30, 12, 30)]
    resp = audit_handover_plan(make_request(events, plan))
    assert plan_tuples(resp) == [
        ("matched", 1, "on_time", 0.0),
        ("matched", 2, "on_time", 0.0),
    ]


# ---------------------------------------------------------------------------
# 空输入
# ---------------------------------------------------------------------------


def test_empty_plan_marks_all_records_unexpected():
    events = [
        ev(EventType.SEAL, 9, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
    ]
    resp = audit_handover_plan(make_request(events, []))
    assert resp.plan_results == []
    assert [(u.event_index, u.custodian) for u in resp.unexpected] == [
        (0, "Alice"),
        (1, "Bob"),
    ]
    assert resp.summary.matched == 0
    assert resp.summary.missing == 0
    assert resp.summary.unexpected == 2
    assert resp.summary.total_deviation_seconds == 0.0


def test_empty_events_marks_all_plan_nodes_missing():
    resp = audit_handover_plan(
        make_request([], [node("Bob", 10, 0, 11, 0), node("Carol", 12, 0, 13, 0)])
    )
    assert plan_tuples(resp) == [
        ("missing", None, None, None),
        ("missing", None, None, None),
    ]
    assert resp.unexpected == []
    assert resp.summary.missing == 2
    assert resp.validation.valid is True


def test_empty_request():
    resp = audit_handover_plan(make_request([], []))
    assert resp.plan_results == []
    assert resp.unexpected == []
    assert resp.summary.matched == resp.summary.missing == resp.summary.unexpected == 0


# ---------------------------------------------------------------------------
# 422 语义错误
# ---------------------------------------------------------------------------


def test_window_start_after_end_raises_422():
    with pytest.raises(ValidationError):
        node("Bob", 12, 0, 11, 0)


def test_window_start_equal_to_end_is_allowed():
    plan_node = node("Bob", 12, 0, 12, 0)  # 零长度窗口：要求恰在该时刻到任
    events = [ev(EventType.SEAL, 12, seal_id="S1", custodian="Bob")]
    resp = audit_handover_plan(make_request(events, [plan_node]))
    assert plan_tuples(resp) == [("matched", 0, "on_time", 0.0)]


def test_mixed_timezone_window_raises_422():
    request = HandoverPlanAuditRequest.model_construct(
        events=[
            Event(
                type=EventType.SEAL,
                timestamp=datetime.fromisoformat(f"{DAY}T09:00:00+00:00"),
                seal_id="S1",
                custodian="Alice",
            )
        ],
        plan=[
            HandoverPlanNode.model_construct(
                custodian="Alice",
                window_start=datetime.fromisoformat(f"{DAY}T08:00:00"),  # naive
                window_end=datetime.fromisoformat(f"{DAY}T10:00:00"),
            )
        ],
    )
    with pytest.raises(HandoverPlanAuditError):
        audit_handover_plan(request)


# ---------------------------------------------------------------------------
# API 集成
# ---------------------------------------------------------------------------


def test_api_handover_plan_audit_full_response():
    payload = {
        "events": [
            {"type": "seal", "timestamp": f"{DAY}T09:00:00Z",
             "seal_id": "S001", "custodian": "Alice"},
            {"type": "handover", "timestamp": f"{DAY}T10:10:00Z", "custodian": "Bob"},
            {"type": "handover", "timestamp": f"{DAY}T11:30:00Z", "custodian": "Carol"},
            {"type": "unseal", "timestamp": f"{DAY}T12:00:00Z"},
            {"type": "reseal", "timestamp": f"{DAY}T12:20:00Z",
             "seal_id": "S002", "custodian": "Dave"},
        ],
        "plan": [
            {"custodian": "Bob",
             "window_start": f"{DAY}T10:00:00Z", "window_end": f"{DAY}T10:30:00Z"},
            {"custodian": "Carol",
             "window_start": f"{DAY}T11:00:00Z", "window_end": f"{DAY}T11:20:00Z"},
            {"custodian": "Dave",
             "window_start": f"{DAY}T12:10:00Z", "window_end": f"{DAY}T12:40:00Z"},
        ],
    }
    r = client.post("/api/v1/seal-chain/handover-plan-audit", json=payload)
    assert r.status_code == 200
    body = r.json()

    # validation 保留原单链校验结构
    assert body["validation"]["valid"] is True
    assert body["validation"]["final_custodian"] == "Dave"
    assert [e["status"] for e in body["validation"]["events"]] == ["ok"] * 5

    assert body["plan_results"] == [
        {"plan_index": 0, "custodian": "Bob",
         "window_start": f"{DAY}T10:00:00Z", "window_end": f"{DAY}T10:30:00Z",
         "status": "matched", "event_index": 1, "timestamp": f"{DAY}T10:10:00Z",
         "timing": "on_time", "deviation_seconds": 0.0},
        {"plan_index": 1, "custodian": "Carol",
         "window_start": f"{DAY}T11:00:00Z", "window_end": f"{DAY}T11:20:00Z",
         "status": "matched", "event_index": 2, "timestamp": f"{DAY}T11:30:00Z",
         "timing": "late", "deviation_seconds": 600.0},
        {"plan_index": 2, "custodian": "Dave",
         "window_start": f"{DAY}T12:10:00Z", "window_end": f"{DAY}T12:40:00Z",
         "status": "matched", "event_index": 4, "timestamp": f"{DAY}T12:20:00Z",
         "timing": "on_time", "deviation_seconds": 0.0},
    ]
    assert body["unexpected"] == [
        {"event_index": 0, "type": "seal", "custodian": "Alice",
         "timestamp": f"{DAY}T09:00:00Z"}
    ]
    assert body["summary"] == {
        "matched": 3, "missing": 0, "unexpected": 1,
        "early": 0, "on_time": 2, "late": 1,
        "total_deviation_seconds": 600.0,
    }


def test_api_chain_violations_still_return_200():
    payload = {
        "events": [
            {"type": "handover", "timestamp": f"{DAY}T08:00:00Z", "custodian": "Bob"},
            {"type": "seal", "timestamp": f"{DAY}T09:00:00Z",
             "seal_id": "S001", "custodian": "Alice"},
        ],
        "plan": [
            {"custodian": "Bob",
             "window_start": f"{DAY}T08:00:00Z", "window_end": f"{DAY}T08:30:00Z"},
        ],
    }
    r = client.post("/api/v1/seal-chain/handover-plan-audit", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["validation"]["valid"] is False
    # 违规的 handover 不参与匹配：Bob 的计划节点 missing，且不构成 unexpected
    assert body["plan_results"][0]["status"] == "missing"
    assert body["unexpected"] == [
        {"event_index": 1, "type": "seal", "custodian": "Alice",
         "timestamp": f"{DAY}T09:00:00Z"}
    ]


def test_api_window_start_after_end_returns_422():
    payload = {
        "events": [],
        "plan": [
            {"custodian": "Bob",
             "window_start": f"{DAY}T12:00:00Z", "window_end": f"{DAY}T11:00:00Z"},
        ],
    }
    r = client.post("/api/v1/seal-chain/handover-plan-audit", json=payload)
    assert r.status_code == 422


def test_api_empty_custodian_returns_422():
    payload = {
        "events": [],
        "plan": [
            {"custodian": "",
             "window_start": f"{DAY}T11:00:00Z", "window_end": f"{DAY}T12:00:00Z"},
        ],
    }
    r = client.post("/api/v1/seal-chain/handover-plan-audit", json=payload)
    assert r.status_code == 422


def test_api_mixed_timezone_returns_422():
    payload = {
        "events": [
            {"type": "seal", "timestamp": f"{DAY}T09:00:00Z",
             "seal_id": "S001", "custodian": "Alice"},
        ],
        "plan": [
            {"custodian": "Alice",
             "window_start": f"{DAY}T08:00:00", "window_end": f"{DAY}T10:00:00"},
        ],
    }
    r = client.post("/api/v1/seal-chain/handover-plan-audit", json=payload)
    assert r.status_code == 422
    assert "时区" in r.json()["detail"]
