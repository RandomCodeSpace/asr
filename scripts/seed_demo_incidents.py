"""Seed the local incident store with a handful of resolved priors so the
similarity-lookup path is demonstrable. Idempotent — keyed on a `demo:seed`
tag, skips if a seed with the same tag already exists.

Run from repo root:
    .venv/bin/python scripts/seed_demo_incidents.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.incident import IncidentStore


SEEDS = [
    {
        "tag": "demo:seed:redis-oom",
        "query": (
            "production: redis-prod-1 in OOMKill crashloop, memory pressure "
            "spiking; cluster degraded for past 20 minutes"
        ),
        "environment": "production",
        "summary": "production: redis OOMKill crashloop on redis-prod-1",
        "severity": "high",
        "category": "availability",
        "tags": ["env:production", "component:redis",
                 "symptom:oom_crashloop", "demo:seed:redis-oom"],
        "findings_triage": (
            "redis-prod-1 maxmemory hit at 14:02 UTC. No deploy in last 24h. "
            "Eviction policy was noeviction — under write pressure the box "
            "OOM-killed itself rather than evicting. Memory growth correlates "
            "with a new analytics job spamming SETs without TTLs."
        ),
        "findings_di": [
            {
                "hypothesis": "noeviction policy + unbounded SETs from new "
                              "analytics job exhausts maxmemory",
                "evidence": "INFO memory shows used_memory_peak == maxmemory; "
                            "MONITOR sample shows SET keys with no EXPIRE "
                            "from analytics-worker:v3.1.2",
                "next_probe": "verify TTL on analytics keys",
            },
        ],
        "resolution": {
            "fix": "Set maxmemory-policy=allkeys-lru on redis-prod-1; "
                   "patched analytics-worker to issue EXPIRE on every SET "
                   "(8h TTL).",
            "validation": "memory holds at ~70% of maxmemory; no OOMKill in "
                          "30 minutes; analytics dashboards green.",
        },
    },
    {
        "tag": "demo:seed:postgres-replica-lag",
        "query": (
            "staging: postgres replica lag spiked to 90s after the 03:00 "
            "deploy, writes still landing on primary"
        ),
        "environment": "staging",
        "summary": "staging: postgres replica lag 90s after 03:00 deploy",
        "severity": "medium",
        "category": "data",
        "tags": ["env:staging", "component:postgres",
                 "symptom:replica_lag", "demo:seed:postgres-replica-lag"],
        "findings_triage": (
            "Deploy at 02:58 UTC introduced a hot-row update pattern in the "
            "session_events table — replica wal_apply lag started climbing "
            "two minutes after the deploy."
        ),
        "findings_di": [
            {
                "hypothesis": "missing index on session_events(user_id, ts) "
                              "causes the replica to scan the table on every "
                              "WAL apply",
                "evidence": "EXPLAIN ANALYZE on the replica shows seq scan; "
                            "primary uses the index because of cached plans",
                "next_probe": "create the index, restart replica plan cache",
            },
        ],
        "resolution": {
            "fix": "Created index session_events_user_ts_idx on "
                   "session_events(user_id, ts) on both primary and replica.",
            "validation": "replica lag dropped from 90s → <1s within 4 "
                          "minutes; no further regression.",
        },
    },
    {
        "tag": "demo:seed:payment-service-leak",
        "query": (
            "production: payment-service pods OOMKilled every ~6h, RSS "
            "climbs from 800MB to 4GB on a sawtooth pattern"
        ),
        "environment": "production",
        "summary": "production: payment-service slow memory leak, sawtooth OOM",
        "severity": "high",
        "category": "availability",
        "tags": ["env:production", "component:payment-service",
                 "symptom:memory_leak", "demo:seed:payment-service-leak"],
        "findings_triage": (
            "Memory growth is linear in handled requests, not in time — "
            "points to an unbounded cache or a request-scoped resource not "
            "being released."
        ),
        "findings_di": [
            {
                "hypothesis": "in-process LRU cache for currency conversion "
                              "rates was upgraded to TTL=24h with no max size",
                "evidence": "heap dump shows 3.1GB held by ConversionRateCache",
                "next_probe": "cap cache at 5k entries",
            },
        ],
        "resolution": {
            "fix": "Capped ConversionRateCache at maxsize=5000 with LRU "
                   "eviction; reduced TTL to 1h.",
            "validation": "RSS holds steady at ~1.2GB across 24h; no "
                          "OOMKill events; p99 request latency unchanged.",
        },
    },
    {
        "tag": "demo:seed:auth-401-spike",
        "query": (
            "production: 401s on /auth/login climbing since 10:15 UTC, no "
            "deploys in last 24h"
        ),
        "environment": "production",
        "summary": "production: auth 401 spike on /auth/login from 10:15",
        "severity": "high",
        "category": "security",
        "tags": ["env:production", "component:auth-service",
                 "symptom:auth_failures", "demo:seed:auth-401-spike"],
        "findings_triage": (
            "No code deploy. Identity provider rotated its signing key at "
            "10:14 UTC per IDP audit log; auth-service was still verifying "
            "tokens against the old key cached at 10:00."
        ),
        "findings_di": [
            {
                "hypothesis": "stale JWKS cache in auth-service after IDP "
                              "key rotation",
                "evidence": "auth-service logs show signature_verification_"
                            "failed for tokens issued after 10:14",
                "next_probe": "force JWKS cache refresh",
            },
        ],
        "resolution": {
            "fix": "Forced JWKS cache invalidation; reduced cache TTL from "
                   "1h to 15m; subscribed auth-service to IDP key-rotation "
                   "webhook.",
            "validation": "401 rate dropped from 38% → 0.2% within 90 "
                          "seconds of cache flush.",
        },
    },
]


def seed(store: IncidentStore) -> int:
    existing_tags: set[str] = set()
    for inc in store.list_all():
        existing_tags.update(inc.tags)

    created = 0
    for s in SEEDS:
        if s["tag"] in existing_tags:
            continue
        inc = store.create(
            query=s["query"],
            environment=s["environment"],
            reporter_id="demo-seed",
            reporter_team="platform",
        )
        inc.status = "resolved"
        inc.summary = s["summary"]
        inc.severity = s["severity"]
        inc.category = s["category"]
        inc.tags = list(s["tags"])
        inc.findings.triage = s["findings_triage"]
        inc.findings.deep_investigator = s["findings_di"]
        inc.resolution = s["resolution"]
        store.save(inc)
        created += 1
        print(f"  + {inc.id}  {s['tag']}")
    return created


if __name__ == "__main__":
    store = IncidentStore(base_dir=Path(__file__).resolve().parents[1] / "incidents")
    n = seed(store)
    if n:
        print(f"seeded {n} demo incidents")
    else:
        print("nothing to seed — all demo INCs already present")
