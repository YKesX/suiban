"""Deep research: the async job engine behind /v1/jobs (api.md §3).

Research is never a chat mode — it is a long-running job with COARSE progress only
(stage + percent). Queries, URLs, drafts and sub-agent chatter are internal working
state and never appear in any API response (product rule, docs/research.md).
"""
