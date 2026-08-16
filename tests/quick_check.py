import urllib.request, json, sys

payload = json.dumps({
    "text_query": "malaria ke mukhya lakshan kya hain",
    "language": "hi",
    "chunking_strategy": "metadata_aware",
    "top_k": 3,
    "enable_cache": True
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/query",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
r = urllib.request.urlopen(req)
resp = json.loads(r.read().decode())

lat = resp["latency"]
lines = [
    f"SUCCESS: {resp['success']}",
    f"TOTAL ms: {lat['total_pipeline_ms']}",
    f"RETRIEVAL ms: {lat['retrieval_ms']}",
    f"GENERATION ms: {lat['generation_ms']}",
    f"GUARDRAIL pre ms: {lat['guardrails_pre_ms']}",
    f"CACHE HIT: {lat['is_cache_hit']}",
    f"UNDER 200ms: {lat['under_target_latency']}",
    f"CITATIONS: {len(resp['citations'])}",
    f"GUARDRAIL PASSED: {resp['guardrail_decision']['passed']}",
]
sys.stdout.buffer.write("\n".join(lines).encode("utf-8"))
sys.stdout.buffer.write(b"\n")
