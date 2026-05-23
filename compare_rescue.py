"""Quick comparison of rescue results between two models."""
import json
import sys

def load_results(path):
    data = json.loads(open(path).read())
    results = {}
    for r in data.get("llm_task_results", []):
        proposal = (r.get("payload") or {}).get("proposal_payload", {})
        task_id = r["task_id"]
        key = task_id
        for p in data.get("llm_task_packets", []):
            if p["task_id"] == task_id:
                key = p["source_object_id"]
                break
        results[task_id] = {
            "key": key,
            "rescue": proposal.get("rescue", False),
            "type_hint": proposal.get("type_hint", proposal.get("entity_type", "")),
            "canonical_name": proposal.get("canonical_name", ""),
            "confidence": proposal.get("confidence", ""),
        }
    return results

gpt = load_results("logs/llm-tasks/manuscript-suppression-rescue-gptoss120b-results.json")
qwen = load_results("logs/llm-tasks/manuscript-suppression-rescue-qwen36-35b-results.json")

agree = 0
disagree = []
for task_id in gpt:
    g = gpt[task_id]
    q = qwen.get(task_id, {})
    if g["rescue"] == q.get("rescue"):
        agree += 1
    else:
        disagree.append((g, q))

print(f"Total: {len(gpt)}")
print(f"Agree on rescue verdict: {agree}")
print(f"Disagree: {len(disagree)}")
if disagree:
    print()
    print("DISAGREEMENTS:")
    for g, q in disagree:
        print(f"  {g['key']:<20s}  gpt={str(g['rescue']):<6}  qwen={str(q.get('rescue', '?')):<6}")
        print(f"    gpt:  type_hint={str(g['type_hint']):<14} canonical={g['canonical_name']}")
        print(f"    qwen: type_hint={str(q.get('type_hint','')):<14} canonical={q.get('canonical_name','')}")

print()
print("ALL RESCUED (either model):")
print(f"  {'entity':<20} {'gpt':<6} {'gpt_hint':<14} {'qwen':<6} {'qwen_hint':<14}")
print(f"  {'-'*20} {'-'*6} {'-'*14} {'-'*6} {'-'*14}")
for task_id in gpt:
    g = gpt[task_id]
    q = qwen.get(task_id, {})
    if g["rescue"] or q.get("rescue"):
        gh = str(g.get("type_hint", "") or "")
        qh = str(q.get("type_hint", "") or "")
        print(f"  {g['key']:<20} {str(g['rescue']):<6} {gh:<14} {str(q.get('rescue','')):<6} {qh:<14}")
