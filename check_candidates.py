"""Show rescue candidate list from latest extraction."""
import json

data = json.loads(open('logs/manuscript-review/manuscript-corpus-report.json').read())
diags = data.get('llm_task_diagnostics', [])
packets = data.get('llm_task_packets', [])

from collections import Counter
reasons = Counter()
for d in diags:
    reasons[d['reason']] += 1
for reason, count in reasons.most_common():
    print(f'  {count:4d}  {reason}')

print(f'\ndiagnostic records selected: {sum(1 for d in diags if d["selected"])}')
print(f'deduplicated packets: {len(packets)}')
print()
for p in sorted(packets, key=lambda x: -x['payload']['occurrence_count']):
    pay = p['payload']
    docs = len(p['source_document_paths'])
    ev = len(p['evidence_payload'])
    print(
        f'  {pay["normalized_key"]:<24}'
        f' occ={pay["occurrence_count"]:<4}'
        f' scenes={pay["scene_count"]:<3}'
        f' docs={docs}'
        f' evidence={ev}'
        f' reasons={",".join(pay["suppression_reasons"])}'
    )
