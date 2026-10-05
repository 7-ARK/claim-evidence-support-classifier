# Data files

Each line in the jsonl files is one example with three fields.

claim is the job requirement. It is one sentence.
evidence is the snippet you judge. It is one or two sentences.
label is Supported, Partially Supported, or Not Evidenced.

train.jsonl, val.jsonl, and test.jsonl are one generated set split three ways. 70 percent of the rows are in train, 15 percent are in val, and 15 percent are in test. The split is stratified by label, so each file keeps the same label mix. The seed is 42.

scripts/prepare_data.py builds these files. The same seed writes the same rows. data/stats.json records counts, the pattern mix, and file hashes. data/preview.txt shows one row of each pattern.

Nothing here is copied from a resume product or from a private evaluation set. The text is synthetic. The generator assigns the label from the rules in docs/labels.md. There is no separate human rating pass.

Pattern names in stats.json:

supported_single means one skill, and the evidence shows that skill in production.
supported_years means the evidence meets or beats the years in the claim.
supported_two_part means the claim has two parts and the evidence covers both.
partial_years means the same skill appears, with fewer years than the claim.
partial_missing_part means one part of a two part claim is present and the other is not.
partial_weaker means the skill appears only in a course, a class project, or a short prototype.
not_near_miss means the evidence is about a related tool, not the claimed tool.
not_years_on_other_tech means the years match the claim but they belong to the related tool.
not_negated means the claimed tool is named and then rejected.
not_unrelated means the evidence is about other work and does not mention the tool.
