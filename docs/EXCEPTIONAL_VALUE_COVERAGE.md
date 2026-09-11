# Exceptional-value coverage — 0.3.0rc4 candidate

The authoritative scoped inventory is the [axis-to-test map](CAPABILITY_AND_REFUSAL_CENSUS.md).
It distinguishes exceptional GPU cases from finite controls, integer edge cases, compile-time
refusals and checker self-tests. Collection proves existence, not execution. The old aggregate
“300 exceptional-value contracts” combined categories and must not be read as 300 independently
qualified non-finite inputs.

Retained waves exercise NaN/Inf forward/backward roles, row locality, masked padding, narrow
rounding, KDA recurrence and signed-zero identities. Oracles follow source propagation/select
semantics; a mask can legitimately bypass a non-finite operand. Finite results are checked
separately from non-finite membership. NaN payloads are diagnostic, not a universal guarantee.

test_welford_nonfinite_inputs.py covers NaN/+Inf/−Inf layernorm input with exact all-NaN row
membership and bitwise clean siblings. Its Welford checks instead require non-finite mean and
m2 in {NaN, +Inf}, never negative/−Inf: **order-robust admissible bounds**, not an exact
non-finite recurrence oracle. Clean cases use float64 references; checker controls reject
deliberately wrong classifications. These distinctions supersede the broad propagation claim.

The biased-decode suite adds Q-NaN, V-Inf and all-masked bias rows with actual execution observed,
exact exceptional classification and unaffected siblings protected. Masked-off queries in the
small-query tests contain NaNs and output sentinels remain untouched.

Final rc4 credit comes from its own frozen-tree and installed-artifact results, not predecessor
totals. No bounded sweep proves all exceptional inputs or spellings safe; newly discovered
wrong results in admitted paths remain defects to repair.
