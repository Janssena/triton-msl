# TriFast test-kernel provenance

Some attention test kernels derive from [TriFast](https://github.com/latkins/trifast),
Copyright (c) 2025 Liam Atkinson, distributed under the MIT License.
The full upstream notice is preserved in [third_party/trifast/LICENSE](../third_party/trifast/LICENSE).
The surrounding test harnesses, independent references and adversarial mutations
are triton-msl additions; this attribution does not claim upstream authors wrote
or endorsed those additions.

## Verified reference

On 2026-09-08 we compared against immutable upstream revision
[`2d5251135c774d0585e9bc042d7209c6b7c470cf`](https://github.com/latkins/trifast/tree/2d5251135c774d0585e9bc042d7209c6b7c470cf).
The reference file is
[`src/trifast/triton.py`](https://github.com/latkins/trifast/blob/2d5251135c774d0585e9bc042d7209c6b7c470cf/src/trifast/triton.py).

| Local source | Origin and changes |
|---|---|
| `tests/test_fa_bwd_routing.py`: `_bwd_kv`, `_bwd_q`, `_bwd_b` | Argument lists and complete body ASTs match upstream after removing decorators only. The test harness does not use upstream's autotuner. |
| `tests/test_fa_biased_routing.py`: `_biased_fa`, `_biased_tri_fa` | Adaptations of upstream `_fwd`, with local address/grid spellings and test switches. Not verbatim copies. |
| `tests/test_fa_mask_coordinates.py`: `_biased_mask` | Adversarial mask/index variants of the local biased forward kernel. Other kernels in this file are separate tests. |
| `tests/test_fa_biased_value_paths.py`: `_biased_value` | Adversarial value-path variants of the local biased forward kernel. |
| `tests/test_fa_alibi.py`: `_alibi_fa` | Local maskless/additive-bias adaptation of the biased forward kernel. |
| `tests/test_bwd_gqa_refuse.py`: `_bwd_kv_gqa`, `_bwd_kv_2d` | Local grouped-query and two-dimensional launch variants of the copied backward `_bwd_kv`; not verbatim copies. |
| `tests/test_bwd_fa_offset_order_guard.py`: `_bwd_kv_v_offsets_swapped` | Local adversarial value-offset ordering variant of the copied backward `_bwd_kv`. |
| `tests/test_fa_rereview_guards.py`: `_bwd_kv_vshift` | Local adversarial V-address-shift variant of the copied backward `_bwd_kv`. |

Matching today's immutable reference does not establish the historical revision
from which the initial copy was made. The original copy revision was not recorded;
do not invent it or call adapted kernels verbatim. Tests that import these kernels
or generate altered private copies retain this provenance. Distributing such
copies separately requires carrying the upstream copyright and permission notice
with them, not just a link to this repository. A mere mention of TriFast in a
test for an independently written kernel does not make that kernel a copy.

Reference SHA256 values:

- Upstream `LICENSE`: `69f396ba45867d5dd7e79cb9a9016df9c6fc6f396b007dcd7a6a1a85418a4340`
- Upstream `src/trifast/triton.py`: `fb6e4385c1b345c39604416828cfb06007b286d12fba9a8da33d023e9e695743`

## Distribution

Tests remain repository-only, excluded from both wheel and source distribution.
The TriFast license is explicitly included in the source distribution and wheel
license metadata nevertheless, so a downstream source repackaging does not depend
on an implicit root-level filename glob. This does not install TriFast as a runtime
dependency or claim its full integration suite has run.

The project already uses PEP 639's SPDX license expression. Its build-system
minimum is aligned with the [packaging guide's supported setuptools version](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license-and-license-files),
and `project.license-files` explicitly names both the project's existing license
and TriFast's license. Root-license coverage is preserved.

Reporter credit is distinct from this upstream copyright attribution. The local
compiler changes and reporter diagnoses/patches need their own accurate changelog
and acknowledgement; this notice does not establish authorship of those changes.
