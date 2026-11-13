# Working on NF12

NF12 compresses BF16 weights for GPU inference. Decoding must recover the
original BF16 representations, including signed zero and NaN payloads.
Arithmetic must satisfy the numerical contracts in [docs/SPEC.md](docs/SPEC.md).

## Read for the current task

Read [README.md](README.md) for setup and supported hardware. Load only the
relevant guides below, plus any nested `AGENTS.md` covering files you change.

- [Kernels and C ABI](.agents/kernels.md) for `csrc/`, `include/`, or format work.
- [Python API](.agents/python-api.md) for `python/nf12/` or torch operators.
- [Validation and benchmarks](.agents/validation.md) for choosing checks or
  making numerical, memory, or performance claims.
- [Builds and releases](.agents/build-release.md) for CMake, packaging, JIT,
  dependency changes, or `.github/workflows/`.

## Working rules

- Trace the affected execution path before editing. Reuse existing utilities
  and conventions. Address demonstrated failures rather than speculative ones.
- Make the smallest complete change. Keep unrelated refactors and formatting
  out of the patch. New abstractions or dependencies need a concrete consumer.
- Preserve user changes. Keep experiments and downloaded weights outside the
  tracked source tree. Check free storage before downloading checkpoints.
- Use `uv` for Python environments, dependencies, builds, and test commands.
  Reuse suitable environments without modifying system Python or toolchains.
- Update affected callers when an interface changes. Preserve existing file
  formats and public ABI unless the task explicitly requires a coordinated
  change.
- Comments should explain constraints or non-obvious decisions. Keep task
  history and conversations out of code and user documentation.
- Keep documentation concise and factual. Do not use punchy slogans or
  promotional language.
- Update documentation where users need it to understand changed usage. Small
  features do not automatically need a README headline, new section, or feature
  announcement. Prefer a short update in the existing relevant section.
- Do not commit superfluous documentation or scatter unnecessary Markdown files
  through the repository. New documents need a clear purpose beyond recording
  the current task. Link to authoritative specs and configuration instead of
  duplicating details.
- Do not commit binary blobs, including compiled libraries, wheels, model
  weights, or profiling dumps. Keep build and experiment artifacts out of Git.
- Report the checks actually run and their results. Name hardware gaps and
  skipped cases. Do not present inferred correctness or speedups as measured.
- Do not write the PR description yourself. Ask the user to write it out after
  giving them sufficient information about the changes. If the user still asks you
  to write the description yourself, give them a warning but proceed with noting
  this override in the PR description.
- Do not reply to the reviewer's comment yourself. Same rule as above.

## Review and publication

Do not commit, push, tag, publish packages, or post GitHub comments unless asked.
When asked to commit, use a conventional commit subject without an extended
body. Preserve licensing and author attribution. Do not invent co-author credits.

Keep this file short. Put task-specific detail in the linked guides and update
it when the implementation changes.
