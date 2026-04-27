# Publishing

This repository uses a tag-driven release flow.

- Pull requests and normal pushes run verification only.
- Pushing a `v*` tag publishes crates, creates the GitHub Release, and builds
  release binaries.

The release source of truth is the workspace version in `Cargo.toml`. A release
tag must match that version exactly, for example `version = "0.1.1"` must be
released with tag `v0.1.1`.

## What Gets Published

This workspace publishes two crates:

- `redis-vl`, the public Rust library
- `rvl`, the CLI crate

`rvl` depends on `redis-vl` by version. The release workflow therefore publishes
`redis-vl` first, waits for crates.io indexing, then publishes `rvl`.

The `rvl` binary is also built for GitHub Releases on:

- Linux: `rvl-linux-x86_64`
- macOS: `rvl-macos-aarch64`
- Windows: `rvl-windows-x86_64.exe`

## Required Setup

Add this repository secret:

- `CARGO_REGISTRY_TOKEN`: crates.io API token used by `.github/workflows/publish.yml`

Create it in GitHub under Settings -> Secrets and variables -> Actions. The
token comes from crates.io Account Settings -> API Tokens.

GitHub Pages, release creation, and binary uploads use the built-in
`GITHUB_TOKEN`; no extra GitHub secret is required.

## Normal Development Flow

For pull requests and normal pushes to `main`, no publishing happens.

These workflows are verification-only:

- `CI`: formatting, checks, clippy, tests, docs, and library packaging
- `Security`: `cargo audit` and `cargo deny check`
- `Docs`: mdBook and rustdoc deployment
- `Benchmarks`: manual benchmark runs

`release-plz` is manual-only. GitHub blocks Actions-created pull requests unless
Settings -> Actions -> General -> "Allow GitHub Actions to create and approve
pull requests" is enabled, and this project does not rely on release PRs for
publishing.

## Local Preflight

Before tagging a release, run:

```bash
cargo fmt --all --check
cargo check --workspace --all-features
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo audit
cargo deny check
cargo package -p redis-vl
```

Optional but recommended:

```bash
cargo clippy --workspace --all-targets --all-features
mdbook build docs
```

`cargo clippy --workspace --all-targets --all-features -- -D warnings` is still
a known lint-debt follow-up. CI tracks it as non-blocking.

## Tag-Driven Release

1. Update the workspace version in `Cargo.toml`.

2. Update `crates/rvl/Cargo.toml` so its `redis-vl` dependency uses the same
   version.

3. Update `CHANGELOG.md` and any visible version references in docs.

4. Commit and push `main`.

5. Create and push the matching version tag:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

The `Publish` workflow runs on `v*` tags and does the release:

1. Verifies the tag matches the Cargo version.
2. Runs formatting, checks, tests, docs, audit, deny, and packaging.
3. Runs `cargo publish -p redis-vl --dry-run`.
4. Publishes `redis-vl`.
5. Waits for `redis-vl` to appear in the crates.io index.
6. Runs `cargo publish -p rvl --dry-run`.
7. Publishes `rvl`.
8. Creates the GitHub Release.

The workflow is idempotent for partial releases: if `redis-vl` or `rvl` at the
target version is already visible on crates.io, the matching publish step is
skipped. The crates.io indexing wait allows up to 30 minutes before failing.

When the GitHub Release is published, the `Release Binaries` workflow runs and
attaches the cross-platform `rvl` binaries.

## Dry Runs

The manual `Publish` workflow is for dry-runs only. It verifies the workspace
and runs `cargo publish -p redis-vl --dry-run`.

The `rvl` dry-run runs only during a tag release because `rvl` needs the matching
`redis-vl` version to exist in the crates.io index after the library publish.

## Rebuilding Release Binaries

If a GitHub Release already exists and you only need to rebuild/upload binaries:

1. Open Actions -> Release Binaries.
2. Run the workflow from `main`.
3. Set `tag` to an existing release tag, for example `v0.1.1`.

The workflow checks out that tag, builds `cargo build -p rvl --release --locked`
on Linux, macOS, and Windows, uploads workflow artifacts, and attaches them to
the GitHub Release.

## Failure Recovery

- If verification fails before publish, fix the issue, delete and recreate the
  tag locally, then force-update the remote tag only if no crates were published.
- If `redis-vl` publishes but `rvl` fails before publishing, first fix the
  workflow and rerun the release against the same tag. The publish job skips
  already-published crate versions and continues with the missing crate. If the
  crate package contents need to change, bump both crates to a new patch version
  instead.
- If both crates publish but the GitHub Release or binaries fail, rerun or
  manually run the `Release Binaries` workflow for the existing tag.
- crates.io versions are immutable. Never try to republish the same crate
  version after crates.io accepts it.

## Trusted Publishing

crates.io also supports Trusted Publishing through GitHub Actions OIDC. The
first release still needs a normal token/manual publish; after that, configure
trusted publishing on crates.io for this repository and workflow, then replace
`CARGO_REGISTRY_TOKEN` with `rust-lang/crates-io-auth-action@v1` in
`publish.yml`.

Trusted Publishing removes the long-lived GitHub secret, but it must be
configured per crate on crates.io before the workflow can exchange an OIDC token
for a short-lived publishing token.

## Current Release Notes

- `cargo audit` currently has no vulnerability findings. It reports one allowed
  unmaintained warning through optional `hf-local` dependencies (`paste` via
  `fastembed`/`tokenizers`).
- `cargo deny check` passes. Duplicate-version findings are warnings only.
- `cargo package -p rvl` succeeds only after the matching `redis-vl` version is
  available in the crates.io index.
