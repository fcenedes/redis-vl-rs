# Publishing

This repository publishes two crates:

- `redis-vl`, the public Rust library
- `rvl`, the CLI crate

`rvl` depends on `redis-vl` by version, so publish `redis-vl` first and wait for
it to appear in the crates.io index before publishing `rvl`.

## Required GitHub Secrets

- `CARGO_REGISTRY_TOKEN`: crates.io API token used by `.github/workflows/publish.yml`

Create it in GitHub under Settings -> Secrets and variables -> Actions. The
token comes from crates.io Account Settings -> API Tokens.

GitHub Pages, release PRs, and GitHub Releases use the built-in `GITHUB_TOKEN`;
they do not need extra repository secrets.

## First Release

1. Verify the workspace locally:

   ```bash
   cargo fmt --all --check
   cargo check --workspace --all-features
   cargo test --workspace --all-features
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
   cargo audit
   cargo deny check
   cargo package -p redis-vl
   ```

2. Confirm the crate names are available on crates.io:

   ```bash
   cargo search redis-vl
   cargo search rvl
   ```

3. Publish the library:

   ```bash
   cargo publish -p redis-vl
   ```

4. Wait for `redis-vl` to appear in the crates.io index.

5. Publish the CLI:

   ```bash
   cargo publish -p rvl
   ```

6. Create the GitHub Release and binaries from CI:

   - Go to Actions -> Release Binaries -> Run workflow.
   - Enter a tag such as `v0.1.1`.
   - The workflow creates the tag/release if needed, builds `rvl` on Linux,
     macOS, and Windows, and attaches the binaries to the GitHub Release.

## Automated Publishing

Use the manual `Publish` workflow in GitHub Actions:

- Select `redis-vl` first.
- Leave `dry_run` enabled for a verification pass.
- Re-run with `dry_run` disabled to publish.
- After crates.io indexes `redis-vl`, repeat for `rvl`.

`release-plz.yml` currently creates release PRs. Actual crate publication stays
manual through `publish.yml` so the first release can respect the workspace
publish order.

## GitHub Release Binaries

Use the `Release Binaries` workflow to create or update a GitHub Release:

1. Open Actions -> Release Binaries.
2. Run the workflow from `main`.
3. Set `tag` to the version tag, for example `v0.1.1`.
4. Leave `prerelease` unchecked for a normal release.

The workflow checks out the tag, builds `cargo build -p rvl --release --locked`
on Linux, macOS, and Windows, uploads workflow artifacts, and attaches the
artifacts to the GitHub Release.

## Trusted Publishing

crates.io also supports Trusted Publishing through GitHub Actions OIDC. The
first release still needs a normal token/manual publish; after that, configure
trusted publishing on crates.io for this repository and workflow, then replace
`CARGO_REGISTRY_TOKEN` with `rust-lang/crates-io-auth-action@v1` in
`publish.yml`.

Trusted Publishing removes the long-lived GitHub secret, but it must be
configured per crate on crates.io before the workflow can exchange an OIDC token
for a short-lived publishing token.

## Current Release Blockers

- `cargo clippy --workspace --all-targets --all-features -- -D warnings` is not
  yet clean because of existing strict pedantic/cargo lint debt. CI runs strict
  clippy as a non-blocking debt tracker until that is resolved.
- `cargo audit` currently has no vulnerability findings. It reports one allowed
  unmaintained warning through optional `hf-local` dependencies (`paste` via
  `fastembed`/`tokenizers`).
- `cargo package -p rvl` succeeds only after `redis-vl` exists in the crates.io
  index.
