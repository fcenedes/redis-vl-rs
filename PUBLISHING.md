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

6. Create the release tag locally and push it:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

   Pushing a `v*` tag runs the `Publish` workflow. That workflow verifies the
   workspace, publishes `redis-vl`, waits for crates.io indexing, publishes
   `rvl`, and creates the GitHub Release.

7. The `Release Binaries` workflow runs when the GitHub Release is published:

   - It builds `rvl` on Linux, macOS, and Windows.
   - It attaches the binaries to the GitHub Release.

## Automated Publishing

Publishing is tag-driven:

```bash
git tag v0.1.1
git push origin v0.1.1
```

Use the manual `Publish` workflow in GitHub Actions only for dry-runs:

- It verifies the workspace.
- It runs `cargo publish -p redis-vl --dry-run`.
- The `rvl` dry-run runs only on tag releases, after `redis-vl` has been
  published and indexed.

`release-plz.yml` currently creates release PRs. Actual crate publication is
driven by pushing version tags so normal PRs and pushes only run verification.

## GitHub Release Binaries

The `Publish` workflow creates the GitHub Release automatically after both
crates publish successfully. The `Release Binaries` workflow then runs from the
GitHub `release.published` event.

Use the `Release Binaries` workflow manually only when rebuilding binaries for
an existing release:

1. Open Actions -> Release Binaries.
2. Run the workflow from `main`.
3. Set `tag` to an existing release tag, for example `v0.1.1`.

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
