.PHONY: fmt check clippy test run

fmt:
	cargo fmt --all

check:
	cargo fmt --all -- --check
	cargo check --all-targets --all-features

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all-targets --all-features

run:
	cargo run --bin mpipe -- ask
