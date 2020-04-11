
check:
	cargo test
	cargo outdated -R
	cargo audit
