.PHONY: model setup build serve dev test reference reference-model check clean

model:
	python3 scripts/fetch_model.py

setup: model
	npm ci --prefix web

build:
	npm run build --prefix web
	cargo build --release --locked

serve:
	cargo run --release --locked -p tensor-trace-server -- serve

dev:
	npm run dev --prefix web

test:
	cargo test --workspace --locked

reference:
	OPENBLAS_NUM_THREADS=1 python3 validation/check.py

reference-model:
	OPENBLAS_NUM_THREADS=1 python3 validation/check.py --model models/stories15M.bin --tokenizer models/tokenizer.bin --output build/pretrained-numeric-validation.json

check:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	npm run format:check --prefix web

clean:
	cargo clean
	rm -rf web/dist
