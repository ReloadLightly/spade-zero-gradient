.PHONY: test smoke gates pilot-gen help

help:
	@echo "make test    - acceptance tests (mock backend, no model calls)"
	@echo "make smoke   - machinery smoke gates R4/R5 (mock backend, no model calls)"
	@echo "make gates   - R1-R3 real-model probe: 10 S0 envs via claude CLI, gate report (STOP gate for Roland)"
	@echo "make pilot-gen - one designer generation + one env evaluation via claude CLI"

test:
	python -m pytest

smoke:
	python -m szg.cli smoke

gates:
	python -m szg.cli probe --strategy strategies/S0_baseline.md --n 10 \
		--backend claude-cli --designer-model sonnet --solver-model haiku \
		--G 3 --out runs/probe_s0

pilot-gen:
	python -m szg.cli gen --strategy strategies/S0_baseline.md --skill logical_deduction \
		--backend claude-cli --designer-model sonnet --out runs/pilot_gen
