.PHONY: test firewall train play eval eval-live dashboard research

firewall:
	pytest tests/firewall/ -v

test:
	pytest tests/ -v

research:
	PYTHONPATH=. python3 research/mechanics_extractor.py
	PYTHONPATH=. python3 research/ui_palette_extractor.py

train:
	PYTHONPATH=. python3 cli.py train --updates-per-stage 6 --rollout-steps 48 --num-envs 4

play:
	PYTHONPATH=. python3 cli.py play --max-steps 100

eval:
	PYTHONPATH=. python3 cli.py eval --mode arena --matches 3

eval-live:
	PYTHONPATH=. python3 cli.py eval --mode live --matches 2 --max-steps 45

dashboard:
	PYTHONPATH=. python3 cli.py dashboard --port 8500
