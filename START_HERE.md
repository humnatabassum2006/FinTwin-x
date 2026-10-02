# Start here

The current setup, run, demo, Docker, GitHub and troubleshooting guide is now in
**[START.md](START.md)**.

Fast development path:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
make data-small && make pipeline && make train && make rag
make web-install
```

Then run `make api` and `make web` in separate terminals and open
http://localhost:3000.
