# Publishing this repo (and your GitHub profile)

Hiring managers spend about a minute on a profile. One public flagship beats many private
ones. This document is the checklist that goes with this project.

## This repository

1. Create a public GitHub repo (do not force-push after the first tag).
2. Scan history before you push: `gitleaks detect --source . --verbose`.
3. Push `main`, then tag `v0.1.0`.
4. Pin this repo on your profile.
5. Add a 20-second GIF of `make ui` answering a question from `eval/corpus` to the README
   (record with [asciinema](https://asciinema.org/) or QuickTime; keep it under 8 MB).

## Profile README

Create a repository named exactly after your username and put this in `README.md`:

```markdown
# Roman Shevchenko

AI / ML engineer. I build retrieval and agent systems that run locally.

**Stack:** Python, LangGraph, vLLM, Qdrant, RAGAS, Langfuse.

## Featured

- **[agentic-rag-local](https://github.com/<you>/agentic-rag-local)** — multimodal
  agentic RAG (hybrid search, reranker, CRAG + Self-RAG, RAGAS gate).
- <one more public project, one line, business problem first>
- <one more>

## Elsewhere

- LinkedIn: ...
- Email: ...
```

## Private repositories

- Keep anything under an NDA private. Never publish client or employer code.
- Before making a personal project public: `gitleaks detect --log-opts=--all`,
  `git filter-repo` if a secret ever landed in history, rotate the key, add a license
  and a README, archive anything you will not maintain.
- Settings → Profile → **Include private contributions on my profile** so the
  contribution graph is honest without exposing those repos.

## What "looks senior" on a repo

- Architecture diagram and a 3-command quickstart in the README
- `docs/adr/` with the decisions that were not obvious
- Green CI: lint, types, tests, eval regression
- Conventional commits, tagged releases, `CHANGELOG.md`, Apache-2.0
- Reproducible: `docker compose up` + `make ingest` + `make eval`
