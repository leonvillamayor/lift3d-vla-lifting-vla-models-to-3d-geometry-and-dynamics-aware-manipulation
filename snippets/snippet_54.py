import httpx

ARXIV_IDS = {
    "lift3d_policy":   "2503.10837",   # aproxima — confirma en el PDF
    "hybridvla":       "2503.00000",   # placeholder si tu versión lo trae
    "pi05":            "2504.16054",
}

def fetch_bibtex(arxiv_id: str) -> str:
    r = httpx.get(f"https://arxiv.org/bibtex/{arxiv_id}", timeout=10)
    r.raise_for_status()
    return r.text

for tag, aid in ARXIV_IDS.items():
    print(f"% --- {tag} ---")
    print(fetch_bibtex(aid))