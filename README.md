# Cottage Booking (Integrated FastAPI)
- Homepage served at `/` via Jinja2 templates
- Static assets under `/static`
- RDF ontology + dataset under `data/`
## Run
```bash
conda env create -f environment.yml
conda activate cottage_booking_env
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000/

## Task 8 – Run-time Ontology Alignment
The `/mediator` page now exposes a two-step workflow to connect to arbitrary SSWAP cottage-booking services:

1. **Alignment step** – enter the remote RDG + invoke URLs (defaults point to the bundled fake remote service), fetch the RDG, and confirm the proposed mappings. Every confirmed alignment is saved to `alignments/<hash>.json` with a timestamp so multiple remote services may coexist.
2. **Invocation step** – once an alignment exists, provide booking criteria and the mediator will build a RIG in the remote ontology, call the service, convert the resulting RRG through the stored mapping, and render normalized `BookingSuggestion` entries.

### Fake remote service
For demo purposes `/fake-remote/rdg` and `/fake-remote/invoke` serve the RDG/RRG provided by another team (copy placed in `group1-files/`). Use these endpoints when testing the alignment module without depending on other teams’ deployments.
