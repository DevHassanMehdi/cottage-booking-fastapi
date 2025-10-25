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
