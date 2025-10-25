
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from datetime import date, timedelta
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS
import uuid
from typing import List

EX = Namespace("http://example.org/cottage#")

app = FastAPI(title="Cottage Booking Service")

# Static + templates (homepage at "/")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Same-origin UI -> CORS not necessary, but harmless
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RDF at startup
g = Graph()
g.parse("data/ontology.owl")
g.parse("data/cottages.ttl", format="turtle")

class SearchInput(BaseModel):
    booker_name: str
    required_places: int = Field(ge=1)
    required_bedrooms: int = Field(ge=0)
    max_distance_lake_m: int = Field(ge=0)
    city: str
    max_distance_city_m: int = Field(ge=0)
    required_days: int = Field(ge=1)
    start_date: date
    max_shift_days: int = Field(ge=0)

class BookingSuggestion(BaseModel):
    booker_name: str
    booking_number: str
    address: str
    image_url: str
    capacity: int
    bedrooms: int
    distance_to_lake_m: int
    nearest_city: str
    distance_to_city_m: int
    booking_start: date
    booking_end: date

def date_ranges_with_shift(start: date, days: int, shift: int):
    for delta in range(-shift, shift + 1):
        s = start + timedelta(days=delta)
        e = s + timedelta(days=days)
        yield s, e

def availability_spans_for_cottage(cottage: URIRef):
    spans = []
    for _, _, avail in g.triples((cottage, EX.hasAvailability, None)):
        start_lit = g.value(avail, EX.startDate)
        end_lit = g.value(avail, EX.endDate)
        if start_lit and end_lit:
            from datetime import date as _d
            s = _d.fromisoformat(str(start_lit))
            e = _d.fromisoformat(str(end_lit))
            spans.append((s, e))
    return spans

def city_name(city_uri: URIRef):
    lbl = g.value(city_uri, EX.cityName)
    if lbl:
        return str(lbl)
    rdfs_label = g.value(city_uri, RDFS.label)
    return str(rdfs_label) if rdfs_label else None

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_model=List[BookingSuggestion])
def search(input: SearchInput):
    q = f"""
    PREFIX ex: <http://example.org/cottage#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?c ?address ?img ?cap ?beds ?dLake ?city ?dCity
    WHERE {{
        ?c a ex:Cottage ;
           ex:capacity ?cap ;
           ex:bedrooms ?beds ;
           ex:distanceToLakeMeters ?dLake ;
           ex:nearestCity ?city ;
           ex:distanceToCityMeters ?dCity ;
           ex:address ?address ;
           ex:imageURL ?img .

        FILTER(xsd:integer(?cap) >= {input.required_places})
        FILTER(xsd:integer(?beds) >= {input.required_bedrooms})
        FILTER(xsd:integer(?dLake) <= {input.max_distance_lake_m})
        FILTER(xsd:integer(?dCity) <= {input.max_distance_city_m})
    }}
    """
    rows = list(g.query(q))

    def matches_city(u):
        cname = city_name(u)
        return (cname or "").lower().strip() == input.city.lower().strip()

    suggestions: List[BookingSuggestion] = []

    for c, address, img, cap, beds, dLake, city, dCity in rows:
        if not matches_city(city):
            continue

        spans = availability_spans_for_cottage(c)
        for start, end in date_ranges_with_shift(input.start_date, input.required_days, input.max_shift_days):
            for avail_start, avail_end in spans:
                if start >= avail_start and end <= avail_end:
                    booking_id = str(uuid.uuid4())[:8].upper()
                    suggestions.append(BookingSuggestion(
                        booker_name=input.booker_name,
                        booking_number=booking_id,
                        address=str(address),
                        image_url=str(img),
                        capacity=int(cap.toPython()),
                        bedrooms=int(beds.toPython()),
                        distance_to_lake_m=int(dLake.toPython()),
                        nearest_city=city_name(city) or str(city),
                        distance_to_city_m=int(dCity.toPython()),
                        booking_start=start,
                        booking_end=end
                    ))
                    break
    return suggestions

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ontology")
def get_ontology_info():
    return {"ontology_base": str(EX), "classes": ["Cottage", "City", "AvailabilityPeriod"]}
