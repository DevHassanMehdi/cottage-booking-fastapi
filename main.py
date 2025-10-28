from fastapi import FastAPI, Request, Body, Header
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from datetime import date, timedelta
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD
from typing import List, Optional
import uuid

# --- Namespaces -------------------------------------------------------
EX = Namespace("http://example.org/cottage#")
SSWAP = Namespace("http://sswap.info/2008/09/sswap#")
SERVICE_BASE = "http://127.0.0.1:8000"
SERVICE_URI = URIRef(f"{SERVICE_BASE}/sswap/service/cottage-booking")

# --- FastAPI setup ----------------------------------------------------
app = FastAPI(title="Cottage Booking Service (SSWAP-enabled)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load ontology + data --------------------------------------------
g = Graph()
g.parse("data/ontology.owl")
g.parse("data/cottages.ttl", format="turtle")

# --- Models -----------------------------------------------------------
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

# --- Utility functions ------------------------------------------------
def date_ranges_with_shift(start: date, days: int, shift: int):
    for delta in range(-shift, shift + 1):
        s = start + timedelta(days=delta)
        e = s + timedelta(days=days)
        yield s, e

def availability_spans_for_cottage(cottage: URIRef):
    spans = []
    for _, _, avail in g.triples((cottage, EX.hasAvailability, None)):
        s_lit, e_lit = g.value(avail, EX.startDate), g.value(avail, EX.endDate)
        if s_lit and e_lit:
            s, e = date.fromisoformat(str(s_lit)), date.fromisoformat(str(e_lit))
            spans.append((s, e))
    return spans

def city_name(city_uri: URIRef):
    lbl = g.value(city_uri, EX.cityName)
    return str(lbl or g.value(city_uri, RDFS.label) or "")

# --- Basic routes -----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ontology")
def get_ontology_info():
    return {"ontology_base": str(EX), "classes": ["Cottage", "City", "AvailabilityPeriod"]}

# --- Core search logic ------------------------------------------------
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
    def matches_city(u): return city_name(u).lower().strip() == input.city.lower().strip()
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
                        capacity=int(cap),
                        bedrooms=int(beds),
                        distance_to_lake_m=int(dLake),
                        nearest_city=city_name(city),
                        distance_to_city_m=int(dCity),
                        booking_start=start,
                        booking_end=end,
                    ))
                    break
    return suggestions

# =====================================================================
#  SSWAP  IMPLEMENTATION
# =====================================================================

@app.get("/sswap/rdg", response_class=PlainTextResponse)
def sswap_rdg():
    g_rdg = Graph()
    g_rdg.bind("sswap", SSWAP)
    g_rdg.bind("ex", EX)
    g_rdg.bind("xsd", XSD)

    svc = SERVICE_URI
    g_rdg.add((svc, RDF.type, SSWAP.Service))
    g_rdg.add((svc, RDFS.label, Literal("Cottage Booking SSWAP Service")))
    g_rdg.add((svc, SSWAP.provides, EX.BookingSuggestion))
    g_rdg.add((svc, SSWAP.requires, EX.BookingRequest))

    # input description
    req = EX.BookingRequest
    g_rdg.add((req, RDF.type, RDFS.Class))
    for prop, dtype in [
        (EX.required_places, XSD.integer),
        (EX.required_bedrooms, XSD.integer),
        (EX.max_distance_lake_m, XSD.integer),
        (EX.city, XSD.string),
        (EX.max_distance_city_m, XSD.integer),
        (EX.required_days, XSD.integer),
        (EX.start_date, XSD.date),
        (EX.max_shift_days, XSD.integer),
        (EX.booker_name, XSD.string),
    ]:
        g_rdg.add((prop, RDF.type, RDF.Property))
        g_rdg.add((prop, RDFS.domain, req))
        g_rdg.add((prop, RDFS.range, dtype))

    # output description
    out = EX.BookingSuggestion
    g_rdg.add((out, RDF.type, RDFS.Class))
    for prop, dtype in [
        (EX.booking_number, XSD.string),
        (EX.address, XSD.string),
        (EX.imageURL, XSD.anyURI),
        (EX.capacity, XSD.integer),
        (EX.bedrooms, XSD.integer),
        (EX.distanceToLakeMeters, XSD.integer),
        (EX.distanceToCityMeters, XSD.integer),
        (EX.nearestCityName, XSD.string),
        (EX.bookingStart, XSD.date),
        (EX.bookingEnd, XSD.date),
        (EX.booker_name, XSD.string),
    ]:
        g_rdg.add((prop, RDF.type, RDF.Property))
        g_rdg.add((prop, RDFS.domain, out))
        g_rdg.add((prop, RDFS.range, dtype))

    return g_rdg.serialize(format="turtle")

# --- RIG -> invoke -> RRG ---------------------------------------------
def _get_str(gx, s, p): v = gx.value(s, p); return str(v) if v else None
def _get_int(gx, s, p): v = gx.value(s, p); return int(v) if v else None
def _get_date(gx, s, p): v = gx.value(s, p); return v.toPython() if v else None

@app.post("/sswap/invoke", response_class=PlainTextResponse)
def sswap_invoke(payload: str = Body(..., media_type="text/turtle"),
                 content_type: Optional[str] = Header(default="text/turtle")):
    rig = Graph()
    try:
        fmt = "xml" if content_type and "rdf+xml" in content_type else "turtle"
        rig.parse(data=payload, format=fmt)
    except Exception as e:
        return PlainTextResponse(f"# RIG parse error: {e}", status_code=400)

    req_nodes = list(rig.subjects(RDF.type, EX.BookingRequest))
    if not req_nodes:
        return PlainTextResponse("# No ex:BookingRequest found in RIG", status_code=400)
    s = req_nodes[0]

    data = dict(
        booker_name=_get_str(rig, s, EX.booker_name) or "Guest",
        required_places=_get_int(rig, s, EX.required_places) or 1,
        required_bedrooms=_get_int(rig, s, EX.required_bedrooms) or 0,
        max_distance_lake_m=_get_int(rig, s, EX.max_distance_lake_m) or 999999,
        city=_get_str(rig, s, EX.city) or "",
        max_distance_city_m=_get_int(rig, s, EX.max_distance_city_m) or 999999,
        required_days=_get_int(rig, s, EX.required_days) or 1,
        start_date=_get_date(rig, s, EX.start_date) or date.today(),
        max_shift_days=_get_int(rig, s, EX.max_shift_days) or 0,
    )

    results = search(SearchInput(**data))  # reuse core search logic

    rrg = Graph()
    rrg.bind("ex", EX)
    rrg.bind("xsd", XSD)
    rrg.bind("sswap", SSWAP)
    resp = BNode()
    rrg.add((resp, RDF.type, EX.BookingResponse))

    for r in results:
        sug = BNode()
        rrg.add((sug, RDF.type, EX.BookingSuggestion))
        rrg.add((sug, EX.booking_number, Literal(r.booking_number)))
        rrg.add((sug, EX.address, Literal(r.address)))
        rrg.add((sug, EX.imageURL, Literal(r.image_url, datatype=XSD.anyURI)))
        rrg.add((sug, EX.capacity, Literal(r.capacity, datatype=XSD.integer)))
        rrg.add((sug, EX.bedrooms, Literal(r.bedrooms, datatype=XSD.integer)))
        rrg.add((sug, EX.distanceToLakeMeters, Literal(r.distance_to_lake_m, datatype=XSD.integer)))
        rrg.add((sug, EX.distanceToCityMeters, Literal(r.distance_to_city_m, datatype=XSD.integer)))
        rrg.add((sug, EX.nearestCityName, Literal(r.nearest_city)))
        rrg.add((sug, EX.bookingStart, Literal(r.booking_start, datatype=XSD.date)))
        rrg.add((sug, EX.bookingEnd, Literal(r.booking_end, datatype=XSD.date)))
        rrg.add((sug, EX.booker_name, Literal(r.booker_name)))
        rrg.add((resp, EX.hasSuggestion, sug))

    return rrg.serialize(format="turtle")

# --- Mediator page ----------------------------------------------------
@app.get("/mediator", response_class=HTMLResponse)
def mediator_page(request: Request):
    return templates.TemplateResponse("mediator.html", {"request": request})
