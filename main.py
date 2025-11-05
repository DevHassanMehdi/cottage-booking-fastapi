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
RESOURCE = Namespace("http://127.0.0.1:8000/sswap/service/cottage-booking/")
SSWAP = Namespace("http://sswapmeet.sswap.info/sswap/")
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
    """
    Canonical SSWAP Resource Description Graph (RDG)
    for the Cottage Booking Semantic Web Service.
    """
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("owl", Namespace("http://www.w3.org/2002/07/owl#"))
    g.bind("sswap", SSWAP)  # canonical http://sswapmeet.sswap.info/sswap/
    g.bind("ex", EX)
    g.bind("resource", RESOURCE)

    svc = RESOURCE["cottageBookingService"]
    g.add((svc, RDF.type, SSWAP.Resource))
    g.add((svc, RDF.type, EX.CottageBookingService))
    g.add((svc, SSWAP.providedBy, RESOURCE["serviceProvider"]))
    g.add((svc, SSWAP.name, Literal("Cottage Booking Semantic Web Service")))
    g.add((svc, SSWAP.oneLineDescription, Literal(
        "A service that accepts booking criteria and returns available cottage booking suggestions."
    )))

    # ----- Graph structure -----
    graph = BNode()
    g.add((svc, SSWAP.operatesOn, graph))
    g.add((graph, RDF.type, SSWAP.Graph))

    # ----- Subject (input – BookingRequest) -----
    subj = BNode()
    g.add((graph, SSWAP.hasMapping, subj))
    g.add((subj, RDF.type, SSWAP.Subject))
    g.add((subj, RDF.type, EX.BookingRequest))

    # Input properties (9 total)
    inputs = [
        "booker_name", "required_places", "required_bedrooms",
        "max_distance_lake_m", "city", "max_distance_city_m",
        "required_days", "start_date", "max_shift_days"
    ]
    for p in inputs:
        g.add((subj, EX[p], Literal("")))

    # ----- Object (output – BookingSuggestion) -----
    obj = BNode()
    g.add((subj, SSWAP.mapsTo, obj))
    g.add((obj, RDF.type, SSWAP.Object))
    g.add((obj, RDF.type, EX.BookingSuggestion))

    # Output properties (11 total)
    outputs = [
        "booker_name",          # 1) same as input booker
        "booking_number",       # 2)
        "address",              # 3)
        "imageURL",             # 4)
        "capacity",             # 5)
        "bedrooms",             # 6)
        "distanceToLakeMeters", # 7)
        "nearestCityName",      # 8)
        "distanceToCityMeters", # 9)
        "bookingStart",         # 10)
        "bookingEnd"            # 11)
    ]
    for p in outputs:
        g.add((obj, EX[p], Literal("")))

    return g.serialize(format="turtle")


# --- RIG -> invoke -> RRG ---------------------------------------------
def _get_str(gx, s, p): v = gx.value(s, p); return str(v) if v else None
def _get_int(gx, s, p): v = gx.value(s, p); return int(v) if v else None
def _get_date(gx, s, p): v = gx.value(s, p); return v.toPython() if v else None

@app.post("/sswap/invoke", response_class=PlainTextResponse)
async def sswap_invoke(request: Request):
    """Canonical SSWAP-compliant invocation (RIG → RRG) without blank node IDs."""
    body = (await request.body()).decode("utf-8")

    g = Graph()
    try:
        g.parse(data=body, format="turtle")
    except Exception as e:
        return PlainTextResponse(f"# RIG parse error: {e}", status_code=400)

    # --- locate Subject node (the BookingRequest) ---
    subject_node = None
    graph_node = None
    for gr in g.subjects(RDF.type, SSWAP.Graph):
        graph_node = gr
        for sub in g.objects(gr, SSWAP.hasMapping):
            if (sub, RDF.type, SSWAP.Subject) in g and (sub, RDF.type, EX.BookingRequest) in g:
                subject_node = sub
                break
        if subject_node:
            break

    if not subject_node:
        return PlainTextResponse(
            "Invalid RIG: Missing sswap:Subject of type ex:BookingRequest.",
            status_code=400,
        )

    # --- extract booking criteria ---
    def get_val(pred, default=None, cast=str):
        val = g.value(subject_node, EX[pred])
        if val is None:
            return default
        v = str(val)
        if cast is int:
            try:
                return int(v)
            except Exception:
                return default
        return v

    data = dict(
        booker_name=get_val("booker_name", "Guest"),
        required_places=get_val("required_places", 1, int),
        required_bedrooms=get_val("required_bedrooms", 0, int),
        max_distance_lake_m=get_val("max_distance_lake_m", 999999, int),
        city=get_val("city", ""),
        max_distance_city_m=get_val("max_distance_city_m", 999999, int),
        required_days=get_val("required_days", 1, int),
        start_date=get_val("start_date", date.today().isoformat()),
        max_shift_days=get_val("max_shift_days", 0, int),
    )

    # --- perform search (reuse Task 6 logic) ---
    results = search(SearchInput(**data))

    # --- build canonical SSWAP response graph manually (human-readable) ---
    response = "@prefix ex: <http://example.org/cottage#> .\n"
    response += "@prefix sswap: <http://sswapmeet.sswap.info/sswap/> .\n"
    response += "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"

    response += "[] a sswap:Graph ;\n"
    response += "   sswap:hasMapping [\n"
    response += "      a ex:BookingRequest , sswap:Subject ;\n"

    # Input request
    for key, val in data.items():
        if isinstance(val, (int, float)):
            response += f"      ex:{key} {val} ;\n"
        else:
            response += f"      ex:{key} \"{val}\" ;\n"

    response += "      sswap:mapsTo \n"

    # Each booking suggestion
    suggestion_blocks = []
    for r in results:
        block = (
            "         [ a ex:BookingSuggestion , sswap:Object ;\n"
            f"           ex:booker_name \"{r.booker_name}\" ;\n"
            f"           ex:booking_number \"{r.booking_number}\" ;\n"
            f"           ex:address \"{r.address}\" ;\n"
            f"           ex:imageURL \"{r.image_url}\"^^xsd:anyURI ;\n"
            f"           ex:capacity {r.capacity} ;\n"
            f"           ex:bedrooms {r.bedrooms} ;\n"
            f"           ex:distanceToLakeMeters {r.distance_to_lake_m} ;\n"
            f"           ex:nearestCityName \"{r.nearest_city}\" ;\n"
            f"           ex:distanceToCityMeters {r.distance_to_city_m} ;\n"
            f"           ex:bookingStart \"{r.booking_start}\"^^xsd:date ;\n"
            f"           ex:bookingEnd \"{r.booking_end}\"^^xsd:date ]"
        )
        suggestion_blocks.append(block)

    response += " ,\n".join(suggestion_blocks) + " \n   ] .\n"

    return PlainTextResponse(response, media_type="text/turtle")


# --- Mediator page ----------------------------------------------------
@app.get("/mediator", response_class=HTMLResponse)
def mediator_page(request: Request):
    return templates.TemplateResponse("mediator.html", {"request": request})
