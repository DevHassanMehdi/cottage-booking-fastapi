import hashlib
import json
import re
import socket
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from datetime import date, datetime, timedelta
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD
from typing import Dict, List, Optional
import uuid

# --- Namespaces -------------------------------------------------------
EX = Namespace("http://example.org/cottage#")
RESOURCE = Namespace("http://127.0.0.1:8000/sswap/service/cottage-booking/")
SSWAP = Namespace("http://sswapmeet.sswap.info/sswap/")
SERVICE_BASE = "http://127.0.0.1:8000"
SERVICE_URI = URIRef(f"{SERVICE_BASE}/sswap/service/cottage-booking")

ALIGNMENTS_DIR = Path("alignments")
ALIGNMENTS_DIR.mkdir(exist_ok=True)
ALIGNMENT_THRESHOLD = 0.75
REMOTE_FILES_DIR = Path("group1-files")
FAKE_REMOTE_RDG = REMOTE_FILES_DIR / "1-description-rdg.ttl"
FAKE_REMOTE_RRG = REMOTE_FILES_DIR / "3-response-rrg.ttl"

LOCAL_INPUT_PROPS = [
    {"key": "booker_name", "label": "Booker Name"},
    {"key": "required_places", "label": "Required Places"},
    {"key": "required_bedrooms", "label": "Required Bedrooms"},
    {"key": "max_distance_lake_m", "label": "Max Distance to Lake (m)"},
    {"key": "city", "label": "City"},
    {"key": "max_distance_city_m", "label": "Max Distance to City (m)"},
    {"key": "required_days", "label": "Required Days"},
    {"key": "start_date", "label": "Start Date"},
    {"key": "max_shift_days", "label": "Max Shift Days"},
]

LOCAL_OUTPUT_PROPS = [
    {"key": "booker_name", "label": "Booker Name"},
    {"key": "booking_number", "label": "Booking Number"},
    {"key": "address", "label": "Address"},
    {"key": "image_url", "label": "Image URL"},
    {"key": "capacity", "label": "Capacity"},
    {"key": "bedrooms", "label": "Bedrooms"},
    {"key": "distance_to_lake_m", "label": "Distance to Lake (m)"},
    {"key": "nearest_city", "label": "Nearest City"},
    {"key": "distance_to_city_m", "label": "Distance to City (m)"},
    {"key": "booking_start", "label": "Booking Start"},
    {"key": "booking_end", "label": "Booking End"},
]

LOCAL_INPUT_TYPES = {
    "booker_name": "string",
    "required_places": "int",
    "required_bedrooms": "int",
    "max_distance_lake_m": "int",
    "city": "string",
    "max_distance_city_m": "int",
    "required_days": "int",
    "start_date": "date",
    "max_shift_days": "int",
}

LOCAL_OUTPUT_TYPES = {
    "booker_name": "string",
    "booking_number": "string",
    "address": "string",
    "image_url": "string",
    "capacity": "int",
    "bedrooms": "int",
    "distance_to_lake_m": "int",
    "nearest_city": "string",
    "distance_to_city_m": "int",
    "booking_start": "date",
    "booking_end": "date",
}

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


# --- Alignment utilities ---------------------------------------------
def _service_hash(service_url: str) -> str:
    return hashlib.sha1(service_url.encode("utf-8")).hexdigest()[:12]


def _alignment_path(service_url: str) -> Path:
    return ALIGNMENTS_DIR / f"{_service_hash(service_url)}.json"


def _load_alignment(service_url: str) -> Optional[Dict]:
    path = _alignment_path(service_url)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_alignment(service_url: str, rdg_url: Optional[str], mappings: List[Dict]) -> Dict:
    payload = {
        "service_url": service_url,
        "rdg_url": rdg_url,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "mappings": mappings,
    }
    path = _alignment_path(service_url)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def _normalize_term(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text.lower().strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_term(a), _normalize_term(b)).ratio()


def _fetch_remote_text(url: str, body: Optional[str] = None) -> str:
    headers = {"Accept": "text/turtle, text/plain;q=0.8, */*;q=0.5"}
    data = None
    method = "GET"
    if body is not None:
        data = body.encode("utf-8")
        headers["Content-Type"] = "text/turtle"
        method = "POST"
    req = URLRequest(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
        raise HTTPException(status_code=502, detail=f"Remote request failed: {exc}") from exc


def _uri_tail(uri: URIRef) -> str:
    txt = str(uri)
    if "#" in txt:
        return txt.split("#")[-1]
    return txt.rstrip("/").split("/")[-1]


def _extract_remote_terms(rdg_text: str) -> List[Dict]:
    result: List[Dict] = []
    gx = Graph()
    gx.parse(data=rdg_text, format="turtle")

    for graph_node in gx.subjects(RDF.type, SSWAP.Graph):
        for subject_node in gx.objects(graph_node, SSWAP.hasMapping):
            if (subject_node, RDF.type, SSWAP.Subject) not in gx:
                continue
            for predicate, obj in gx.predicate_objects(subject_node):
                if predicate == SSWAP.mapsTo:
                    for obj_node in gx.objects(subject_node, predicate):
                        result.extend(_collect_object_terms(gx, obj_node, "output"))
                elif isinstance(predicate, URIRef):
                    result.append({
                        "uri": str(predicate),
                        "name": _uri_tail(predicate),
                        "kind": "input",
                    })
    return result


def _collect_object_terms(gx: Graph, node, kind: str) -> List[Dict]:
    entries: List[Dict] = []
    for predicate, obj in gx.predicate_objects(node):
        if predicate in (RDF.type, SSWAP.mapsTo):
            continue
        if isinstance(predicate, URIRef):
            entries.append({
                "uri": str(predicate),
                "name": _uri_tail(predicate),
                "kind": kind,
            })
    return entries


def _alignment_candidates(kind: str) -> List[Dict]:
    return LOCAL_INPUT_PROPS if kind == "input" else LOCAL_OUTPUT_PROPS


def _build_alignment_preview(remote_terms: List[Dict], existing: Optional[Dict[str, str]] = None) -> List[Dict]:
    preview = []
    for term in remote_terms:
        candidates = _alignment_candidates(term["kind"])
        best = None
        best_score = -1.0
        for candidate in candidates:
            score = _similarity(term["name"], candidate["key"])
            if score > best_score:
                best = candidate
                best_score = score
        preview.append({
            "remote_uri": term["uri"],
            "remote_name": term["name"],
            "kind": term["kind"],
            "suggested_local": best["key"] if best else None,
            "suggested_label": best["label"] if best else None,
            "confidence": round(best_score, 3),
            "requires_confirmation": best_score < ALIGNMENT_THRESHOLD,
            "current_local": existing.get(term["uri"]) if existing else None,
        })
    return preview


def _coerce_literal(value, target_type: str):
    if value is None:
        return None
    if target_type == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if target_type == "date":
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    return str(value)


def _format_literal(value, target_type: str) -> str:
    if target_type == "int":
        return str(int(value))
    if target_type == "date":
        if isinstance(value, date):
            return f"\"{value.isoformat()}\"^^xsd:date"
        return f"\"{date.fromisoformat(str(value)).isoformat()}\"^^xsd:date"
    # default string
    escaped = str(value).replace('"', '\\"')
    return f"\"{escaped}\""


def _mapping_lookup(alignment: Dict, kind: str) -> Dict[str, Dict]:
    entries = {}
    for item in alignment.get("mappings", []):
        if item.get("kind") != kind:
            continue
        local_key = item.get("local")
        remote_uri = item.get("remote_uri")
        if local_key and remote_uri:
            entries[local_key] = item
    return entries


def _build_remote_rig(payload: Dict, alignment: Dict) -> str:
    inputs = _mapping_lookup(alignment, "input")
    lines = [
        "@prefix sswap: <http://sswapmeet.sswap.info/sswap/> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "[] a sswap:Graph ;",
        "   sswap:hasMapping [",
        "     a sswap:Subject ;",
    ]
    for local_key, item in inputs.items():
        remote_uri = item["remote_uri"]
        value = payload.get(local_key)
        if value is None:
            continue
        literal = _format_literal(value, LOCAL_INPUT_TYPES.get(local_key, "string"))
        lines.append(f"     <{remote_uri}> {literal} ;")
    lines.append("     sswap:mapsTo []")
    lines.append("   ] .")
    return "\n".join(lines)


def _parse_rrg_with_alignment(rrg_text: str, alignment: Dict) -> List[BookingSuggestion]:
    outputs = _mapping_lookup(alignment, "output")
    results: List[BookingSuggestion] = []
    gx = Graph()
    gx.parse(data=rrg_text, format="turtle")

    for obj in gx.subjects(RDF.type, SSWAP.Object):
        data = {}
        for local_key, item in outputs.items():
            remote_uri = URIRef(item["remote_uri"])
            value = gx.value(obj, remote_uri)
            if value is None:
                continue
            literal_value = value.toPython() if hasattr(value, "toPython") else str(value)
            converted = _coerce_literal(literal_value, LOCAL_OUTPUT_TYPES.get(local_key, "string"))
            if converted is None:
                continue
            data[local_key] = converted
        if data:
            try:
                results.append(BookingSuggestion(**data))
            except Exception:
                continue
    return results

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


# --- Mediator alignment + invocation ---------------------------------
@app.post("/mediator/alignment/preview")
async def mediator_alignment_preview(request: Request):
    payload = await request.json()
    service_url = (payload.get("service_url") or "").strip()
    rdg_url = (payload.get("rdg_url") or "").strip()
    if not service_url or not rdg_url:
        raise HTTPException(status_code=400, detail="service_url and rdg_url are required.")

    rdg_text = _fetch_remote_text(rdg_url)
    remote_terms = _extract_remote_terms(rdg_text)
    existing = _load_alignment(service_url)
    existing_map = {}
    if existing:
        for item in existing.get("mappings", []):
            remote_uri = item.get("remote_uri")
            if remote_uri:
                existing_map[remote_uri] = item.get("local")
    preview = _build_alignment_preview(remote_terms, existing_map)
    return {
        "service_url": service_url,
        "rdg_url": rdg_url,
        "threshold": ALIGNMENT_THRESHOLD,
        "preview": preview,
        "has_alignment": existing is not None,
        "saved_at": existing.get("saved_at") if existing else None,
    }


@app.post("/mediator/alignment/save")
async def mediator_alignment_save(request: Request):
    payload = await request.json()
    service_url = (payload.get("service_url") or "").strip()
    rdg_url = (payload.get("rdg_url") or "").strip()
    mappings = payload.get("mappings") or []
    if not service_url or not rdg_url:
        raise HTTPException(status_code=400, detail="service_url and rdg_url are required.")

    normalized = []
    for item in mappings:
        remote_uri = item.get("remote_uri")
        local_key = (item.get("local") or "").strip() or None
        kind = item.get("kind")
        if not remote_uri or kind not in {"input", "output"} or not local_key:
            continue
        normalized.append({
            "remote_uri": remote_uri,
            "remote_name": item.get("remote_name"),
            "local": local_key,
            "kind": kind,
        })

    if not normalized:
        raise HTTPException(status_code=400, detail="No valid mappings were provided.")

    saved = _save_alignment(service_url, rdg_url, normalized)
    return {"status": "ok", "saved_at": saved["saved_at"]}


@app.post("/mediator/invoke")
async def mediator_invoke(request: Request):
    payload = await request.json()
    service_url = (payload.get("service_url") or "").strip()
    if not service_url:
        raise HTTPException(status_code=400, detail="service_url is required.")

    alignment = _load_alignment(service_url)
    if not alignment:
        raise HTTPException(
            status_code=400,
            detail="No alignment saved for this service. Run the alignment workflow first.",
        )

    search_payload = {key: payload.get(key) for key in LOCAL_INPUT_TYPES.keys()}
    try:
        search_input = SearchInput(**search_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    rig = _build_remote_rig(search_input.model_dump(), alignment)
    rrg_text = _fetch_remote_text(service_url, body=rig)
    suggestions = _parse_rrg_with_alignment(rrg_text, alignment)
    return {
        "suggestions": [s.model_dump() for s in suggestions],
        "raw_rrg": rrg_text,
        "rig": rig,
        "alignment_saved_at": alignment.get("saved_at"),
    }


# --- Mediator page ----------------------------------------------------
@app.get("/mediator", response_class=HTMLResponse)
def mediator_page(request: Request):
    ctx = {
        "request": request,
        "local_inputs": LOCAL_INPUT_PROPS,
        "local_outputs": LOCAL_OUTPUT_PROPS,
    }
    return templates.TemplateResponse("mediator.html", ctx)


# --- Fake remote service for Task 8 ----------------------------------
def _read_fake_file(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing file: {path}")
    return path.read_text()


@app.get("/fake-remote/rdg", response_class=PlainTextResponse)
def fake_remote_rdg():
    """Serve a static RDG that imitates an external cottage booking service."""
    return PlainTextResponse(_read_fake_file(FAKE_REMOTE_RDG), media_type="text/turtle")


@app.post("/fake-remote/invoke", response_class=PlainTextResponse)
async def fake_remote_invoke(request: Request):
    """
    Minimal fake cottage booking service that ignores the RIG payload and
    returns a canned RRG from another ontology for alignment testing.
    """
    _ = await request.body()  # consume body for completeness
    return PlainTextResponse(_read_fake_file(FAKE_REMOTE_RRG), media_type="text/turtle")
