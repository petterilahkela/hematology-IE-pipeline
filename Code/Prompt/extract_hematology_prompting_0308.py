"""
Hematology NLP Extraction Pipeline
====================================
Extracts structured fields from Finnish hematology referrals and bone marrow reports.

Fields extracted  (in output order):
  From referral only:
    suspected_dg         - clinical suspicion / question asked in the referral (free Finnish text)
    prev_cur_dg          - confirmed hematological diagnosis before this sample
    prev_cur_dg_status   - remissio / relapsi / uusi diagnoosi / refraktaari
                           (or free text if it doesn't fit neatly)
    treatment            - recent / ongoing treatment for the haematological condition

  From referral + report combined:
    cur_dg               - current diagnosis or key morphological finding after the report
    cur_dg_status        - status based on report; standard category or descriptive finding

MODEL RECOMMENDATION (14 GB GPU)
----------------------------------
  Best:  qwen2.5:14b  (Ollama, Q4_K_M ≈ 8.7 GB VRAM)
         - Multilingual with strong Finnish; best medical reasoning in this size class.
         - Run: ollama pull qwen2.5:14b

  Challenger worth benchmarking:
    hf.co/LumiOpen/Llama-Poro-2-8B-Instruct  (≈ 5 GB Q4, ~2× faster)
    - Llama 3 base continued-pretrained on Finnish by LumiOpen, then instruction-tuned.
    - Better Finnish than vanilla Llama 3.1 8B; inherits Llama's reasoning & world knowledge.
    - Run via Ollama: ollama run hf.co/LumiOpen/Llama-Poro-2-8B-Instruct
    - Use --benchmark to compare it head-to-head against qwen2.5:14b on your ground truth.

  Too large for 14 GB:
    Poro 2 70B  (≥ 42 GB VRAM)
    Poro 1 34B  (≥ 20 GB VRAM, also no instruction tuning)

SETUP
-----
  pip install pandas openpyxl aiohttp anthropic tqdm

  For Ollama (recommended):
    ollama pull qwen2.5:14b
    ollama serve          # starts on localhost:11434

  For Anthropic:
    export ANTHROPIC_API_KEY=sk-...

USAGE
-----
  # Ollama (default)
  python extract_hematology.py --input data.xlsx --output results.xlsx

  # Anthropic
  python extract_hematology.py --input data.xlsx --output results.xlsx \\
      --backend anthropic --model claude-haiku-4-5-20251001

  # Tune for 50k rows (raise batch-size and concurrency cautiously)
  python extract_hematology.py --input data.xlsx --output results.xlsx \\
      --batch-size 5 --concurrency 3

BENCHMARK MODE
--------------
  Compare two models head-to-head on labelled ground-truth rows.
  The input file must already contain the corrected/gold extraction columns
  (suspected_dg, prev_cur_dg, …) — those are used as the reference.

  python extract_hematology.py --benchmark \\
      --input MGG_combined1_extracted.xlsx \\
      --model-a qwen2.5:14b \\
      --model-b hf.co/LumiOpen/Llama-Poro-2-8B-Instruct \\
      --benchmark-rows 20

  Outputs:
    • Per-field accuracy table for each model (exact + partial match)
    • Speed comparison (rows/second, estimated 50k runtime)
    • Side-by-side diff for every disagreement
    • benchmark_results.xlsx saved next to --input
"""

import pandas as pd
from typing import Optional
import json
import asyncio
import aiohttp
import argparse
import sys
import time
from tqdm import tqdm
import re

# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_REFERRAL = """Olet kokenut hematologi, joka analysoi tarkasti suomenkielisiä lähetteitä.
Poimi tiedot VAIN lähetteestä ja palauta AINA validi JSON-objekti – ei muuta tekstiä.
Palauta JSON näillä kentillä: suspected_dg, prev_cur_dg, prev_cur_dg_status, treatment

## Yleiset säännöt
- TÄRKEÄÄ: Palauta AINA pelkkä JSON-objekti. Ei selityksiä, ei johdantoa, ei muuta tekstiä. Vain JSON.
- Käytä aina lyhintä mahdollista ilmausta — pelkkä pääsana ilman kuvailevia lisämääreitä:
  "morfologinen remissio" → "remissio"
  "täydellinen remissio" → "remissio"
  "kantasolujensiirto CyTBI-esihoidon jälkeen" → "kantasolujensiirto"
- ÄLÄ lisää kysymysmerkkiä arvoihin
- ÄLÄ hallusinoi.
- ÄLÄ poimi termejä jotka ovat tutkimusmenetelmiä eivätkä hematologisia diagnooseja (esim. Bence-Jones, MRD, FISH, PCR, virtaussytometria, Munasarjasyöpä), mutta kysymysmerkillä merkitty taudin nimi on epäily ja kuuluu suspected_dg:hen (esim. "MM?" → suspected_dg = "myelooma")
- Lue lähete KOKONAAN — otsikko voi olla harhaanjohtava, myöhemmin mainittu vahvistettu diagnoosi ohittaa otsikon epäilyn
- Analysoi lähete huolellisesti ennen kuin täytät kentät


## Kenttäohjeet


### suspected_dg  (lähete → MAX 5 SANAA)
TÄRKEÄÄ, käy ohjeet läpi tässä järjestyksessä:
1. Skannaa KOKO lähete ensin — jos missä tahansa kohdassa 
   viitataan käsitteisiin: seurantanäyte, seurantatutkimus, kontrolli, 
   tilannearvio, vastearvio, hoitovasteen arvio, 
   määräaikaistutkimus, D+numero, Day+numero 
   → suspected_dg = "jäännöstauti" riippumatta muusta tekstistä
2. Jos kyseessä on tuleva kantasoluluovuttaja → "kantasoluluovuttaja"
3. Jos ei ole jäännöstauti ja on suora kysymys "onko X?", "viitteitä X:stä?", "Selvitellään X"→ suspected_dg = kysytty asia (esim. "MDS leukemisoituminen")
4. Jos ei ole jäännöstauti ja on epäily tai oire → taudin tai oireen nimi
5. Jos ei ole jäännöstauti ja on vahvistettu diagnoosi + lisätutkimukset, → diagnoosin nimi
6. Täytä kenttä vaikka olisit epävarma. Jos olet epävarma onko "jäännöstauti" vai diagnoosi, käytä "jäännöstauti"




### prev_cur_dg  (lähete → vahvistettu diagnoosi)
Skannaa koko lähete ja kysy: Onko lähetteessä mainittu varmistettu diagnoosi?
- Jos Kyllä → kirjoita diagnoosin nimi
- Jos tekstissä ei ole mitään diagnoosia tai olet täysin varma että se on pelkkä epäily → null
Jos suspected_dg on "jäännöstauti" niin prev_cur_dg on lähetteessä mainittu tauti, tällöin ei ikinä null.
Lyhyin mahdollinen vastaus, ei hoitoja tai muita tietoja, ja käytä taudeista lyhenteitä (esim. "B-ALL" eikä "akuutti B-soluinen lymfoblastileukemia")
Esimerkkejä: "KML", "AML", "B-ALL", "Ph+ ALL", "Ph pos ALL", "multippeli myelooma", "MDS", "follikulaarinen lymfooma", "MCL", "DLBCL", "KLL", "Blackfan-Diamond anemia"
Jos lähete on selvästi tuore diagnoosi (alkukokeet, täydentävät tutkimukset, juuri todettu) ja suspected_dg on diagnoosi → prev_cur_dg = sama arvo kuin suspected_dg

### prev_cur_dg_status  (lähete → taudin tila ennen näytettä)
Null jos prev_cur_dg on null.
Päättele tila lähetteen tekstistä:
- Tauti todettu ensimmäistä kertaa tässä hoitojaksossa → "uusi diagnoosi"
- Hoito toimii, tauti hallinnassa → "remissio"
- Tauti palannut remission jälkeen → "relapsi"
- Hoito ei tehoa → "refraktaari"
- Hyvä mutta ei täydellinen vaste → säilytä alkuperäinen ilmaus kuten "hyvä vaste"
- Epäselvä tai ei mainita → null

### treatment  (lähete → käynnissä oleva hematologinen hoito)
Kysy: Onko lähetteessä mainittu käynnissä oleva tai äskettäin annettu hematologinen hoito?
- Kyllä → kirjoita hoidon nimi
- Hoito on päättynyt → null
- Hoito ei ole vielä alkanut → null
- Hoito on muuhun sairauteen → null
Sisällytä ylläpitolääkitykset jos mainittu käynnissä olevana: imatinibi, bosutinibi, VelDex, Interfant

## Esimerkit

### Esimerkki 1 – Seurantanäyte, tauti remissiossa
LÄHETE: KML, imatinibilääkitys käynnissä hyvällä vasteella. P.k. seurantanäyte.
{"suspected_dg": null, "prev_cur_dg": "KML", "prev_cur_dg_status": "hyvä vaste", "treatment": "imatinibi"}

### Esimerkki 2 – Jäännöstauti-seuranta ennen siirtoa
LÄHETE: MDS/AML remissiossa. Toipunut konsolidaatiohoidosta. Edetään allogeeniseen siirtoon. Esikokeissa p.k. morfologia.
{"suspected_dg": "jäännöstauti ennen kantasolujen siirtoa", "prev_cur_dg": "MDS/AML", "prev_cur_dg_status": "remissio", "treatment": "konsolidaatiohoito"}

### Esimerkki 3 – Relapsi, siirto tehty äskettäin
LÄHETE: Relaboituneen lymfooman vuoksi allogeeninen kantasolujensiirto 6 vk sitten. Määräaikaistutkimus.
{"suspected_dg": "jäännöstauti", "prev_cur_dg": "follikulaarinen lymfooma", "prev_cur_dg_status": "relapsi", "treatment": "allogeeninen kantasolujensiirto"}
→ prev_cur_dg_status = tila ENNEN siirtoa. treatment = siirto, koska tehty äskettäin.

### Esimerkki 4 – Uusi diagnoosi selvittelyssä
LÄHETE: Jatkotutkimuksiin plasmasolutautiepäilynä, seerumin M-komponentti koholla.
{"suspected_dg": "plasmasolutauti", "prev_cur_dg": null, "prev_cur_dg_status": null, "treatment": null}

### Esimerkki 5 – Selvittely ilman diagnoosia
LÄHETE: Neutropenian vuoksi tutkimuksiin. P.k. luuydintutkimus.
{"suspected_dg": "neutropenia", "prev_cur_dg": null, "prev_cur_dg_status": null, "treatment": null}

### Esimerkki 6 – Harhaanjohtava otsikko
LÄHETE: Akuutin leukemian epäily / alkukokeet. Luuydinaspiraatista B-ALL diagnoosi jo saatu. Pk täydentävät kokeet.
{"suspected_dg": "B-ALL", "prev_cur_dg": "B-ALL", "prev_cur_dg_status": "uusi diagnoosi", "treatment": null}
→ Vahvistettu diagnoosi tekstissä ohittaa aina otsikon epäilyn.

### Esimerkki 7 – Jäännöstauti
LÄHETE: ALL, tehty allogeeninen kantasolujensiirto 6 vk sitten. P.k. seurantatutkimuksena luuytimen morfologia.
{"suspected_dg": "jäännöstauti", "prev_cur_dg": "ALL", "prev_cur_dg_status": null, "treatment": "allogeeninen kantasolujensiirto"}
→ mainitaan "seurantatutkimus" → jäännöstauti
"""

SYSTEM_PROMPT_REPORT = """Olet kokenut hematologi, joka analysoi tarkasti suomenkielisiä luuydinnäytteiden lausuntoja.
Poimi tiedot VAIN lausunnosta ja palauta AINA validi JSON-objekti – ei muuta tekstiä.
Palauta JSON näillä kentillä: cur_dg, cur_dg_status

## Yleiset säännöt
- TÄRKEÄÄ: Palauta AINA pelkkä JSON-objekti. Ei selityksiä, ei johdantoa, ei muuta tekstiä. Vain JSON.
- Käytä aina lyhintä mahdollista ilmausta — pelkkä pääsana ilman kuvailevia lisämääreitä:
  "morfologinen remissio" → "remissio"
  "täydellinen remissio" → "remissio"
- ÄLÄ lisää kysymysmerkkiä arvoihin — jos olet epävarma käytä null, poikkeuksena cur_dg joka tulee aina täyttää
- ÄLÄ hallusinoi
- Analysoi lausunto huolellisesti ennen kuin täytät kentät

## Kenttäohjeet

### cur_dg  (lausunto → diagnoosi tai löydös, MAX 8 SANAA)
TÄRKEÄÄ: "AL" ei ole hyväksytty arvo — se on liian epätarkka. Jos Diagnosis-rivi sanoo "AL" → ohita se ja siirry yhteenvetoon
Etsi tieto TÄSSÄ JÄRJESTYKSESSÄ:
1. Lausunnon "Diagnosis:"-rivi — ohita jos sisältää "Katso lausuntoa" / "kts lausunto" / "seurantanäyte" / tyhjä / pelkkä blastiprosentti / "Ei diagnostista" / alkaa "AL," → siirry yhteenvetoon
2. Lausunnon YHTEENVETO-osio — viimeinen kappale

Päättele löydöksen perusteella tässä järjestyksessä:
1. Näyte epäonnistunut, riittämätön tai luotettava arvio mahdoton → "teknisesti epäonnistunut näyte"
- Lausunto on aina tärkein tietolähde — konteksti on vain tukitieto
- Jos lausunto sanoo normaali tai remissio → cur_dg = "normaali" riippumatta siitä mitä prev_cur_dg on
- Jos suspected_dg = "jäännöstauti" JA lausunto normaali → cur_dg = "normaali"
2. Yhteenveto vahvistaa tunnetun diagnoosin aktiivisena → käytä diagnoosin nimeä
   Jos lausunto on normaali tai remissiossa → "normaali" vaikka prev_cur_dg olisi täytetty
3. Solukkuus on alentunut pääasiallisena löydöksenä → "hypoplasia"
4. Yhteenvedossa mainitaan dysplasia, kypsymishäiriö tai muu poikkeava löydös → lyhyt kuvaus MAX 8 sanaa, älä kirjoita vain "normaali"
- ÄLÄ kirjoita "remissio" cur_dg:hen — se kuuluu cur_dg_status kenttään
- Jos Diagnosis-rivi sanoo "Ei diagnostista" ja luuydin on normaali → "normaali"
5. prev_cur_dg = null JA luuydin puhdas → "normaali"



### cur_dg_status  (lausunto → tilan kuvaus, MAX 5 SANAA)
Täytä kun lausunto selkeästi ilmaisee taudin tilan — joko suoraan sanomalla "remissio", "relapsi" tms. tai kun blastimäärä tai muu selkeä löödös osoittaa tilan yksiselitteisesti. null jos tila on epäselvä tai ei mainita. Epäselvissä tapauksissa käytä aina null.
- Blastit alle 5%, "morfologinen remissio", "ei tautiviitteitä" → "remissio"
- Kontekstissa suspected_dg = "jäännöstauti" ja lausunto normaali → "remissio"
- On cur_dg tauti, joka ei ole prev_cur_dg → "uusi diagnoosi"
- Tauti palannut → "relapsi"
- Hoito ei tehoa, tauti aktiivinen → "refraktaari"
- Epäselvä löydös joka ei sovi yllä oleviin → lyhyt kuvaus MAX 5 sanaa
- Ei diagnoosia → null

## Esimerkit

### Esimerkki 1 – Lausunto normaali
LAUSUNTO: Morfologisia viitteitä KML:stä ei havaita. Normaali luuydin.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=jäännöstauti, prev_cur_dg=KML
{"cur_dg": "normaali", "cur_dg_status": "remissio"}

### Esimerkki 2 – Diagnoosi vahvistuu
LAUSUNTO: Plasmasolujen osuus 50-70% solukkuudesta. Sopii plasmasolutautiin.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=plasmasolutauti, prev_cur_dg=None
{"cur_dg": "plasmasolutauti", "cur_dg_status": "uusi diagnoosi"}

### Esimerkki 3 – Hypoplasia
LAUSUNTO: Solukkuus alentunut. Megakaryosyyttien määrä alentunut. Blasteja alle 5%.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=neutropenia, prev_cur_dg=None
{"cur_dg": "hypoplasia", "cur_dg_status": null}

### Esimerkki 4 – Teknisesti epäonnistunut
LAUSUNTO: Suuri osa soluista hajonnut. Plasmasolujen osuutta ei voi luotettavasti arvioida.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=jäännöstauti, prev_cur_dg=myelooma
{"cur_dg": "teknisesti epäonnistunut näyte", "cur_dg_status": null}

### Esimerkki 5 – Lyhyt löydöskuvaus
LAUSUNTO: Blasteja 64% solukkuudesta. Sopii akuuttiin B-soluiseen lymfoblastileukemiaan.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=B-ALL, prev_cur_dg=None
{"cur_dg": "B-ALL", "cur_dg_status": "uusi diagnoosi"}

### Esimerkki 6 — cur_dg lyhyesti
LAUSUNTO: Blasteja 20% solukkuudesta. Löydös sopii akuutin myelooisen leukemian relapsin morfologiseen kuvaan.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=jäännöstauti, prev_cur_dg=AML
{"cur_dg": "AML", "cur_dg_status": "relapsi"}
→ ÄLÄ kopioi koko lausunnon tekstiä — käytä diagnoosin lyhennettä ja laita tila cur_dg_status kenttään.

### Esimerkki 7 — AL Diagnosis-rivillä
LAUSUNTO: Diagnosis: AL, seurantanäyte: blasteja <5%;
Yhteenveto: Blasteja alle 5%, morfologinen remissio.
KONTEKSTI LÄHETTEESTÄ: suspected_dg=jäännöstauti, prev_cur_dg=AML
{"cur_dg": "normaali", "cur_dg_status": "remissio"}
→ "AL" Diagnosis-rivillä ohitetaan. Lausunto normaali joten cur_dg = "normaali"

"""

BATCH_TEMPLATE = """Seuraavaksi on {n} potilasriviä. Palauta JSON-TAULUKKO [{{...}}, ...] SAMASSA JÄRJESTYKSESSÄ, ilman muuta tekstiä.
{rows}

Palauta JSON-taulukko:"""


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT FIELDS AND NORMALISATION
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_FIELDS = [
    "suspected_dg",
    "prev_cur_dg",
    "prev_cur_dg_status",
    "treatment",
    "cur_dg",
    "cur_dg_status",
]

STANDARD_STATUSES = {"remissio", "relapsi", "uusi diagnoosi", "refraktaari"}

_NULL_STRINGS = {None, "", "null", "Null", "NULL", "nan", "none",
                 "None", "ei tietoa", "ei mainintaa", "tuntematon", "-", "N/A", "n/a"}


def _clean(val):
    if val in _NULL_STRINGS:
        return None
    if isinstance(val, float):  # NaN from json
        return None
    if isinstance(val, str):
        val = val.strip()
        if val in _NULL_STRINGS:
            return None
    return val or None


def normalise_result(raw: dict) -> dict:
    out = {f: _clean(raw.get(f)) for f in OUTPUT_FIELDS}

    # Cascading null: no diagnosis → no status
    if not out["prev_cur_dg"]:
        out["prev_cur_dg_status"] = None
    if not out["cur_dg"]:
        out["cur_dg_status"] = None

    return out


EMPTY = {f: None for f in OUTPUT_FIELDS}


# ──────────────────────────────────────────────────────────────────────────────
# BATCH HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_referral_msg(batch):
    return f"LÄHETE: {batch[0]['referral']}"

def _build_report_msg(batch, referral_results=None):
    msg = f"LAUSUNTO: {batch[0]['report']}\n"
    if referral_results:
        msg += (f"KONTEKSTI LÄHETTEESTÄ: suspected_dg={referral_results[0].get('suspected_dg')}, "
                f"prev_cur_dg={referral_results[0].get('prev_cur_dg')}")
    return msg

def _parse_batch_response(text: str, expected_n: int) -> list:
    """Extract JSON array from model output, tolerate common formatting issues."""
    text = text.strip()

     # Handle {"rows": [...]} wrapper format
    if '"rows"' in text:
        try:
            obj = json.loads(text[text.find('{'):text.rfind('}')+1])
            if 'rows' in obj and isinstance(obj['rows'], list):
                results = [normalise_result(r) for r in obj['rows'][:expected_n]]
                while len(results) < expected_n:
                    results.append(EMPTY.copy())
                return results
        except json.JSONDecodeError:
            pass


     # Delete JS type comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*?\n', '', text)
    # Try array first
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            arr = json.loads(text[start:end])
            if isinstance(arr, list):
                results = [normalise_result(r) for r in arr[:expected_n]]
                while len(results) < expected_n:
                    results.append(EMPTY.copy())
                return results
        except json.JSONDecodeError:
            pass
    
    # Fallback: collect all {...} objects
    objects = []
    depth, in_str, i = 0, False, 0
    while i < len(text):
        c = text[i]
        if c == '"' and (i == 0 or text[i-1] != '\\'):
            in_str = not in_str
        if not in_str:
            if c == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        objects.append(json.loads(text[obj_start:i+1]))
                    except json.JSONDecodeError:
                        pass
        i += 1

    if objects:
        results = [normalise_result(r) for r in objects[:expected_n]]
        while len(results) < expected_n:
            results.append(EMPTY.copy())
        return results

    raise ValueError(f"No parseable JSON found in: {text[:300]}")


# ──────────────────────────────────────────────────────────────────────────────
# OLLAMA BACKEND (async)
# ──────────────────────────────────────────────────────────────────────────────

async def _ollama_batch(session: aiohttp.ClientSession, model: str,
                         user_msg: str, base_url: str, system: str = None) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 768, "num_ctx": 4096},
    }
    async with session.post(
        f"{base_url}/api/chat", json=payload,
        timeout=aiohttp.ClientTimeout(total=240)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["message"]["content"]

async def _process_batch_ollama(sem: asyncio.Semaphore, session: aiohttp.ClientSession,
                                 model: str, batch: list, base_url: str,
                                 retries: int = 3) -> list:
    async with sem:
        # call 1, Only referral (first 4 fields)
        referral_msg = _build_referral_msg(batch)

        results1 = [EMPTY.copy() for _ in batch]
        results2 = [EMPTY.copy() for _ in batch]

        for attempt in range(retries):
            try:
                text1 = await _ollama_batch(session, model, referral_msg, base_url,
                                            system=SYSTEM_PROMPT_REFERRAL)
                results1 = _parse_batch_response(text1, len(batch))
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"\n[WARN] Referral batch failed: {e}", file=sys.stderr)
                await asyncio.sleep(2 ** attempt)

        # Build report_msg when results1 is ready
        report_msg = _build_report_msg(batch, results1)

        for attempt in range(retries):
            try:
                text2 = await _ollama_batch(session, model, report_msg, base_url,
                                            system=SYSTEM_PROMPT_REPORT)
                results2 = _parse_batch_response(text2, len(batch))
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"\n[WARN] Report batch failed: {e}", file=sys.stderr)
                await asyncio.sleep(2 ** attempt)

        # Combine results
        combined = []
        for r1, r2 in zip(results1, results2):
            merged = {**r1}
            for key, val in r2.items():
                if val is not None:
                    merged[key] = val
            combined.append(merged)
        return combined


async def run_ollama(df: pd.DataFrame, model: str, batch_size: int,
                      concurrency: int, base_url: str) -> list:
    sem  = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 2)

    batches = [
        (i, [
            {
                "referral": str(df.iloc[j].get("referral") or "Ei tietoa")[:4000],
                "report":   str(df.iloc[j].get("report")   or "Ei tietoa")[:4000],
            }
            for j in range(i, min(i + batch_size, len(df)))
        ])
        for i in range(0, len(df), batch_size)
    ]

    all_results = [None] * len(df)

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for start, batch in batches:
            task = asyncio.ensure_future(
                _process_batch_ollama(sem, session, model, batch, base_url)
            )
            tasks.append((start, len(batch), task))
        pbar = tqdm(total=len(df), desc="Extracting")
        for start, n, task in tasks:
            results = await task
            for k, res in enumerate(results):
                all_results[start + k] = res
            pbar.update(n)
        pbar.close()
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# ANTHROPIC BACKEND (sync, batched)
# ──────────────────────────────────────────────────────────────────────────────

def run_anthropic(df: pd.DataFrame, model: str, batch_size: int) -> list:
    try:
        import anthropic as _anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = _anthropic.Anthropic()
    all_results = [None] * len(df)

    for i in tqdm(range(0, len(df), batch_size), desc="Batches"):
        batch = [
            {
                "referral": str(df.iloc[j].get("referral") or "Ei tietoa")[:4000],
                "report":   str(df.iloc[j].get("report")   or "Ei tietoa")[:4000],
            }
            for j in range(i, min(i + batch_size, len(df)))
        ]
        user_msg = _build_batch_msg(batch)
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=600 * len(batch),
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                results = _parse_batch_response(resp.content[0].text, len(batch))
                for k, res in enumerate(results):
                    all_results[i + k] = res
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\n[WARN] Batch {i} failed: {e}", file=sys.stderr)
                    for k in range(len(batch)):
                        all_results[i + k] = EMPTY.copy()
                else:
                    time.sleep(2 ** attempt)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# EXCEL OUTPUT WITH FORMATTING
# ──────────────────────────────────────────────────────────────────────────────

def write_formatted_excel(out_df: pd.DataFrame, path: str) -> None:
    """Save with colour-coded status columns."""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    out_df.to_excel(path, index=False)
    wb = load_workbook(path)
    ws = wb.active

    HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
    NEW_COL_FILL = PatternFill("solid", fgColor="EBF5FB")
    STATUS_COLORS = {
        "remissio":               "D5F5E3",
        "relapsi":                 "FADBD8",
        "uusi diagnoosi":           "FEF9E7",
        "refraktaari": "F9EBEA",
    }
    header_font = Font(bold=True, color="FFFFFF")
    extracted_cols = set(OUTPUT_FIELDS)
    headers = [cell.value for cell in ws[1]]

    for cell in ws[1]:
        cell.fill  = HEADER_FILL
        cell.font  = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            col_name = headers[cell.column - 1]
            val      = str(cell.value or "").strip().lower()
            if col_name in extracted_cols:
                cell.fill = PatternFill("solid", fgColor=NEW_COL_FILL.fgColor)
            if col_name in ("prev_cur_dg_status", "cur_dg_status"):
                c = STATUS_COLORS.get(val)
                if c:
                    cell.fill = PatternFill("solid", fgColor=c)
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    for col_idx, header in enumerate(headers, 1):
        col_letter = get_column_letter(col_idx)
        if header in ("referral", "report"):
            ws.column_dimensions[col_letter].width = 50
        elif header in extracted_cols:
            ws.column_dimensions[col_letter].width = 16
        else:
            ws.column_dimensions[col_letter].width = 12

            # Set row height
    for i in range(1, ws.max_row + 1):
         ws.row_dimensions[i].height = 20

    ws.freeze_panes = "A2"
    wb.save(path)




# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK SCORING
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_STATUSES = {"remissio", "relapsi", "uusi diagnoosi", "refraktaari"}

# Fields where an exact category match is the primary goal
STATUS_FIELDS = {"prev_cur_dg_status", "cur_dg_status"}

# Canonical aliases: map variant spellings to a standard token for comparison
_ALIASES = {
    "manttelisolulymfooma": "mcl",
    "mantle cell lymphoma": "mcl",
    "follicular lymphoma": "follikulaarinen lymfooma",
    "FL ": "follikulaarinen lymfooma",
    "multippeli myelooma": "myelooma",
    "multiple myeloma": "myelooma",
    "sekundaarinen aml": "AML",
    "secondary aml": "AML",
    "akuutti myelooinen leukemia": "AML",
    "CML": "KML",
    "krooninen myelooinen leukemia": "KML",
    "CLL": "KML",
    "krooninen lymfaattinen leukemia": "KLL",
    "plasmasoluleukemia": "PCL",
    "akuutti lymfoblastileukemia": "ALL",
    "b-all": "ALL",
    "ph + all": "ALL",
    "ph+all": "ALL",
    "kr. myeloprolif.": "MPN",
    "krooninen myeloproliferatiivinen tauti": "MPN",
    "MPN": "MPN",
    "essentielli trombosytemia": "et",
    "essential thrombocythemia": "et",
    "erinomainen vaste": "remissio",
    "täysi remissio": "remissio",
    "hyvä vaste": "remissio",
    "progressio": "relapsi",
    "relaboitunut": "relapsi",
}


def _norm(val) -> str:
    """Normalise a value for comparison: lowercase, strip, apply aliases."""
    if val is None or (isinstance(val, float)):
        return ""
    s = str(val).strip().lower()
    return _ALIASES.get(s, s)


def score_field(pred, gold, field: str):
    """
    Return (score, reason) where score is:
      1.0  exact / alias match
      0.5  partial match (one string contained in the other, min 3 chars)
      0.0  mismatch
    Both-null counts as 1.0 (correct negative).
    Pred-null / gold-null counts as 0.0 (false negative / false positive).
    """
    p = _norm(pred)
    g = _norm(gold)

    # Both empty → correct negative
    if p == "" and g == "":
        return 1.0, "both null ✓"

    # One empty, other not
    if p == "" and g != "":
        return 0.0, f"false negative (gold={repr(gold)})"
    if p != "" and g == "":
        return 0.0, f"false positive (pred={repr(pred)})"

    # Exact match (after normalisation)
    if p == g:
        return 1.0, "exact ✓"

    # For status fields: check if both map to the same standard category
    if field in STATUS_FIELDS:
        def to_std(v):
            for s in STANDARD_STATUSES:
                if s in v:
                    return s
            return v
        if to_std(p) == to_std(g) and to_std(p) in STANDARD_STATUSES:
            return 1.0, f"status match ✓ ({to_std(p)})"

    # Partial match: one contains the other (min 3 chars overlap)
    min_len = 3
    if len(p) >= min_len and len(g) >= min_len:
        if p in g or g in p:
            return 0.5, f"partial ◑ (pred={repr(pred)}, gold={repr(gold)})"

    return 0.0, f"mismatch ✗ (pred={repr(pred)}, gold={repr(gold)})"


def run_benchmark(
    df_full: pd.DataFrame,
    n_rows: int,
    model_a: str,
    model_b: str,
    batch_size: int,
    concurrency: int,
    ollama_url: str,
) -> None:
    """Run model_a and model_b on the first n_rows, score against gold, print report."""
    import os

    # ── Ground truth ────────────────────────────────────────────────────────
    gold_cols_present = [c for c in OUTPUT_FIELDS if c in df_full.columns]
    if not gold_cols_present:
        print("ERROR: no gold columns found in input. Run extraction first and "
              "correct rows manually, then use those as ground truth.", file=sys.stderr)
        sys.exit(1)

    df_bench = df_full.iloc[:n_rows].copy()
    gold = df_bench[gold_cols_present].copy()

    # Strip gold columns so models get a clean input
    df_input = df_bench.drop(columns=OUTPUT_FIELDS, errors="ignore")

    # ── Run each model ───────────────────────────────────────────────────────
    models = {"model_a": model_a, "model_b": model_b}
    all_preds = {}
    timings   = {}

    for key, model in models.items():
        label = f"{key} ({model})"
        print(f"\n{'─'*60}")
        print(f"Running {label} on {n_rows} rows …")
        t0 = time.time()
        preds = asyncio.run(
            run_ollama(df_input, model, batch_size, concurrency, ollama_url)
        )
        elapsed = time.time() - t0
        timings[key] = elapsed
        all_preds[key] = pd.DataFrame(preds, columns=OUTPUT_FIELDS)
        print(f"  Finished in {elapsed:.1f}s ({elapsed/n_rows:.2f}s/row)")

    # ── Score ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  BENCHMARK RESULTS  —  {n_rows} labelled rows")
    print(f"{'═'*72}")

    field_scores = {
        f: {k: [] for k in models} for f in OUTPUT_FIELDS
    }
    diffs = []  # (row, field, model_key, pred, gold_val, score)

    for i in range(n_rows):
        for field in OUTPUT_FIELDS:
            gold_val = gold[field].iloc[i] if field in gold.columns else None
            if isinstance(gold_val, float):
                gold_val = None
            for key in models:
                pred_val = all_preds[key][field].iloc[i]
                if isinstance(pred_val, float):
                    pred_val = None
                sc, reason = score_field(pred_val, gold_val, field)
                field_scores[field][key].append(sc)
                if sc < 1.0:
                    diffs.append((i, field, key, pred_val, gold_val, reason))

    # ── Per-field table ──────────────────────────────────────────────────────
    col_w = max(len(model_a), len(model_b), 22)
    header = f"  {'Field':<28}  {model_a[:col_w]:<{col_w}}  {model_b[:col_w]:<{col_w}}"
    print(header)
    print("  " + "-" * (28 + 2 * col_w + 4))

    overall = {k: [] for k in models}
    for field in OUTPUT_FIELDS:
        parts = []
        for key in models:
            scores = field_scores[field][key]
            mean = sum(scores) / len(scores) if scores else 0
            overall[key].extend(scores)
            exact_n = sum(1 for s in scores if s == 1.0)
            part_n  = sum(1 for s in scores if s == 0.5)
            parts.append(f"{mean*100:5.1f}%  ({exact_n}✓ {part_n}◑ {len(scores)-exact_n-part_n}✗)")
        print(f"  {field:<28}  {parts[0]:<{col_w}}  {parts[1]:<{col_w}}")

    print("  " + "-" * (28 + 2 * col_w + 4))
    parts = []
    for key in models:
        sc = overall[key]
        m = sum(sc) / len(sc) if sc else 0
        exact_n = sum(1 for s in sc if s == 1.0)
        part_n  = sum(1 for s in sc if s == 0.5)
        parts.append(f"{m*100:5.1f}%  ({exact_n}✓ {part_n}◑ {len(sc)-exact_n-part_n}✗)")
    print(f"  {'OVERALL':<28}  {parts[0]:<{col_w}}  {parts[1]:<{col_w}}")

    # ── Speed table ──────────────────────────────────────────────────────────
    print(f"\n  {'Speed':<28}  {model_a[:col_w]:<{col_w}}  {model_b[:col_w]:<{col_w}}")
    print("  " + "-" * (28 + 2 * col_w + 4))
    rates = []
    for key, model in models.items():
        t = timings[key]
        rps = n_rows / t
        est_50k = 50_000 / rps / 3600
        rates.append(f"{rps:.2f} row/s  (50k ≈ {est_50k:.1f}h)")
    print(f"  {'Throughput':<28}  {rates[0]:<{col_w}}  {rates[1]:<{col_w}}")

    # ── Diff listing ─────────────────────────────────────────────────────────
    if diffs:
        print(f"\n  DISAGREEMENTS ({len(diffs)} total, showing first 30):")
        print("  " + "-" * 72)
        shown = 0
        for row_i, field, key, pred, gold_val, reason in diffs[:30]:
            model_label = model_a if key == "model_a" else model_b
            print(f"  Row {row_i:2d} | {field:<28} | {model_label[:18]:<18} | {reason}")
            shown += 1

    # ── Save to Excel ────────────────────────────────────────────────────────
    out_path = "benchmark_results.xlsx"

    rows_out = []
    for i in range(n_rows):
        row = {"row": i}
        for field in OUTPUT_FIELDS:
            gv = gold[field].iloc[i] if field in gold.columns else None
            row[f"gold_{field}"] = None if isinstance(gv, float) else gv
        for key, label in [("model_a", model_a), ("model_b", model_b)]:
            for field in OUTPUT_FIELDS:
                pv = all_preds[key][field].iloc[i]
                row[f"{key}_{field}"] = None if isinstance(pv, float) else pv
                sc, _ = score_field(pv, row[f"gold_{field}"], field)
                row[f"{key}_{field}_score"] = sc
        rows_out.append(row)

    pd.DataFrame(rows_out).to_excel(out_path, index=False)
    print(f"\n  Detailed results saved → {out_path}")
    print(f"{'═'*72}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Finnish hematology NLP extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── shared arguments ────────────────────────────────────────────────────
    parser.add_argument("--input",       required=True,
                        help="Input Excel file (.xlsx)")
    parser.add_argument("--batch-size",  type=int, default=5,
                        help="Rows per LLM call (default 5)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Parallel batches for Ollama (default 3)")
    parser.add_argument("--ollama-url",  default="http://localhost:11434")

    # ── normal extraction ────────────────────────────────────────────────────
    parser.add_argument("--output",      default=None,
                        help="Output Excel file (.xlsx) — required unless --benchmark")
    parser.add_argument("--backend",     default="ollama", choices=["ollama", "anthropic"])
    parser.add_argument("--model",       default=None,
                        help="Default: qwen2.5:14b (ollama) / claude-haiku-4-5-20251001 (anthropic)")
    parser.add_argument("--no-format",   action="store_true",
                        help="Skip Excel colour formatting (faster for large files)")

    # ── benchmark mode ───────────────────────────────────────────────────────
    parser.add_argument("--benchmark",   action="store_true",
                        help="Run two models head-to-head against labelled ground truth")
    parser.add_argument("--model-a",     default="qwen2.5:14b",
                        help="First model for benchmark (default: qwen2.5:14b)")
    parser.add_argument("--model-b",     default="hf.co/LumiOpen/Llama-Poro-2-8B-Instruct",
                        help="Second model for benchmark "
                             "(default: hf.co/LumiOpen/Llama-Poro-2-8B-Instruct)")
    parser.add_argument("--benchmark-rows", type=int, default=20,
                        help="How many labelled rows to use (default: 20, uses rows 0…N-1)")

    args = parser.parse_args()

    print(f"Loading {args.input} …")
    df = pd.read_excel(args.input, engine='openpyxl')
    print(f"  {len(df)} rows | columns: {df.columns.tolist()}")

    # ── BENCHMARK MODE ───────────────────────────────────────────────────────
    if args.benchmark:
        n = min(args.benchmark_rows, len(df))
        print(f"\nBenchmark mode: {n} rows | "
              f"model-a={args.model_a} | model-b={args.model_b}")
        run_benchmark(
            df_full=df,
            n_rows=n,
            model_a=args.model_a,
            model_b=args.model_b,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            ollama_url=args.ollama_url,
        )
        return

    # ── NORMAL EXTRACTION MODE ───────────────────────────────────────────────
    if not args.output:
        parser.error("--output is required when not using --benchmark")

    if args.model is None:
        args.model = ("qwen2.5:14b" if args.backend == "ollama"
                      else "claude-haiku-4-5-20251001")

    print(f"Backend: {args.backend} | Model: {args.model} | "
          f"Batch: {args.batch_size} | Concurrency: {args.concurrency}")

    # Drop any previously extracted columns so we can re-run cleanly
    df = df.drop(columns=[c for c in OUTPUT_FIELDS if c in df.columns], errors="ignore")

    t0 = time.time()
    if args.backend == "ollama":
        results = asyncio.run(run_ollama(df, args.model, args.batch_size,
                                          args.concurrency, args.ollama_url))
    else:
        results = run_anthropic(df, args.model, args.batch_size)
    elapsed = time.time() - t0

    result_df = pd.DataFrame(results, columns=OUTPUT_FIELDS)

    # Insert extracted columns right after 'report'
    insert_after = "report" if "report" in df.columns else df.columns[-1]
    pos = list(df.columns).index(insert_after) + 1
    for i, col in enumerate(OUTPUT_FIELDS):
        df.insert(pos + i, col, result_df[col])

    if args.no_format:
        df.to_excel(args.output, index=False)
    else:
        write_formatted_excel(df, args.output)

    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(df):.2f}s/row) → {args.output}")
    for col in OUTPUT_FIELDS:
        n = df[col].notna().sum()
        print(f"  {col}: {n}/{len(df)} non-null ({100*n//len(df)}%)")


if __name__ == "__main__":
    main()
