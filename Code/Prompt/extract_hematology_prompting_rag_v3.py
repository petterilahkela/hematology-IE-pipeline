"""
extract_hematology_prompting_rag_v3.py
=======================================
Extracts structured fields from Finnish hematology referrals and bone marrow reports.
Uses two TF-IDF RAG indexes (referral + report) matching the V3 fine-tuning setup.
No static examples — only dynamic RAG examples, consistent with training data.

Usage:
    python3 extract_hematology_prompting_rag_v3.py \
        --input   /mnt/data/petteril/V3/Data/test_v3.xlsx \
        --output  /mnt/data/petteril/V3/Results/results_poro2_ft_rag_v3.xlsx \
        --model   poro2-finetuned-rag-v3 \
        --training-file /mnt/data/petteril/V3/Data/train_v3.xlsx \
        --batch-size 1 --concurrency 1
"""

import pandas as pd
import json
import asyncio
import aiohttp
import argparse
import sys
import time
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS — identical to prepare_finetune_data_rag_v3.py (no static examples)
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
"""

# ──────────────────────────────────────────────────────────────────────────────
# FIELD DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

STAGE1_FIELDS = ["suspected_dg", "prev_cur_dg", "prev_cur_dg_status", "treatment"]
STAGE2_FIELDS = ["cur_dg", "cur_dg_status"]
OUTPUT_FIELDS = STAGE1_FIELDS + STAGE2_FIELDS

EMPTY = {f: None for f in OUTPUT_FIELDS}


# ──────────────────────────────────────────────────────────────────────────────
# RAG INDEX — two separate indexes for referral and report
# ──────────────────────────────────────────────────────────────────────────────

def init_rag(training_file):
    print(f"Building RAG index from {training_file} ...")
    df = pd.read_excel(training_file, engine='openpyxl')

    # Report index (YHTEENVETO only)
    reports = []
    for r in df['report'].astype(str):
        if 'YHTEENVETO' in r:
            reports.append(r[r.find('YHTEENVETO'):])
        else:
            reports.append(r[:500])
    tfidf_report = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    vectors_report = tfidf_report.fit_transform(reports)

    # Referral index
    referrals = df['referral'].astype(str).tolist()
    tfidf_referral = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    vectors_referral = tfidf_referral.fit_transform(referrals)

    print(f"  RAG index built: {len(reports)} training cases")
    return tfidf_report, vectors_report, tfidf_referral, vectors_referral, df.to_dict('records')


def retrieve_similar(current_text, tfidf, vectors, training_data,
                     text_field, k=2, min_similarity=0.3, max_yhteenveto_chars=1000):
    query_vec = tfidf.transform([current_text])
    sims = cosine_similarity(query_vec, vectors)[0]
    top_indices = sims.argsort()[::-1]
    retrieved = []
    for idx in top_indices:
        if sims[idx] < min_similarity:
            break
        case = training_data[idx]
        # For report retrieval, check YHTEENVETO length
        if text_field == 'report':
            report = str(case.get('report', ''))
            yhteenveto = report[report.find('YHTEENVETO'):] if 'YHTEENVETO' in report else report[-1000:]
            if len(yhteenveto) > max_yhteenveto_chars:
                continue
        # Skip exact same row
        if case.get(text_field) != current_text and pd.notna(case.get('cur_dg')):
            retrieved.append(case)
        if len(retrieved) == k:
            break
    return retrieved


def clean_value(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.lower() in {"", "null", "none", "nan", "-", "n/a"}:
        return None
    return s


# ──────────────────────────────────────────────────────────────────────────────
# MESSAGE BUILDERS — single row, with RAG examples
# ──────────────────────────────────────────────────────────────────────────────

def build_referral_msg(referral, retrieved_referrals):
    """Build stage 1 user message with dynamic RAG referral examples."""
    ref_clean = str(referral or "")[:4000]
    if ref_clean.strip().lower() == "nan":
        ref_clean = ""

    rag_text = ""
    if retrieved_referrals:
        rag_text = "Samankaltaisia tapauksia:\n\n"
        for k, case in enumerate(retrieved_referrals):
            ref_full = str(case.get('referral', ''))
            ref = ref_full[:250] + ref_full[-250:] if len(ref_full) > 500 else ref_full
            if ref.strip().lower() == "nan":
                ref = ""
            output = {f: clean_value(case.get(f)) for f in STAGE1_FIELDS}
            rag_text += f"Tapaus {k+1}:\nLÄHETE: {ref}\n"
            rag_text += f"{json.dumps(output, ensure_ascii=False)}\n\n"
        rag_text += "Käsiteltävä tapaus:\n"

    return f"{rag_text}LÄHETE: {ref_clean}"


def build_report_msg(report, stage1_result, retrieved_reports):
    """Build stage 2 user message with dynamic RAG report examples."""
    rep_clean = str(report or "")[:4000]
    if rep_clean.strip().lower() == "nan":
        rep_clean = ""

    rag_text = ""
    if retrieved_reports:
        rag_text = "Samankaltaisia tapauksia:\n\n"
        for k, case in enumerate(retrieved_reports):
            yhteenveto = case.get('report', '')
            if str(yhteenveto).strip().lower() == "nan":
                yhteenveto = ""
            elif 'YHTEENVETO' in str(yhteenveto):
                yhteenveto = str(yhteenveto)[str(yhteenveto).find('YHTEENVETO'):]
            else:
                yhteenveto = str(yhteenveto)[-1000:]
            cur_dg = case.get('cur_dg', '')
            cur_dg_status = case.get('cur_dg_status', '')
            rag_text += f"Tapaus {k+1}:\nLAUSUNTO: {yhteenveto}\n"
            rag_text += f'{{"cur_dg": "{cur_dg}", "cur_dg_status": "{cur_dg_status}"}}\n\n'
        rag_text += "Käsiteltävä tapaus:\n"

    context = (
        f"KONTEKSTI LÄHETTEESTÄ: "
        f"suspected_dg={stage1_result.get('suspected_dg')}, "
        f"prev_cur_dg={stage1_result.get('prev_cur_dg')}"
    )

    return f"{rag_text}LAUSUNTO: {rep_clean}\n{context}"


# ──────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_json_response(text, fields):
    """Parse JSON from model response, return dict with given fields."""
    text = text.strip()
    # Try direct parse
    try:
        data = json.loads(text)
        return {f: data.get(f) for f in fields}
    except json.JSONDecodeError:
        pass
    # Try extracting JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {f: data.get(f) for f in fields}
        except json.JSONDecodeError:
            pass
    return {f: None for f in fields}


# ──────────────────────────────────────────────────────────────────────────────
# OLLAMA BACKEND
# ──────────────────────────────────────────────────────────────────────────────

async def ollama_call(session, model, user_msg, system, base_url, retries=3):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 768, "num_ctx": 8192},
    }
    for attempt in range(retries):
        try:
            async with session.post(
                f"{base_url}/api/chat", json=payload,
                timeout=aiohttp.ClientTimeout(total=240)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"\n[WARN] API call failed: {e}", file=sys.stderr)
                return ""
            await asyncio.sleep(2 ** attempt)
    return ""


async def process_row(sem, session, model, row, base_url,
                      tfidf_report, vectors_report,
                      tfidf_referral, vectors_referral,
                      rag_training_data):
    async with sem:
        referral = str(row.get("referral") or "")
        report   = str(row.get("report")   or "")

        # Retrieve similar cases
        retrieved_referrals = retrieve_similar(
            referral, tfidf_referral, vectors_referral, rag_training_data, 'referral')
        retrieved_reports = retrieve_similar(
            report, tfidf_report, vectors_report, rag_training_data, 'report')

        global rag_ref_found, rag_rep_found
        if retrieved_referrals:
            rag_ref_found += 1
        if retrieved_reports:
            rag_rep_found += 1

        # Stage 1: referral
        ref_msg = build_referral_msg(referral, retrieved_referrals)
        text1   = await ollama_call(session, model, ref_msg, SYSTEM_PROMPT_REFERRAL, base_url)
        result1 = parse_json_response(text1, STAGE1_FIELDS)

        # Stage 2: report
        rep_msg = build_report_msg(report, result1, retrieved_reports)
        text2   = await ollama_call(session, model, rep_msg, SYSTEM_PROMPT_REPORT, base_url)
        result2 = parse_json_response(text2, STAGE2_FIELDS)

        return {**result1, **result2}

rag_ref_found = 0
rag_rep_found = 0

async def run_ollama(df, model, batch_size, concurrency, base_url,
                     tfidf_report, vectors_report,
                     tfidf_referral, vectors_referral,
                     rag_training_data):
    sem  = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 2)
    all_results = [None] * len(df)

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for i in range(len(df)):
            row = {
                "referral": str(df.iloc[i].get("referral") or "")[:4000],
                "report":   str(df.iloc[i].get("report")   or "")[:4000],
            }
            task = asyncio.ensure_future(
                process_row(sem, session, model, row, base_url,
                            tfidf_report, vectors_report,
                            tfidf_referral, vectors_referral,
                            rag_training_data)
            )
            tasks.append((i, task))

        pbar = tqdm(total=len(df), desc="Extracting")
        for i, task in tasks:
            all_results[i] = await task
            pbar.update(1)
        pbar.close()

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# EXCEL OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def write_formatted_excel(out_df, path):
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    out_df.to_excel(path, index=False)
    wb = load_workbook(path)
    ws = wb.active

    HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
    NEW_COL_FILL = PatternFill("solid", fgColor="EBF5FB")
    STATUS_COLORS = {
        "remissio":      "D5F5E3",
        "relapsi":       "FADBD8",
        "uusi diagnoosi": "FEF9E7",
        "refraktaari":   "F9EBEA",
    }
    header_font = Font(bold=True, color="FFFFFF")
    extracted_cols = set(OUTPUT_FIELDS)
    headers = [cell.value for cell in ws[1]]

    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            col_name = headers[cell.column - 1]
            val = str(cell.value or "").strip().lower()
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

    for i in range(1, ws.max_row + 1):
        ws.row_dimensions[i].height = 20

    ws.freeze_panes = "A2"
    wb.save(path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Finnish hematology NLP extraction pipeline — RAG V3",
    )
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--model",         default="poro2-finetuned-rag-v3")
    parser.add_argument("--batch-size",    type=int, default=1)
    parser.add_argument("--concurrency",   type=int, default=1)
    parser.add_argument("--ollama-url",    default="http://localhost:11434")
    parser.add_argument("--training-file", default="/mnt/data/petteril/V3/Data/train_v3.xlsx",
                        help="Training data file for RAG index")
    parser.add_argument("--no-format",     action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_excel(args.input, engine='openpyxl')
    print(f"  {len(df)} rows | columns: {df.columns.tolist()}")

    print(f"Backend: ollama | Model: {args.model} | "
          f"Batch: {args.batch_size} | Concurrency: {args.concurrency}")

    # Drop any previously extracted columns
    df = df.drop(columns=[c for c in OUTPUT_FIELDS if c in df.columns], errors="ignore")

    # Build RAG indexes
    tfidf_report, vectors_report, tfidf_referral, vectors_referral, rag_training_data = \
        init_rag(args.training_file)

    t0 = time.time()
    results = asyncio.run(run_ollama(
        df, args.model, args.batch_size, args.concurrency, args.ollama_url,
        tfidf_report, vectors_report, tfidf_referral, vectors_referral, rag_training_data
    ))
    elapsed = time.time() - t0

    result_df = pd.DataFrame(results, columns=OUTPUT_FIELDS)

    insert_after = "report" if "report" in df.columns else df.columns[-1]
    pos = list(df.columns).index(insert_after) + 1
    for i, col in enumerate(OUTPUT_FIELDS):
        df.insert(pos + i, col, result_df[col])

    if args.no_format:
        df.to_excel(args.output, index=False)
    else:
        write_formatted_excel(df, args.output)

    print(f"\n[RAG] Referral examples found: {rag_ref_found}/{len(df)}")
    print(f"[RAG] Report examples found: {rag_rep_found}/{len(df)}")

    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(df):.2f}s/row) → {args.output}")
    for col in OUTPUT_FIELDS:
        n = df[col].notna().sum()
        print(f"  {col}: {n}/{len(df)} non-null ({100*n//len(df)}%)")


if __name__ == "__main__":
    main()
