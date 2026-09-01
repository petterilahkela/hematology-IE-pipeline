"""
prepare_finetune_data.py
========================
Converts annotated hematology Excel file to JSONL format for LoRA fine-tuning.

Each Excel row becomes TWO training examples:
  - Example 1: referral text → 4 fields (stage 1)
  - Example 2: report text + stage 1 context → 2 fields (stage 2)

Usage:
    python3 prepare_finetune_data.py \
        --input /mnt/data/petteril/data/finetune_data.xlsx \
        --output_dir /mnt/data/petteril/data/finetune/
"""

import pandas as pd
import json
import argparse
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def clean_value(val):
    """Convert NaN, empty strings, and null-like values to None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in {"", "null", "none", "nan", "-", "n/a"}:
        return None
    return s

def init_rag(training_file):
    print(f"Building RAG index from {training_file} ...")
    df = pd.read_excel(training_file)
    
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


def retrieve_similar(current_report, tfidf, vectors, training_data, k=2, min_similarity=0.3, max_yhteenveto_chars=1000):
    query_vec = tfidf.transform([current_report])
    sims = cosine_similarity(query_vec, vectors)[0]
    top_indices = sims.argsort()[::-1]
    retrieved = []
    for idx in top_indices:
        if sims[idx] < min_similarity:
            break
        case = training_data[idx]
        report = str(case.get('report', ''))
        yhteenveto = report[report.find('YHTEENVETO'):] if 'YHTEENVETO' in report else report[-1000:]
        if len(yhteenveto) > max_yhteenveto_chars:
            continue
        if case['report'] != current_report and pd.notna(case.get('cur_dg')):
            retrieved.append(case)
        if len(retrieved) == k:
            break
    return retrieved




def row_to_examples(row, retrieved_referrals=None, retrieved_reports=None):
    """
    Convert one annotated Excel row into two training examples.
    Returns a list of two dicts: [stage1_example, stage2_example]
    """
    referral = str(row.get("referral") or "")[:4000]
    report   = str(row.get("report")   or "")[:4000]

    # Clean nan values
    if referral.strip().lower() == "nan":
        referral = ""
    if report.strip().lower() == "nan":
        report = ""

    # Stage 1: referral → 4 fields
    stage1_output = {f: clean_value(row.get(f)) for f in STAGE1_FIELDS}
    
    
    rag_text_referral = ""
    if retrieved_referrals:
        rag_text_referral = "Samankaltaisia tapauksia:\n\n"
        for k, case in enumerate(retrieved_referrals):
            ref_full = str(case.get('referral', ''))
            ref = ref_full[:250] + ref_full[-250:] if len(ref_full) > 500 else ref_full
            if ref.strip().lower() == "nan":
                ref = ""
            output = {f: clean_value(case.get(f)) for f in STAGE1_FIELDS}
            rag_text_referral += f"Tapaus {k+1}:\nLÄHETE: {ref}\n"
            rag_text_referral += f"{json.dumps(output, ensure_ascii=False)}\n\n"
        rag_text_referral += "Käsiteltävä tapaus:\n"

    example1 = {
        "system":    SYSTEM_PROMPT_REFERRAL,
        "user":      f"{rag_text_referral}LÄHETE: {referral}",
        "assistant": json.dumps(stage1_output, ensure_ascii=False)
    }

    # Stage 2: report + stage 1 context + RAG examples → 2 fields
    stage2_output = {f: clean_value(row.get(f)) for f in STAGE2_FIELDS}
    context = (
        f"KONTEKSTI LÄHETTEESTÄ: "
        f"suspected_dg={stage1_output.get('suspected_dg')}, "
        f"prev_cur_dg={stage1_output.get('prev_cur_dg')}"
    )

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

    example2 = {
        "system":    SYSTEM_PROMPT_REPORT,
        "user":      f"{rag_text}LAUSUNTO: {report}\n{context}",
        "assistant": json.dumps(stage2_output, ensure_ascii=False)
    }

    return [example1, example2]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare RAG fine-tuning data from annotated Excel files")
    parser.add_argument("--train_file", required=True, help="Train Excel file (.xlsx)")
    parser.add_argument("--val_file",   required=True, help="Val Excel file (.xlsx)")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    args = parser.parse_args()

    # Load train and val
    print(f"Loading train: {args.train_file}")
    train_df = pd.read_excel(args.train_file, engine="openpyxl")
    print(f"  {len(train_df)} rows")

    print(f"Loading val: {args.val_file}")
    val_df = pd.read_excel(args.val_file, engine="openpyxl")
    print(f"  {len(val_df)} rows")

    # Build RAG index from training data only
    tfidf_report, vectors_report, tfidf_referral, vectors_referral, rag_training_data = init_rag(args.train_file)

    def process_df(df, label):
        examples = []
        skipped = 0
        for _, row in df.iterrows():
            if pd.isna(row.get("referral")) and pd.isna(row.get("report")):
                skipped += 1
                continue
            current_report   = str(row.get("report")   or "")
            current_referral = str(row.get("referral") or "")
            retrieved_reports   = retrieve_similar(current_report,   tfidf_report,   vectors_report,   rag_training_data, k=2)
            retrieved_referrals = retrieve_similar(current_referral, tfidf_referral, vectors_referral, rag_training_data, k=2)
            exs = row_to_examples(row, retrieved_referrals, retrieved_reports)
            examples.extend([format_as_chat(e) for e in exs])
        print(f"  {label}: {len(examples)} examples created ({skipped} rows skipped)")
        return examples

    train_examples = process_df(train_df, "Train")
    val_examples   = process_df(val_df,   "Val")

    print(f"\n  Train: {len(train_examples)} examples")
    print(f"  Val:   {len(val_examples)} examples")

    # Save to JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_v3.jsonl")
    val_path   = os.path.join(args.output_dir, "val_v3.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  Train → {train_path}")
    print(f"  Val   → {val_path}")

    print(f"\nSample training example (first one):")
    sample = train_examples[0]
    print(f"  system | length: {len(sample['messages'][0]['content'])} chars")
    print(f"  user   | {sample['messages'][1]['content'][:100]}...")
    print(f"  asst   | {sample['messages'][2]['content']}")



def format_as_chat(example):
    """Format example in the standard Llama/Poro2 chat format."""
    return {
        "messages": [
            {"role": "system",    "content": example["system"]},
            {"role": "user",      "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]}
        ]
    }

if __name__ == "__main__":
    main()
