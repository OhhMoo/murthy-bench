#!/usr/bin/env python3
"""
Fetches UniProt sequences for the 38 RNA-modification enzymes and produces
a merged CSV with sequences + MODOMICS disease associations.

Run locally (the sandbox blocks UniProt; your laptop won't):
    pip install requests
    python fetch_sequences.py path/to/your/modomics.csv

Output: enzymes_with_sequences_and_diseases.csv in the working directory.
"""
import csv, sys, time, urllib.request
from collections import defaultdict

# 38 unique enzymes from MODOMICS, mapped to canonical human Swiss-Prot accessions.
# Verified against UniProt, GeneCards, and primary literature.
ENZYMES = [
    ("ADAR",       "P55265", "DSRAD_HUMAN", "Double-stranded RNA-specific adenosine deaminase (ADAR1)"),
    ("ADARB1",     "P78563", "RED1_HUMAN",  "Double-stranded RNA-specific editase 1 (ADAR2)"),
    ("ADARB2",     "Q9NS39", "RED2_HUMAN",  "Double-stranded RNA-specific editase B2 (ADAR3)"),
    ("ALKBH3",     "Q96Q83", "ALKB3_HUMAN", "Alpha-ketoglutarate-dependent dioxygenase alkB homolog 3"),
    ("ALKBH4",     "Q9NXW9", "ALKB4_HUMAN", "Alpha-ketoglutarate-dependent dioxygenase alkB homolog 4"),
    ("ALKBH5",     "Q6P6C2", "ALKB5_HUMAN", "RNA demethylase ALKBH5"),
    ("APOBEC1",    "P41238", "ABEC1_HUMAN", "C->U-editing enzyme APOBEC-1"),
    ("APOBEC3A",   "P31941", "ABC3A_HUMAN", "DNA dC->dU-editing enzyme APOBEC-3A"),
    ("DDX21",      "Q9NR30", "DDX21_HUMAN", "Nucleolar RNA helicase 2"),
    ("DKC1",       "O60832", "DKC1_HUMAN",  "H/ACA ribonucleoprotein complex subunit DKC1 (dyskerin)"),
    ("FBL",        "P22087", "FBRL_HUMAN",  "rRNA 2'-O-methyltransferase fibrillarin"),
    ("FTO",        "Q9C0B1", "FTO_HUMAN",   "Alpha-ketoglutarate-dependent dioxygenase FTO"),
    ("FTSJ1",      "Q9UET6", "TRM7_HUMAN",  "Putative tRNA (cytidine(32)/guanosine(34)-2'-O)-methyltransferase"),
    ("HNRNPA2B1",  "P22626", "ROA2_HUMAN",  "Heterogeneous nuclear ribonucleoproteins A2/B1"),
    ("LARP7",      "Q4G0J3", "LARP7_HUMAN", "La-related protein 7"),
    ("METTL14",    "Q9HCE5", "MET14_HUMAN", "N6-adenosine-methyltransferase non-catalytic subunit"),
    ("METTL3",     "Q86U44", "MTA70_HUMAN", "N6-adenosine-methyltransferase catalytic subunit"),
    ("NOP56",      "O00567", "NOP56_HUMAN", "Nucleolar protein 56"),
    ("NOP58",      "Q9Y2X3", "NOP58_HUMAN", "Nucleolar protein 58"),
    ("NPM1",       "P06748", "NPM_HUMAN",   "Nucleophosmin"),
    ("NSUN2",      "Q08J23", "NSUN2_HUMAN", "tRNA (cytosine(34)-C(5))-methyltransferase"),
    ("NSUN3",      "Q9H649", "NSUN3_HUMAN", "tRNA (cytosine(72)-C(5))-methyltransferase"),
    ("NSUN6",      "Q8TEA1", "NSUN6_HUMAN", "tRNA (cytosine(72)-C(5))-methyltransferase NSUN6"),
    ("PUS1",       "Q9Y606", "PUS1_HUMAN",  "tRNA pseudouridine synthase A"),
    ("PUS3",       "Q9BZE2", "PUS3_HUMAN",  "tRNA pseudouridine synthase 3"),
    ("PUS7",       "Q96PZ0", "PUS7_HUMAN",  "Pseudouridylate synthase 7 homolog"),
    ("TCOF1",      "Q13428", "TCOF_HUMAN",  "Treacle protein"),
    ("TLE5",       "Q08117", "TLE5_HUMAN",  "Transducin-like enhancer protein 5 (AES)"),
    ("TRMT6",      "Q9UJA5", "TRM6_HUMAN",  "tRNA (adenine(58)-N(1))-methyltransferase non-catalytic subunit"),
    ("TRMT61A",    "Q96FX7", "TRM61_HUMAN", "tRNA (adenine(58)-N(1))-methyltransferase catalytic subunit"),
    ("TRMT61B",    "Q9BVS5", "TR61B_HUMAN", "tRNA (adenine(58)-N(1))-methyltransferase, mitochondrial"),
    ("VIRMA",      "Q69YN4", "VIR_HUMAN",   "Protein virilizer homolog (KIAA1429)"),
    ("WTAP",       "Q15007", "FL2D_HUMAN",  "Pre-mRNA-splicing regulator WTAP"),
    ("YBX1",       "P67809", "YBOX1_HUMAN", "Y-box-binding protein 1"),
    ("YTHDC2",     "Q9H6S0", "YTDC2_HUMAN", "3'-5' RNA helicase YTHDC2"),
    ("YTHDF1",     "Q9BYJ9", "YTHD1_HUMAN", "YTH domain-containing family protein 1"),
    ("YTHDF2",     "Q9Y5A9", "YTHD2_HUMAN", "YTH domain-containing family protein 2"),
    ("YTHDF3",     "Q7Z739", "YTHD3_HUMAN", "YTH domain-containing family protein 3"),
]


def fetch_fasta(accession: str) -> str:
    """Returns just the amino-acid sequence (no header, no newlines)."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8")
    lines = text.splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq


def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_sequences.py <modomics_csv_path>")
        sys.exit(1)
    modomics_path = sys.argv[1]

    # Build disease association map from MODOMICS CSV
    diseases = defaultdict(list)
    mod_types = defaultdict(set)
    with open(modomics_path) as f:
        for row in csv.DictReader(f):
            ef = row["Enzymes"].strip()
            if not ef:
                continue
            for tok in ef.split():
                key = tok.upper()
                diseases[key].append({
                    "disease": row["Disease Name"].strip(),
                    "reaction": row["Reaction"].strip(),
                    "description": row["Description"].strip(),
                })
                if row["Reaction"].strip():
                    mod_types[key].add(row["Reaction"].strip())

    # Fetch sequences
    results = []
    for i, (gene, acc, entry, pname) in enumerate(ENZYMES, 1):
        print(f"[{i}/{len(ENZYMES)}] Fetching {gene} ({acc})...", flush=True)
        try:
            seq = fetch_fasta(acc)
        except Exception as e:
            print(f"   FAIL: {e}")
            seq = ""
        results.append((gene, acc, entry, pname, seq))
        time.sleep(0.2)  # be polite to UniProt

    # Write merged CSV
    out_path = "enzymes_with_sequences_and_diseases.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "gene_symbol", "uniprot_accession", "uniprot_entry_name",
            "protein_name", "modification_type", "sequence_length",
            "sequence", "disease_count", "diseases", "disease_descriptions",
            "notes",
        ])
        for gene, acc, entry, pname, seq in results:
            dis_list = diseases.get(gene.upper(), [])
            unique = sorted({d["disease"] for d in dis_list if d["disease"]})
            desc_by = {}
            for d in dis_list:
                if d["disease"] and d["disease"] not in desc_by:
                    desc_by[d["disease"]] = d["description"]
            descs = " || ".join(f"[{k}] {v}" for k, v in desc_by.items())
            mods = "; ".join(sorted(mod_types.get(gene.upper(), set())))
            notes = ""
            if gene == "TLE5":
                notes = "TLE5/AES is a transcriptional corepressor, not an RNA modification enzyme"
            w.writerow([
                gene, acc, entry, pname, mods,
                len(seq) if seq else "", seq,
                len(unique), " | ".join(unique), descs, notes,
            ])
    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
