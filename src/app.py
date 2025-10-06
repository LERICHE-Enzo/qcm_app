import ast
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

APP_TITLE = "Révise tes QCM"

# ====== Répertoires ======
BASE_DIR = Path(__file__).resolve().parent.parent   # src/.. -> racine projet
DATA_DIR = BASE_DIR / "data"                        # data/<Ecole>/<Année>/<Matière>/questions.json


# ========================================================================
#                               UTILITAIRES
# ========================================================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
    alias = {
        "question": "question", "questions": "question",
        "choices": "choices", "choix": "choices", "propositions": "choices",
        "answer": "answer", "réponse": "answer", "reponse": "answer",
        "explanation": "explanation", "explication": "explanation",
        # on tolère ces champs mais on ne les utilise pas forcément
        "tags": "tags", "tag": "tags",
        "qcm": "qcm", "theme": "theme", "thème": "theme", "id": "id",
        "id_gen": "id_gen",
        "image": "image",
    }
    df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)

    # colonnes minimales
    required = {"question", "choices", "answer"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "Colonnes manquantes dans questions.json.\n"
            f"Requis: {required}.\n"
            f"Trouvé: {set(df.columns)}"
        )
        st.stop()

    # choices -> liste propre
    def _fix_choices(x):
        if isinstance(x, list):
            return [str(c).strip() for c in x]
        # fallback si ancien format CSV "A||B||C"
        return [c.strip() for c in str(x).split("||")]

    df["choices_list"] = df["choices"].apply(_fix_choices)

    # answer_parsed : int ou [int,...]
    def parse_answer(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, list):
            try:
                lst = [int(v) for v in x]
                return lst[0] if len(lst) == 1 else lst  # 🔧 aplatit [i] -> i
            except Exception:
                return None
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return int(x)
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                lst = ast.literal_eval(s)
                lst = [int(v) for v in lst]
                return lst[0] if len(lst) == 1 else lst  # 🔧 aplatit "[i]" -> i
            except Exception:
                return None
        try:
            return int(float(s))
        except Exception:
            return None

    df["answer_parsed"] = df["answer"].apply(parse_answer)

    # colonnes optionnelles
    for col in ["explanation", "tags", "qcm", "theme", "id", "id_gen", "image"]:
        if col not in df.columns:
            df[col] = None

    return df.reset_index(drop=True)


@st.cache_data
def load_questions_json(path: Path) -> pd.DataFrame:
    """Charge data/.../questions.json (liste ou {'questions': [...]}) en DataFrame normalisée.
       Ajoute la colonne 'orig_idx' pour référencer l'index original dans le fichier JSON, utile pour persister des lots."""
    if not path.exists():
        st.error(f"Fichier introuvable : {path}")
        st.stop()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        st.error(f"JSON invalide dans {path.name} : {e}")
        st.stop()

    if isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    elif isinstance(data, list):
        items = data
    else:
        st.error(f"Format JSON non reconnu dans {path.name}. Attendu liste ou {{'questions':[...]}}.")
        st.stop()

    df = pd.DataFrame(items)
    # très important : mémoriser l'index d'origine dans le fichier
    df["orig_idx"] = list(range(len(df)))
    df = _normalize_columns(df)
    return df


def load_errors(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_errors(path: Path, errors: list) -> None:
    try:
        path.write_text(json.dumps(errors, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except Exception as e:
        st.warning(f"Impossible d'enregistrer les erreurs : {e}")


def _fixed_session_key(school: str, year: str, subject: str, selected_qcm: Optional[str]) -> str:
    # "(Tous)" si pas de filtre QCM
    label_qcm = selected_qcm if (selected_qcm and selected_qcm != "(Tous)") else "(Tous)"
    return f"{school}|{year}|{subject}|{label_qcm}"


# ===== Diagnostiqueur =====
def validate_answers_and_choices(df_: pd.DataFrame):
    errors = []
    for i, row in df_.iterrows():
        q = str(row.get("question", "")).strip()
        ch_list = row.get("choices_list", [])
        ans = row.get("answer_parsed", None)

        if not q:
            errors.append({"row_idx": i, "issue": "question vide", "detail": ""})

        if not isinstance(ch_list, list) or len(ch_list) == 0:
            errors.append({"row_idx": i, "issue": "choices_list vide", "detail": ""})
        else:
            empty_opts = [j for j, c in enumerate(ch_list) if str(c).strip() == ""]
            if empty_opts:
                errors.append({
                    "row_idx": i,
                    "issue": "options vides",
                    "detail": f"indices: {empty_opts} (souvent '||' final en trop)"
                })

        if ans is None and row.get("answer", None) is not None:
            errors.append({
                "row_idx": i,
                "issue": "answer illisible",
                "detail": f"value={row.get('answer')!r} (attendu: entier ou liste [i,j])"
            })

        n = len(ch_list) if isinstance(ch_list, list) else 0

        def _bad_index(a):
            return not (isinstance(a, int) and 0 <= a < n)

        if isinstance(ans, list):
            bad = [a for a in ans if _bad_index(a)]
            if bad:
                errors.append({
                    "row_idx": i,
                    "issue": "indices hors plage (liste)",
                    "detail": f"indices invalides {bad} ; nb_options={n} (0..{max(n-1,0)})"
                })
        elif isinstance(ans, int):
            if _bad_index(ans):
                errors.append({
                    "row_idx": i,
                    "issue": "indice hors plage (entier)",
                    "detail": f"index={ans} ; nb_options={n} (0..{max(n-1,0)})"
                })
        else:
            if row.get("answer", None) in (None, ""):
                errors.append({"row_idx": i, "issue": "answer manquant", "detail": ""})

    return errors


def _freeze_order(df_: pd.DataFrame, shuffle_flag: bool):
    # Si on a généré un sous-ensemble aléatoire, on ne touche pas à l'ordre
    if st.session_state.get("custom_subset", False):
        return
    base_key = tuple(df_.index)
    need_init = (
        "indices" not in st.session_state
        or st.session_state.get("indices_base") != base_key
        or st.session_state.get("shuffle_q") != shuffle_flag
        or st.session_state.get("nb_rows") != len(df_)
    )
    if need_init:
        order = list(df_.index)
        if shuffle_flag:
            random.shuffle(order)
        st.session_state.indices = order
        st.session_state.indices_base = base_key
        st.session_state.shuffle_q = shuffle_flag
        st.session_state.nb_rows = len(df_)
        st.session_state.choice_shuffle = {}  # reset permutations options


def _map_true_answer(row_i: int, row_series: pd.Series):
    shuffle_order = st.session_state.choice_shuffle.get(
        row_i, list(range(len(row_series["choices_list"])))
    )
    choices = [row_series["choices_list"][k] for k in shuffle_order]
    map_old_to_new = {old: new for new, old in enumerate(shuffle_order)}
    true_orig = row_series["answer_parsed"]
    if isinstance(true_orig, list):
        true_shuf = [map_old_to_new[a] for a in true_orig]
    else:
        true_shuf = map_old_to_new[true_orig]
    return choices, true_shuf


def _compute_score(df_: pd.DataFrame):
    answered = 0
    correct = 0
    for i, row in df_.iterrows():
        if i not in st.session_state.answers:
            continue
        answered += 1
        _, true_ans = _map_true_answer(i, row)
        given = st.session_state.answers.get(i)
        if isinstance(true_ans, list):
            ok = isinstance(given, list) and sorted(given) == sorted(true_ans)
        else:
            ok = (given == true_ans)
        if ok:
            correct += 1
    return correct, answered


def _score_on_indices(df_: pd.DataFrame, indices_: list[int]):
    """Score uniquement sur le sous-ensemble présenté (st.session_state.indices)."""
    answered = 0
    correct = 0
    for i in indices_:
        if i not in st.session_state.answers:
            continue
        row = df_.loc[i]
        _, true_ans = _map_true_answer(i, row)
        given = st.session_state.answers.get(i)
        if isinstance(true_ans, list):
            ok = isinstance(given, list) and sorted(given) == sorted(true_ans)
        else:
            ok = (given == true_ans)
        if ok:
            correct += 1
        answered += 1
    return correct, answered


# ========================================================================
#         PERSISTENCE DES LOTS FIGÉS (5) + ANNOTATION DANS LE JSON
# ========================================================================
def _lots_file(subject_dir: Path) -> Path:
    return subject_dir / "lots_figes.json"

def load_fixed_lots(subject_dir: Path) -> dict:
    p = _lots_file(subject_dir)
    if not p.exists():
        return {"lots": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"lots": []}

def save_fixed_lots(subject_dir: Path, payload: dict) -> None:
    p = _lots_file(subject_dir)
    try:
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Impossible d'enregistrer lots_figes.json : {e}")


def _new_lot_id(existing_ids: set[str]) -> str:
    """
    Génère un ID court du type: qcmNNN_YYYY-MM-DD
    - NNN : compteur incrémental (001, 002, …) pour la date du jour
    - YYYY-MM-DD : date du jour
    """
    today = datetime.now().strftime("%Y-%m-%d")
    prefix = "qcm"
    # On cherche les IDs déjà existants de la forme qcmNNN_YYYY-MM-DD (pour AUJOURD’HUI)
    # afin d'incrémenter le compteur.
    max_n = 0
    for _id in existing_ids:
        # ignore les anciens formats (fixed5_…)
        if not _id or not _id.startswith(prefix):
            continue
        # attend un format "qcmNNN_YYYY-MM-DD"
        parts = _id.split("_")
        if len(parts) != 2:
            continue
        num_part = parts[0][len(prefix):]  # "NNN"
        date_part = parts[1]
        if date_part == today and num_part.isdigit():
            max_n = max(max_n, int(num_part))
    next_n = max_n + 1
    return f"{prefix}{next_n:03d}_{today}"


def annotate_questions_with_lot(questions_file: Path, orig_indices: list[int], lot_id: str):
    """
    Ajoute 'id_gen' (liste d'IDs) sur chaque question du lot dans questions.json.
    Les indices fournis sont des indices 'orig_idx' (ordre du fichier).
    """
    try:
        raw = json.loads(questions_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "questions" in raw:
            items = raw["questions"]
            is_wrapped = True
        elif isinstance(raw, list):
            items = raw
            is_wrapped = False
        else:
            st.warning("Format JSON non reconnu pour annotation id_gen (liste ou {'questions':[...]})")
            return

        for oi in orig_indices:
            if 0 <= oi < len(items):
                q = items[oi]
                cur = q.get("id_gen")
                if cur is None or cur == "" or cur == []:
                    q["id_gen"] = [lot_id]
                else:
                    if isinstance(cur, list):
                        if lot_id not in cur:
                            cur.append(lot_id)
                            q["id_gen"] = cur
                    else:
                        # ancien format scalaire -> liste
                        if cur != lot_id:
                            q["id_gen"] = [cur, lot_id]
                        else:
                            q["id_gen"] = [cur]

        new_raw = {"questions": items} if is_wrapped else items
        questions_file.write_text(json.dumps(new_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception as e:
        st.warning(f"Annotation id_gen échouée : {e}")


# ========================================================================
#                                   UI
# ========================================================================
st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")
st.title(APP_TITLE)

# ===== Sélection hiérarchique =====
with st.sidebar:
    st.header("📚 Choix du dataset")

    if not DATA_DIR.exists():
        st.error(f"Dossier des données introuvable : {DATA_DIR}")
        st.stop()

    schools = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    if not schools:
        st.error("Aucune école trouvée dans data/.")
        st.stop()
    school = st.selectbox("École", schools)

    school_dir = DATA_DIR / school
    years = sorted([d.name for d in school_dir.iterdir() if d.is_dir()])
    if not years:
        st.error("Aucune année trouvée pour cette école.")
        st.stop()
    year = st.selectbox("Année (1A, 2A, 3A, L3GE…)", years)

    year_dir = school_dir / year
    subjects = sorted([d.name for d in year_dir.iterdir() if d.is_dir()])
    if not subjects:
        st.error("Aucune matière trouvée pour cette année.")
        st.stop()
    subject = st.selectbox("Matière", subjects)
    subject_dir = year_dir / subject

    # Unique fichier requis
    QUESTIONS_FILE = subject_dir / "questions.json"
    ERRORS_FILE = subject_dir / "erreurs.json"

    # Paramètres
    st.header("Paramètres")
    mode = st.selectbox("Mode", ["Entraînement", "Examen blanc", "Révisions ciblées"])
    show_timer = st.toggle("Afficher un minuteur (indicatif)", value=False)
    shuffle_q = st.toggle("Mélanger l'ordre des questions", value=True)

# ===== Chargement du JSON unique =====
df = load_questions_json(QUESTIONS_FILE)

# Filtre QCM (si des champs 'qcm' existent)
available_qcms = sorted({q for q in df.get("qcm", pd.Series()).dropna().unique()})
if available_qcms:
    with st.sidebar:
        selected_qcm = st.selectbox("Filtrer par QCM", ["(Tous)"] + available_qcms)

    # reset si changement de QCM pour éviter indices décalés après un sous-ensemble
    prev_qcm = st.session_state.get("selected_qcm")
    if prev_qcm is not None and prev_qcm != selected_qcm:
        st.session_state.custom_subset = False
        st.session_state.indices = []
        st.session_state.idx_ptr = 0
        st.session_state.answers = {}
        st.session_state.choice_shuffle = {}
        st.session_state.current_lot_id = None
    st.session_state.selected_qcm = selected_qcm

    if selected_qcm != "(Tous)":
        # On filtre mais on garde 'orig_idx' qui référence le fichier d'origine
        df = df[df["qcm"] == selected_qcm].reset_index(drop=True)
else:
    selected_qcm = None

# ===== QCM aléatoire + Lot figé (5) =====
with st.sidebar:
    st.header("🎯 QCM aléatoire")
    k = st.number_input("Nombre de questions", min_value=1, max_value=50, value=5, step=1)
    if st.button("Générer ce QCM"):
        pop_indices = list(df.index)  # df est déjà filtré par QCM si sélectionné
        if len(pop_indices) == 0:
            st.warning("Aucune question disponible pour ce filtre.")
            st.stop()
        k = min(int(k), len(pop_indices))
        order = random.sample(pop_indices, k)

        # on fige ce sous-ensemble
        st.session_state.indices = order
        st.session_state.idx_ptr = 0
        st.session_state.answers = {}
        st.session_state.choice_shuffle = {}
        st.session_state.custom_subset = True
        st.session_state.current_lot_id = None
        st.rerun()

    # ===== Lot figé (5) ultra simple + PERSISTANT =====
    st.header("🔒 Lot figé (5)")

    if "fixed5" not in st.session_state:
        st.session_state.fixed5 = {}  # dict: key -> [indices]

    fixed_key = _fixed_session_key(
        school, year, subject, selected_qcm if available_qcms else None
    )

    # ===== Chargement / persistance des lots =====
    LOTS_PAYLOAD = load_fixed_lots(subject_dir)
    EXISTING_IDS = {lot.get("id") for lot in LOTS_PAYLOAD.get("lots", []) if lot.get("id")}

    # --- Sélecteur de lot existant (contexte courant) ---
    def _lot_matches_context(l):
        if l.get("school") != school or l.get("year") != year or l.get("subject") != subject:
            return False
        ctx_filter = selected_qcm if available_qcms else None
        return l.get("qcm_filter") == ctx_filter

    lots_here = [l for l in LOTS_PAYLOAD.get("lots", []) if _lot_matches_context(l)]
    lots_here = sorted(lots_here, key=lambda x: x.get("created_at",""), reverse=True)

    if lots_here:
        labels = [
            f"{l['id']}  ({l.get('created_at','')})  — {l.get('size', len(l.get('orig_indices', [])))} q"
            for l in lots_here
        ]
        chosen = st.selectbox("Recharger un lot figé existant", labels, index=0, key="fixed5_pick")
        picked = lots_here[labels.index(chosen)]

        if st.button("Charger ce lot"):
            # On mappe les 'orig_indices' persistés vers les indices du df filtré courant
            orig_set = set(int(x) for x in picked.get("orig_indices", []))
            orig_to_df = {int(row.orig_idx): i for i, row in df.iterrows()}
            ids = [orig_to_df[oi] for oi in picked.get("orig_indices", []) if oi in orig_to_df]
            if not ids:
                st.warning("Ce lot ne correspond plus aux indices du dataset courant (filtre différent ?).")
            else:
                st.session_state.indices = ids
                st.session_state.idx_ptr = 0
                st.session_state.answers = {}
                st.session_state.choice_shuffle = {}
                st.session_state.custom_subset = True
                st.session_state.current_lot_id = picked.get("id")
                st.toast(f"Lot {picked['id']} chargé", icon="📂")
                st.rerun()
    else:
        st.caption("Aucun lot enregistré pour ce contexte.")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("Utiliser/Créer (5)"):
            pool = list(df.index)  # df déjà filtré par QCM si sélectionné
            if len(pool) == 0:
                st.warning("Aucune question disponible pour ce filtre.")
                st.stop()
            ids = st.session_state.fixed5.get(fixed_key)
            # si absent/cassé → on (re)crée
            if (not ids) or any((i < 0 or i >= len(df)) for i in ids) or len(ids) != min(5, len(pool)):
                ids = random.sample(pool, min(5, len(pool)))
                st.session_state.fixed5[fixed_key] = ids

            # --- Génère un identifiant + enregistre le lot + annote questions ---
            lot_id = _new_lot_id(EXISTING_IDS)
            orig_indices = [int(df.loc[i, "orig_idx"]) for i in ids]
            lot_info = {
                "id": lot_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "school": school,
                "year": year,
                "subject": subject,
                "qcm_filter": selected_qcm if available_qcms else None,
                "orig_indices": orig_indices,   # toujours en indices d'origine fichier
                "size": len(ids),
            }
            LOTS_PAYLOAD.setdefault("lots", []).append(lot_info)
            save_fixed_lots(subject_dir, LOTS_PAYLOAD)
            try:
                annotate_questions_with_lot(QUESTIONS_FILE, orig_indices, lot_id)
                st.toast(f"Lot {lot_id} enregistré et annoté", icon="✅")
            except Exception as e:
                st.warning(f"Lot créé mais annotation échouée : {e}")

            # appliquer le lot à la session (indices du df filtré)
            st.session_state.indices = ids
            st.session_state.idx_ptr = 0
            st.session_state.answers = {}
            st.session_state.choice_shuffle = {}
            st.session_state.custom_subset = True  # empêche _freeze_order de re-mélanger
            st.session_state.current_lot_id = lot_id
            st.rerun()

    with c2:
        if st.button("Changer (5)"):
            pool = list(df.index)
            if len(pool) == 0:
                st.warning("Aucune question disponible pour ce filtre.")
                st.stop()
            prev = st.session_state.fixed5.get(fixed_key, [])
            tries = 0
            while True:
                ids = random.sample(pool, min(5, len(pool)))
                tries += 1
                if set(ids) != set(prev) or tries > 10 or len(pool) < 6:
                    break
            st.session_state.fixed5[fixed_key] = ids

            # nouveau lot persistant
            lot_id = _new_lot_id(EXISTING_IDS)
            orig_indices = [int(df.loc[i, "orig_idx"]) for i in ids]
            lot_info = {
                "id": lot_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "school": school,
                "year": year,
                "subject": subject,
                "qcm_filter": selected_qcm if available_qcms else None,
                "orig_indices": orig_indices,
                "size": len(ids),
            }
            LOTS_PAYLOAD.setdefault("lots", []).append(lot_info)
            save_fixed_lots(subject_dir, LOTS_PAYLOAD)
            try:
                annotate_questions_with_lot(QUESTIONS_FILE, orig_indices, lot_id)
                st.toast(f"Nouveau lot {lot_id} enregistré + annotation OK", icon="✅")
            except Exception as e:
                st.warning(f"Lot créé mais annotation échouée : {e}")

            st.session_state.indices = ids
            st.session_state.idx_ptr = 0
            st.session_state.answers = {}
            st.session_state.choice_shuffle = {}
            st.session_state.custom_subset = True
            st.session_state.current_lot_id = lot_id
            st.rerun()

    with c3:
        if st.button("Désactiver"):
            # enlever le lot figé pour cette clé et revenir au comportement normal
            st.session_state.fixed5.pop(fixed_key, None)
            st.session_state.custom_subset = False
            order = list(df.index)
            if shuffle_q:
                random.shuffle(order)
            st.session_state.indices = order
            st.session_state.idx_ptr = 0
            st.session_state.answers = {}
            st.session_state.choice_shuffle = {}
            st.session_state.current_lot_id = None
            st.rerun()

    # (optionnel) affichage debug du lot courant
    if fixed_key in st.session_state.fixed5:
        st.caption(f"Indices lot courant (session) : {st.session_state.fixed5[fixed_key]}")
    if st.session_state.get("current_lot_id"):
        st.caption(f"Lot figé actif : **{st.session_state.current_lot_id}**")

# ===== Diagnostic =====
diag = validate_answers_and_choices(df)
if diag:
    st.error(f"{len(diag)} problème(s) détecté(s) — corrige le JSON puis relance.")
    for e in diag[:50]:
        st.markdown(f"- **Ligne {e['row_idx']}** — {e['issue']}  \n  {e['detail']}")
    if len(diag) > 50:
        st.caption(f"... et {len(diag)-50} autres.")
    st.stop()

# ===== (Tags retirés de l'UI) =====
with st.sidebar:
    if st.button("Exporter les erreurs en CSV"):
        errors = load_errors(ERRORS_FILE)
        if not errors:
            st.info("Aucune erreur enregistrée.")
        else:
            err_df = pd.DataFrame(errors)
            st.download_button(
                "Télécharger erreurs.csv",
                err_df.to_csv(index=False).encode("utf-8"),
                file_name="erreurs.csv",
                mime="text/csv",
            )

# ===== Ordre figé =====
_freeze_order(df, shuffle_q)
indices = st.session_state.indices

# ===== États =====
if "idx_ptr" not in st.session_state:
    st.session_state.idx_ptr = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}  # row_index -> int | list[int]
if "choice_shuffle" not in st.session_state:
    st.session_state.choice_shuffle = {}  # row_index -> permutation
st.session_state.exam_mode = (mode == "Examen blanc")

if show_timer:
    st.caption("⏱️ Le minuteur est indicatif (ne bloque rien).")
    timer_placeholder = st.empty()

if df.empty:
    st.warning("Aucune question à afficher (filtres trop restrictifs ?).")
    st.stop()

# ========================================================================
#                           AFFICHAGE PRINCIPAL
# ========================================================================
def _score(df_):
    answered = 0
    correct = 0
    for i, row in df_.iterrows():
        if i not in st.session_state.answers:
            continue
        answered += 1
        _, true_ans = _map_true_answer(i, row)
        given = st.session_state.answers.get(i)
        if isinstance(true_ans, list):
            ok = isinstance(given, list) and sorted(given) == sorted(true_ans)
        else:
            ok = (given == true_ans)
        if ok:
            correct += 1
    return correct, answered


current_pos = st.session_state.idx_ptr
total_questions = len(indices) if indices else 0

if current_pos >= total_questions:
    # ===== FIN =====
    correct, answered = _score_on_indices(df, indices)
    total = total_questions
    score_pct = 100 * correct / total if total > 0 else 0.0

    st.success(f"Terminé ✅ — Score: {correct}/{total} ({score_pct:.1f}%)")
    if answered < total:
        st.caption(f"Répondues : {answered}/{total} — {total - answered} non répondues.")

    st.subheader("Récapitulatif")
    presented_order = st.session_state.indices
    order_map = {idx: pos for pos, idx in enumerate(presented_order)}

    recap_rows = []
    answered_set = set(st.session_state.answers.keys())
    iter_indices = [i for i in presented_order if i in answered_set] if answered < total else list(presented_order)

    for i in iter_indices:
        row = df.loc[i]
        choices, true_ans = _map_true_answer(i, row)
        given = st.session_state.answers.get(i, None)

        if isinstance(true_ans, list):
            ok = isinstance(given, list) and sorted(given) == sorted(true_ans)
        else:
            ok = (given == true_ans)

        n = order_map[i] + 1
        badge = "🟩 Correct" if ok else "🟥 Faux"
        st.markdown(f"**Q{n}.** {row['question']}  \n_{badge}_")

        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for j, opt in enumerate(choices):
            ok_correct = (j in true_ans) if isinstance(true_ans, list) else (j == true_ans)
            ok_given = (j in given) if isinstance(given, list) else (j == given)
            prefix = "✅" if ok_correct else ("❌" if ok_given else "•")
            st.markdown(f"{prefix} {opt}")

        if str(row.get("explanation", "") or "").strip():
            with st.expander("Explication"):
                st.markdown(row["explanation"])

        st.divider()

        if not ok:
            recap_rows.append({
                "question": row["question"],
                "choices": "||".join(choices),
                "correct": true_ans,
                "given": given,
                "explanation": str(row.get("explanation", "")),
                "qcm": row.get("qcm", None),
                "theme": row.get("theme", None),
            })

    if recap_rows:
        if st.button("Enregistrer ces erreurs"):
            save_errors(ERRORS_FILE, recap_rows)
            st.toast(f"Erreurs enregistrées → {ERRORS_FILE.name}")

    # === Recommencer (respecte le lot figé / sous-ensemble)
    if st.button("Recommencer"):
        st.session_state.idx_ptr = 0
        st.session_state.answers = {}
        st.session_state.choice_shuffle = {}

        fixed_ids = st.session_state.get("fixed5", {}).get(
            _fixed_session_key(school, year, subject, selected_qcm if available_qcms else None),
            []
        )
        fixed_ids = [i for i in fixed_ids if 0 <= i < len(df)]

        if fixed_ids:
            st.session_state.indices = fixed_ids
            st.session_state.custom_subset = True
            st.rerun()

        if st.session_state.get("custom_subset", False) and st.session_state.get("indices"):
            st.session_state.custom_subset = True
            st.rerun()

        st.session_state.custom_subset = False
        order = list(df.index)
        if st.session_state.shuffle_q:
            random.shuffle(order)
        st.session_state.indices = order
        st.rerun()

else:
    # ===== QUESTION EN COURS =====
    row_idx = indices[current_pos]
    row = df.loc[row_idx]
    st.write(f"**Question {current_pos + 1}/{total_questions}**")

    # permutation des options — mémorisée
    if row_idx not in st.session_state.choice_shuffle:
        perm = list(range(len(row["choices_list"])))
        random.shuffle(perm)
        st.session_state.choice_shuffle[row_idx] = perm
    else:
        perm = st.session_state.choice_shuffle[row_idx]

    options = [row["choices_list"][k] for k in perm]
    map_old_to_new = {old: new for new, old in enumerate(perm)}
    true_ans_orig = row["answer_parsed"]

    # validation indices
    n_opts = len(row["choices_list"])

    def _valid_idx(a: int) -> bool:
        return isinstance(a, int) and 0 <= a < n_opts

    if isinstance(true_ans_orig, list):
        bad = [a for a in true_ans_orig if not _valid_idx(a)]
        if bad:
            st.error(
                f"Indice(s) de réponse invalide(s) {bad} pour :\n\n"
                f"« {row['question']} »\n\n"
                f"Nb d'options = {n_opts} → indices valides: 0..{n_opts-1}.\n"
                f"Corrige 'answer' dans questions.json."
            )
            st.stop()
        shuffled_answer = [map_old_to_new[a] for a in true_ans_orig]
    else:
        if not _valid_idx(true_ans_orig):
            st.error(
                f"Indice de réponse invalide ({true_ans_orig}) pour :\n\n"
                f"« {row['question']} »\n\n"
                f"Nb d'options = {n_opts} → indices valides: 0..{n_opts-1}.\n"
                f"Corrige 'answer' dans questions.json."
            )
            st.stop()
        shuffled_answer = map_old_to_new[true_ans_orig]

    # 🔧 aplatir si une seule bonne réponse après permutation
    if isinstance(shuffled_answer, list) and len(shuffled_answer) == 1:
        shuffled_answer = shuffled_answer[0]

    # Badge multi/unique — vrai multi seulement si >1 réponses
    is_multi = isinstance(shuffled_answer, list) and len(shuffled_answer) > 1
    label = "Choix multiples" if is_multi else "Choix unique"
    color = "#9333EA" if is_multi else "#2563EB"
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            margin: 6px 0 10px 0;
            padding: 6px 14px;
            border-radius: 12px;
            background: {color};
            color: white;
            font-weight: 800;
            font-size: 28px;
            letter-spacing: .2px;">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"### {row['question']}")

    # ---- Affichage de l'image si présente ----
    img_field = str(row.get("image", "") or "").strip()
    if img_field:
        # On essaye plusieurs emplacements plausibles
        rel = img_field.lstrip("/")                           # ex: "Images/2023-q030.png"
        candidates = [
            BASE_DIR / rel,                                   # racine du projet
            BASE_DIR / "public" / rel,                        # si tu as mis public/Images/...
            subject_dir / rel,                                # data/.../Matière/Images/...
            DATA_DIR / rel                                    # data/... à la racine data/
        ]
        img_path = next((p for p in candidates if p.exists()), None)

        if img_path:
            try:
                st.image(str(img_path), caption=None, use_container_width=True)
            except TypeError:
                # compatibilité anciennes versions de Streamlit
                st.image(str(img_path), caption=None, use_column_width=True)

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    st.markdown("\n".join([f"- **{letters[i]}**. {opt}" for i, opt in enumerate(options)]))

    # UI réponses
    if is_multi:
        prev = st.session_state.answers.get(row_idx, [])
        if not isinstance(prev, list):
            prev = []
        checks = []
        for i in range(len(options)):
            checks.append(
                st.checkbox(
                    f"{letters[i]}",
                    value=(i in prev),
                    key=f"q{row_idx}_opt{i}"
                )
            )
        choice_indexes = [i for i, checked in enumerate(checks) if checked]
    else:
        prev = st.session_state.answers.get(row_idx, None)
        if isinstance(prev, int):
            chosen_letter = st.radio(
                "Ta réponse :",
                options=[letters[i] for i in range(len(options))],
                index=prev,
            )
        else:
            chosen_letter = st.radio(
                "Ta réponse :",
                options=[letters[i] for i in range(len(options))],
                index=None,
            )
        choice_indexes = letters.index(chosen_letter) if chosen_letter is not None else None

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Valider"):
            st.session_state.answers[row_idx] = choice_indexes
    with col2:
        if st.button("Question suivante ➜"):
            if choice_indexes is not None:
                st.session_state.answers[row_idx] = choice_indexes
            st.session_state.idx_ptr += 1
            st.rerun()
    with col3:
        if st.button("Terminer maintenant"):
            if choice_indexes is not None:
                st.session_state.answers[row_idx] = choice_indexes
            st.session_state.idx_ptr = len(indices)
            st.rerun()
    with col4:
        if st.button("⟲ Réinitialiser"):
            st.session_state.idx_ptr = 0
            st.session_state.answers = {}
            st.session_state.choice_shuffle = {}

            # 1) Si un lot figé existe pour ce filtre, on repart dessus
            fixed_ids = st.session_state.get("fixed5", {}).get(
                _fixed_session_key(school, year, subject, selected_qcm if available_qcms else None),
                []
            )
            fixed_ids = [i for i in fixed_ids if 0 <= i < len(df)]  # validation

            if fixed_ids:
                st.session_state.indices = fixed_ids
                st.session_state.custom_subset = True
                st.rerun()

            # 2) Sinon, si on était déjà sur un sous-ensemble custom (ex: aléatoire généré)
            if st.session_state.get("custom_subset", False) and st.session_state.get("indices"):
                st.session_state.custom_subset = True
                st.rerun()

            # 3) Fallback: pas de lot figé ni de sous-ensemble → pool complet
            st.session_state.custom_subset = False
            order = list(df.index)
            if st.session_state.shuffle_q:
                random.shuffle(order)
            st.session_state.indices = order
            st.session_state.current_lot_id = None
            st.rerun()

    if show_timer:
        timer_placeholder.write("⏳ Temps indicatif en cours…")

    # Feedback persistant
    if not st.session_state.exam_mode and row_idx in st.session_state.answers:
        given = st.session_state.answers[row_idx]
        if isinstance(shuffled_answer, list):
            ok = isinstance(given, list) and sorted(given) == sorted(shuffled_answer)
            correct_letters = ", ".join(letters[i] for i in shuffled_answer)
        else:
            ok = (given == shuffled_answer)
            correct_letters = letters[shuffled_answer]

        _ = st.success("Bonne réponse ✅") if ok else st.error("Mauvaise réponse ❌")
        st.info(f"Bonne(s) réponse(s) : **{correct_letters}**")
        if str(row.get("explanation", "")).strip():
            st.info(row.get("explanation"))

st.caption(
    "Astuce : LaTeX accepté. Un JSON par matière → data/École/Année/Matière/questions.json. "
    "Champs 'qcm' et 'tags' sont tolérés dans le JSON, mais seuls 'qcm' est filtrable ici. "
    "Les lots figés sont enregistrés dans lots_figes.json et chaque question est annotée via 'id_gen'."
)
