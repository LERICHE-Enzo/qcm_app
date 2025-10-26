import ast
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

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
        # tolérés
        "tags": "tags", "tag": "tags",
        "qcm": "qcm", "theme": "theme", "thème": "theme", "id": "id",
        "id_gen": "id_gen", "image": "image",
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
        return [c.strip() for c in str(x).split("||")]  # ancien format "A||B||C"

    df["choices_list"] = df["choices"].apply(_fix_choices)

    # answer_parsed : int ou [int,...]
    def parse_answer(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, list):
            try:
                lst = [int(v) for v in x]
                # si une seule valeur dans la liste -> on garde la liste (ici
                # on NE contracte PAS en un entier, car dans le mode QCM on
                # veut respecter la multiplicité même si 1 seule bonne réponse)
                return lst
            except Exception:
                return None
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return [int(x)]  # homogénéise en liste
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                lst = ast.literal_eval(s)
                lst = [int(v) for v in lst]
                return lst
            except Exception:
                return None
        try:
            return [int(float(s))]
        except Exception:
            return None

    df["answer_parsed"] = df["answer"].apply(parse_answer)

    for col in ["explanation", "tags", "qcm", "theme", "id", "id_gen", "image"]:
        if col not in df.columns:
            df[col] = None

    return df.reset_index(drop=True)


@st.cache_data
def load_questions_json(path: Path) -> pd.DataFrame:
    """Charge data/.../questions.json (liste ou {'questions': [...]}) en DataFrame normalisée.
       Ajoute 'orig_idx' (index original dans le JSON) pour persister des lots."""
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
    df["orig_idx"] = list(range(len(df)))  # mémorise l'ordre original
    df = _normalize_columns(df)
    return df


@st.cache_data
def load_exercices_json(path: Path) -> List[Dict[str, Any]]:
    """
    Charge exercices.json.

    Format attendu (liste d'exercices), avec compatibilité rétro :
    [
      {
        "titre": "Exercice 1 ...",
        "intro": "...",                  # ancien format, texte uniquement
        "intro_text": "...",             # nouveau format (optionnel)
        "intro_image": "public/...png",  # nouveau format (optionnel)
        "questions": [
          {
            "question": "...",
            "choices": ["A","B","C","D"],
            "answer": 2        # ou [0,2] pour multi-bonne
            "explanation": "...",
            "image": "chemin/vers.png" | null
          },
          ...
        ]
      },
      ...
    ]

    On renvoie une structure nettoyée prête pour l'UI :
    [
      {
        "titre": str,
        "intro_text": str,    # toujours présent (peut être vide)
        "intro_image": str,   # peut être ""
        "questions": [
            {
              "question": str,
              "choices_list": [str,...],
              "answer_parsed": [int,...],    # toujours LISTE d'indices corrects
              "explanation": str,
              "image": str (ou "")
            },
            ...
        ]
      }
    ]
    """
    if not path.exists():
        st.error(f"Fichier introuvable : {path}")
        st.stop()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        st.error(f"JSON invalide dans {path.name} : {e}")
        st.stop()

    if not isinstance(raw, list):
        st.error(f"Format JSON non reconnu dans {path.name}. Attendu une liste d'exercices.")
        st.stop()

    cleaned_exercices = []

    for exo_idx, exo in enumerate(raw):
        titre = str(exo.get("titre", f"Exercice {exo_idx+1}")).strip()

        # compatibilité : si intro_text existe on l'utilise, sinon on fallback sur intro
        intro_text = str(exo.get("intro_text", "") or exo.get("intro", "") or "").strip()
        intro_image = str(exo.get("intro_image", "") or "").strip()

        questions = exo.get("questions", [])
        if not isinstance(questions, list) or len(questions) == 0:
            continue  # on ignore les exos vides

        cleaned_questions = []
        for q_idx, q in enumerate(questions):
            q_text = str(q.get("question", "")).strip()

            raw_choices = q.get("choices", [])
            if not isinstance(raw_choices, list):
                raw_choices = [str(raw_choices)]
            choices_list = [str(c).strip() for c in raw_choices]

            # answer peut être int OU liste
            ans_raw = q.get("answer", None)
            if isinstance(ans_raw, list):
                try:
                    ans_parsed_list = [int(v) for v in ans_raw]
                except Exception:
                    ans_parsed_list = []
            elif ans_raw is None:
                ans_parsed_list = []
            else:
                # single int -> on met dans une liste
                try:
                    ans_parsed_list = [int(ans_raw)]
                except Exception:
                    ans_parsed_list = []

            cleaned_questions.append({
                "question": q_text,
                "choices_list": choices_list,
                "answer_parsed": ans_parsed_list,      # LISTE d'indices corrects
                "explanation": str(q.get("explanation", "") or ""),
                "image": str(q.get("image", "") or ""),
            })

        cleaned_exercices.append({
            "titre": titre,
            "intro_text": intro_text,
            "intro_image": intro_image,
            "questions": cleaned_questions,
        })

    if not cleaned_exercices:
        st.error("Aucun exercice exploitable dans exercices.json.")
        st.stop()

    return cleaned_exercices


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
    label_qcm = selected_qcm if (selected_qcm and selected_qcm != "(Tous)") else "(Tous)"
    return f"{school}|{year}|{subject}|{label_qcm}"


# ===== Diagnostiqueur =====
def validate_answers_and_choices(df_: pd.DataFrame):
    errors = []
    for i, row in df_.iterrows():
        q = str(row.get("question", "")).strip()
        ch_list = row.get("choices_list", [])
        ans_list = row.get("answer_parsed", None)

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

        if ans_list is None:
            errors.append({"row_idx": i, "issue": "answer manquant ou illisible", "detail": ""})
            continue

        if not isinstance(ans_list, list):
            errors.append({"row_idx": i, "issue": "answer_parsed pas liste", "detail": repr(ans_list)})
            continue

        n = len(ch_list)
        out_of_bounds = [a for a in ans_list if not (isinstance(a, int) and 0 <= a < n)]
        if out_of_bounds:
            errors.append({
                "row_idx": i,
                "issue": "indice(s) hors plage dans answer_parsed",
                "detail": f"indices invalides {out_of_bounds} ; nb_options={n} (0..{max(n-1,0)})"
            })

    return errors


def _freeze_order(df_: pd.DataFrame, shuffle_flag: bool):
    # Si on a généré un sous-ensemble custom, on ne touche pas à l'ordre
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
    """
    Pour le mode QCM classique.
    Retourne:
      - choices (ordre mélangé)
      - true_shuf (liste d'indices corrects DANS CET ORDRE AFFICHÉ)
    """
    shuffle_order = st.session_state.choice_shuffle.get(
        row_i, list(range(len(row_series["choices_list"])))
    )
    choices = [row_series["choices_list"][k] for k in shuffle_order]
    map_old_to_new = {old: new for new, old in enumerate(shuffle_order)}

    true_orig_list = row_series["answer_parsed"]  # liste d'indices corrects dans l'ordre original
    true_shuf = [map_old_to_new[a] for a in true_orig_list]
    return choices, true_shuf


def _score_on_indices(df_: pd.DataFrame, indices_: list[int]):
    answered = 0
    correct = 0
    for i in indices_:
        if i not in st.session_state.answers:
            continue
        row = df_.loc[i]
        _, true_ans_list = _map_true_answer(i, row)
        given = st.session_state.answers.get(i)

        # normalise given en liste
        if isinstance(given, list):
            given_list = given[:]
        elif given is None:
            given_list = []
        else:
            given_list = [given]

        ok = sorted(given_list) == sorted(true_ans_list)
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
    - NNN : compteur incrémental pour la date du jour
    - YYYY-MM-DD : date du jour
    """
    today = datetime.now().strftime("%Y-%m-%d")
    prefix = "qcm"
    max_n = 0
    for _id in existing_ids:
        if not _id or not _id.startswith(prefix):
            continue
        parts = _id.split("_")
        if len(parts) != 2:
            continue
        num_part = parts[0][len(prefix):]
        date_part = parts[1]
        if date_part == today and num_part.isdigit():
            max_n = max(max_n, int(num_part))
    next_n = max_n + 1
    return f"{prefix}{next_n:03d}_{today}"


def annotate_questions_with_lot(questions_file: Path, orig_indices: list[int], lot_id: str):
    """
    Ajoute 'id_gen' (liste d'IDs) sur chaque question du lot dans questions.json.
    Les indices fournis sont des indices 'orig_idx'.
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
                        if cur != lot_id:
                            q["id_gen"] = [cur, lot_id]
                        else:
                            q["id_gen"] = [cur]

        new_raw = {"questions": items} if is_wrapped else items
        questions_file.write_text(json.dumps(new_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception as e:
        st.warning(f"Annotation id_gen échouée : {e}")


# ========================================================================
#                 UTILITAIRE : appliquer sous-ensemble
# ========================================================================
def _apply_subset(ids: list[int], *, shuffle: bool = True):
    """Applique un sous-ensemble de questions ; si shuffle=True, l'ordre tourne."""
    order = ids[:]  # copie
    if shuffle:
        random.shuffle(order)
    st.session_state.indices = order
    st.session_state.idx_ptr = 0
    st.session_state.answers = {}
    st.session_state.choice_shuffle = {}
    st.session_state.custom_subset = True  # empêche _freeze_order de re-mélanger


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

    # Type de contenu : QCM global vs Exercices guidés
    content_type = st.radio(
        "Type d'entraînement",
        options=["QCM", "Exercices"],
        horizontal=True,
        key="content_type_radio"
    )

    # Fichiers utilisés selon le mode
    QUESTIONS_FILE = subject_dir / "questions.json"
    ERRORS_FILE = subject_dir / "erreurs.json"
    EXERCICES_FILE = subject_dir / "exercices.json"

    # Paramètres QCM
    st.header("Paramètres")
    mode = st.selectbox("Mode", ["Entraînement", "Examen blanc", "Révisions ciblées"])
    show_timer = st.toggle("Afficher un minuteur (indicatif)", value=False)
    shuffle_q = st.toggle("Mélanger l'ordre des questions (pool complet)", value=True)
    shuffle_fixed_inside = st.toggle("Mélanger l'ordre des questions d'un lot figé", value=True)

    # Si on est en mode Exercices, on prépare tout de suite la liste des exercices
    exercices_sidebar = None
    labels_exos = []
    if st.session_state.get("content_type_radio") == "Exercices":
        exercices_sidebar = load_exercices_json(EXERCICES_FILE)

        labels_exos = [
            f"Exercice {i+1} : {exo['titre']}"
            for i, exo in enumerate(exercices_sidebar)
        ]

        if "exo_id" not in st.session_state:
            st.session_state.exo_id = 0

        picked_label = st.selectbox(
            "Choisir l'exercice",
            options=labels_exos,
            index=min(st.session_state.exo_id, len(labels_exos)-1) if labels_exos else 0,
            key="exo_picker_selectbox"
        )

        new_exo_id = labels_exos.index(picked_label) if labels_exos else 0
        if new_exo_id != st.session_state.exo_id:
            st.session_state.exo_id = new_exo_id
            st.session_state.step_id = 0  # on repart au début de l'exercice choisi


# ===================== MODE EXERCICES GUIDÉS =====================
if st.session_state.get("content_type_radio") == "Exercices":
    exercices = exercices_sidebar if exercices_sidebar is not None else load_exercices_json(EXERCICES_FILE)

    # --- états / progression ---
    if "exo_id" not in st.session_state:
        st.session_state.exo_id = 0
    if "step_id" not in st.session_state:
        st.session_state.step_id = 0
    if "exo_answers" not in st.session_state:
        st.session_state.exo_answers = {}       # {(exo_id, step_id): réponse utilisateur}
    if "exo_choice_shuffle" not in st.session_state:
        st.session_state.exo_choice_shuffle = {}  # {(exo_id, step_id): permutation affichée}

    # Sécurité bornes
    st.session_state.exo_id = max(0, min(st.session_state.exo_id, len(exercices)-1))
    current_exo = exercices[st.session_state.exo_id]
    questions_list = current_exo["questions"]
    st.session_state.step_id = max(0, min(st.session_state.step_id, len(questions_list)-1))

    # question courante
    qkey = (st.session_state.exo_id, st.session_state.step_id)
    qrow = questions_list[st.session_state.step_id]

    # Mélange des choix pour CETTE question
    if qkey not in st.session_state.exo_choice_shuffle:
        perm = list(range(len(qrow["choices_list"])))
        random.shuffle(perm)
        st.session_state.exo_choice_shuffle[qkey] = perm
    perm = st.session_state.exo_choice_shuffle[qkey]

    # options affichées dans l'ordre mélangé
    options = [qrow["choices_list"][i] for i in perm]

    # vraie/bonnes réponses après mélange
    map_old_to_new = {old: new for new, old in enumerate(perm)}
    true_answers_list = [map_old_to_new[a] for a in qrow["answer_parsed"]]  # -> liste d'indices corrects dans l'affichage

    is_multi = len(true_answers_list) > 1

    # ===================== RENDER PAGE EXERCICE =====================
    st.header(f"📝 {current_exo['titre']}")
    st.caption(
        f"Question {st.session_state.step_id + 1}/{len(questions_list)} "
        f"(Exercice {st.session_state.exo_id + 1})"
    )

    # ----- INTRO COMMUNE (texte + image éventuelle) -----
    intro_text = str(current_exo.get("intro_text", "") or current_exo.get("intro", "") or "").strip()
    intro_image = str(current_exo.get("intro_image", "") or "").strip()

    if intro_text or intro_image:
        with st.expander("Contexte / Données de l'exercice", expanded=True):
            if intro_text:
                st.markdown(intro_text)
            if intro_image:
                # on va chercher l'image à plusieurs emplacements possibles
                rel = intro_image.lstrip("/")
                candidates = [
                    BASE_DIR / rel,
                    BASE_DIR / "public" / rel,
                    subject_dir / rel,
                    DATA_DIR / rel,
                ]
                img_path = next((p for p in candidates if p.exists()), None)
                if img_path:
                    try:
                        st.image(str(img_path), use_container_width=True)
                    except TypeError:
                        st.image(str(img_path), use_column_width=True)
                else:
                    st.warning(f"Image introuvable : {intro_image}")

    # badge "Choix unique / Choix multiples"
    badge_label = "Choix multiples" if is_multi else "Choix unique"
    badge_color = "#9333EA" if is_multi else "#2563EB"
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            margin: 12px 0 16px 0;
            padding: 6px 14px;
            border-radius: 12px;
            background: {badge_color};
            color: white;
            font-weight: 800;
            font-size: 28px;
            letter-spacing: .2px;">
            {badge_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ÉNONCÉ DE LA QUESTION COURANTE
    st.markdown(f"### {qrow['question']}")

    # Image spécifique à CETTE question
    img_field = str(qrow.get("image", "") or "").strip()
    if img_field:
        rel = img_field.lstrip("/")
        candidates = [
            BASE_DIR / rel,
            BASE_DIR / "public" / rel,
            subject_dir / rel,
            DATA_DIR / rel
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path:
            try:
                st.image(str(img_path), use_container_width=True)
            except TypeError:
                st.image(str(img_path), use_column_width=True)

    # Affichage style "A. ...", "B. ..."
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    st.markdown("\n".join([f"- **{letters[i]}**. {opt}" for i, opt in enumerate(options)]))

    # réponse précédente si on est revenu en arrière
    prev_answer_user = st.session_state.exo_answers.get(qkey, None)

    # =====================
    # UI de réponse
    # =====================
    if is_multi:
        # convertir prev en liste (sinon liste vide)
        if not isinstance(prev_answer_user, list):
            prev_answer_user = []

        checks = []
        for i in range(len(options)):
            checks.append(
                st.checkbox(
                    f"{letters[i]}",
                    value=(i in prev_answer_user),
                    key=f"exo_{qkey}_chk_{i}"
                )
            )
        user_choice_processed = [i for i, checked in enumerate(checks) if checked]

    else:
        # choix unique -> radio
        opts_radio = ["—"] + [letters[i] for i in range(len(options))]
        if isinstance(prev_answer_user, list):
            # si c'est une liste genre [2], extraire
            prev_idx_scalar = prev_answer_user[0] if prev_answer_user else None
        else:
            prev_idx_scalar = prev_answer_user

        idx_radio = (prev_idx_scalar + 1) if isinstance(prev_idx_scalar, int) else 0
        sel = st.radio(
            "Ta réponse :",
            options=opts_radio,
            index=idx_radio,
            key=f"exo_{qkey}_radio",
            horizontal=False
        )
        if sel == "—":
            user_choice_processed = []
        else:
            user_choice_processed = [letters.index(sel)]

    # ====== Boutons ======
    col1, col2, col3, col4 = st.columns([1,1,1,1])

    with col1:
        if st.button("⬅ Question précédente"):
            st.session_state.exo_answers[qkey] = user_choice_processed
            st.session_state.step_id -= 1
            if st.session_state.step_id < 0:
                st.session_state.step_id = 0
            st.rerun()

    with col2:
        if st.button("Valider"):
            st.session_state.exo_answers[qkey] = user_choice_processed

    with col3:
        if st.button("Question suivante ➜"):
            st.session_state.exo_answers[qkey] = user_choice_processed
            st.session_state.step_id += 1
            if st.session_state.step_id >= len(questions_list):
                st.session_state.step_id = len(questions_list) - 1
            st.rerun()

    with col4:
        if st.button("Revenir au début de cet exercice"):
            st.session_state.exo_answers[qkey] = user_choice_processed
            st.session_state.step_id = 0
            st.rerun()

    # ====== Feedback immédiat si répondu ======
    if qkey in st.session_state.exo_answers:
        given_list = st.session_state.exo_answers[qkey]
        if not isinstance(given_list, list):
            given_list = [] if given_list is None else [given_list]

        ok = sorted(given_list) == sorted(true_answers_list)

        _ = st.success("Bonne réponse ✅") if ok else st.error("Mauvaise réponse ❌")

        correct_letters = ", ".join(letters[i] for i in sorted(true_answers_list))
        st.info(f"Bonne(s) réponse(s) : **{correct_letters}**")

        if str(qrow.get("explanation", "")).strip():
            st.info(qrow["explanation"])

    # ====== Score courant de l'exercice sélectionné ======
    correct_count = 0
    total_q = len(questions_list)
    answered = 0

    for local_idx, qqq in enumerate(questions_list):
        local_key = (st.session_state.exo_id, local_idx)
        if local_key not in st.session_state.exo_answers:
            continue

        prev_user = st.session_state.exo_answers[local_key]
        if not isinstance(prev_user, list):
            prev_user = [] if prev_user is None else [prev_user]

        perm_loc = st.session_state.exo_choice_shuffle.get(
            local_key,
            list(range(len(qqq["choices_list"])))
        )
        map_old_new_loc = {old: new for new, old in enumerate(perm_loc)}
        true_loc = [map_old_new_loc[a] for a in qqq["answer_parsed"]]

        if sorted(prev_user) == sorted(true_loc):
            correct_count += 1

        answered += 1

    pct = 100 * correct_count / total_q if total_q else 0.0
    st.markdown(
        f"**Score sur \"{current_exo['titre']}\" : {correct_count}/{total_q} "
        f"({pct:.1f}%) — répondu à {answered}/{total_q} questions.**"
    )

    st.caption(
        "Mode exercices guidés :\n"
        "- Choisis l'exercice dans la barre latérale.\n"
        "- Les questions restent dans l'ordre du fichier exercices.json.\n"
        "- Les propositions de réponse sont mélangées.\n"
        "- 'Choix multiple' = plusieurs cases vraies possibles.\n"
        "- Le bloc 'Contexte / Données' peut inclure du texte (intro_text / intro) et/ou une image (intro_image)."
    )

    st.stop()


# ===================== MODE QCM CLASSIQUE =====================

# ===== Chargement du JSON unique =====
df = load_questions_json(QUESTIONS_FILE)

# Filtre QCM (si champ 'qcm' existe)
available_qcms = sorted({q for q in df.get("qcm", pd.Series()).dropna().unique()})
if available_qcms:
    with st.sidebar:
        selected_qcm = st.selectbox("Filtrer par QCM", ["(Tous)"] + available_qcms)

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
        df = df[df["qcm"] == selected_qcm].reset_index(drop=True)
else:
    selected_qcm = None

# ===== QCM aléatoire + Lot figé (5) =====
with st.sidebar:
    st.header("🎯 QCM aléatoire")
    k = st.number_input("Nombre de questions", min_value=1, max_value=50, value=5, step=1)
    if st.button("Générer ce QCM"):
        pop_indices = list(df.index)
        if len(pop_indices) == 0:
            st.warning("Aucune question disponible pour ce filtre.")
            st.stop()
        k = min(int(k), len(pop_indices))
        order = random.sample(pop_indices, k)

        _apply_subset(order, shuffle=True)  # toujours mélangé
        st.session_state.current_lot_id = None
        st.rerun()

    # ===== Lot figé (5) PERSISTANT =====
    st.header("🔒 Lot figé (5)")

    if "fixed5" not in st.session_state:
        st.session_state.fixed5 = {}  # dict: key -> [indices]

    fixed_key = _fixed_session_key(
        school, year, subject, selected_qcm if available_qcms else None
    )

    LOTS_PAYLOAD = load_fixed_lots(subject_dir)
    EXISTING_IDS = {lot.get("id") for lot in LOTS_PAYLOAD.get("lots", []) if lot.get("id")}

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
            orig_to_df = {int(row.orig_idx): i for i, row in df.iterrows()}
            ids = [orig_to_df[oi] for oi in picked.get("orig_indices", []) if oi in orig_to_df]
            if not ids:
                st.warning("Ce lot ne correspond plus aux indices du dataset courant (filtre différent ?).")
            else:
                _apply_subset(ids, shuffle=shuffle_fixed_inside)  # <<< ordre tourne
                st.session_state.current_lot_id = picked.get("id")
                st.toast(f"Lot {picked['id']} chargé", icon="📂")
                st.rerun()
    else:
        st.caption("Aucun lot enregistré pour ce contexte.")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("Utiliser/Créer (5)"):
            pool = list(df.index)
            if len(pool) == 0:
                st.warning("Aucune question disponible pour ce filtre.")
                st.stop()
            ids = st.session_state.fixed5.get(fixed_key)
            if (not ids) or any((i < 0 or i >= len(df)) for i in ids) or len(ids) != min(5, len(pool)):
                ids = random.sample(pool, min(5, len(pool)))
                st.session_state.fixed5[fixed_key] = ids

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
                st.toast(f"Lot {lot_id} enregistré et annoté", icon="✅")
            except Exception as e:
                st.warning(f"Lot créé mais annotation échouée : {e}")

            _apply_subset(ids, shuffle=shuffle_fixed_inside)  # <<< ordre tourne
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

            _apply_subset(ids, shuffle=shuffle_fixed_inside)  # <<< ordre tourne
            st.session_state.current_lot_id = lot_id
            st.rerun()

    with c3:
        if st.button("Désactiver"):
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

# ===== Export erreurs =====
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

# ===== Ordre figé (hors sous-ensembles) =====
_freeze_order(df, shuffle_q)
indices = st.session_state.indices

# ===== États session QCM =====
if "idx_ptr" not in st.session_state:
    st.session_state.idx_ptr = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "choice_shuffle" not in st.session_state:
    st.session_state.choice_shuffle = {}
st.session_state.exam_mode = (mode == "Examen blanc")

if show_timer:
    st.caption("⏱️ Le minuteur est indicatif (ne bloque rien).")
    timer_placeholder = st.empty()

if df.empty:
    st.warning("Aucune question à afficher (filtres trop restrictifs ?).")
    st.stop()

# ========================================================================
#                           AFFICHAGE PRINCIPAL QCM
# ========================================================================
current_pos = st.session_state.idx_ptr
total_questions = len(indices) if indices else 0

def _current_qcm_question_block():
    row_idx = indices[current_pos]
    row = df.loc[row_idx]
    # permutation des options — mémorisée
    if row_idx not in st.session_state.choice_shuffle:
        perm_local = list(range(len(row["choices_list"])))
        random.shuffle(perm_local)
        st.session_state.choice_shuffle[row_idx] = perm_local
    else:
        perm_local = st.session_state.choice_shuffle[row_idx]

    options_local = [row["choices_list"][k] for k in perm_local]
    map_old_to_new_local = {old: new for new, old in enumerate(perm_local)}

    true_ans_orig_list = row["answer_parsed"]  # toujours liste d'indices corrects
    # validation indices
    n_opts = len(row["choices_list"])

    bad = [a for a in true_ans_orig_list if not (isinstance(a, int) and 0 <= a < n_opts)]
    if bad:
        st.error(
            f"Indice(s) de réponse invalide(s) {bad} pour :\n\n"
            f"« {row['question']} »\n\n"
            f"Nb d'options = {n_opts} → indices valides: 0..{n_opts-1}.\n"
            f"Corrige 'answer' dans questions.json."
        )
        st.stop()

    shuffled_answer_list = [map_old_to_new_local[a] for a in true_ans_orig_list]

    is_multi_here = len(shuffled_answer_list) > 1
    label_here = "Choix multiples" if is_multi_here else "Choix unique"
    color_here = "#9333EA" if is_multi_here else "#2563EB"

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            margin: 6px 0 10px 0;
            padding: 6px 14px;
            border-radius: 12px;
            background: {color_here};
            color: white;
            font-weight: 800;
            font-size: 28px;
            letter-spacing: .2px;">
            {label_here}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"### {row['question']}")

    # ---- Affichage image si présente ----
    img_field_local = str(row.get("image", "") or "").strip()
    if img_field_local:
        rel_local = img_field_local.lstrip("/")
        candidates_local = [
            BASE_DIR / rel_local,
            BASE_DIR / "public" / rel_local,
            subject_dir / rel_local,
            DATA_DIR / rel_local
        ]
        img_path_local = next((p for p in candidates_local if p.exists()), None)
        if img_path_local:
            try:
                st.image(str(img_path_local), caption=None, use_container_width=True)
            except TypeError:
                st.image(str(img_path_local), caption=None, use_column_width=True)

    letters_local = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    st.markdown("\n".join([f"- **{letters_local[i]}**. {opt}" for i, opt in enumerate(options_local)]))

    # UI réponses
    prev_given = st.session_state.answers.get(row_idx, [])
    if not isinstance(prev_given, list):
        # si une seule réponse historique => mettre dans liste
        prev_given = [] if prev_given is None else [prev_given]

    if is_multi_here:
        checks_local = []
        for i in range(len(options_local)):
            checks_local.append(
                st.checkbox(
                    f"{letters_local[i]}",
                    value=(i in prev_given),
                    key=f"q{row_idx}_opt{i}"
                )
            )
        choice_indexes_local = [i for i, checked in enumerate(checks_local) if checked]
    else:
        opts_local = ["—"] + [letters_local[i] for i in range(len(options_local))]
        prev_scalar = prev_given[0] if prev_given else None
        idx_local = (prev_scalar + 1) if isinstance(prev_scalar, int) else 0
        sel_local = st.radio(
            "Ta réponse :",
            options=opts_local,
            index=idx_local,
            key=f"q{row_idx}_radio",
            horizontal=False
        )
        choice_indexes_local = [] if sel_local == "—" else [letters_local.index(sel_local)]

    col1a, col2a, col3a, col4a = st.columns([1, 1, 1, 1])
    with col1a:
        if st.button("Valider"):
            st.session_state.answers[row_idx] = choice_indexes_local
    with col2a:
        if st.button("Question suivante ➜"):
            st.session_state.answers[row_idx] = choice_indexes_local
            st.session_state.idx_ptr += 1
            st.rerun()
    with col3a:
        if st.button("Terminer maintenant"):
            st.session_state.answers[row_idx] = choice_indexes_local
            st.session_state.idx_ptr = len(indices)
            st.rerun()
    with col4a:
        if st.button("⏲ Réinitialiser"):
            st.session_state.idx_ptr = 0
            st.session_state.answers = {}
            st.session_state.choice_shuffle = {}

            fixed_ids_local = st.session_state.get("fixed5", {}).get(
                _fixed_session_key(school, year, subject, selected_qcm if available_qcms else None),
                []
            )
            fixed_ids_local = [i for i in fixed_ids_local if 0 <= i < len(df)]

            if fixed_ids_local:
                _apply_subset(fixed_ids_local, shuffle=shuffle_fixed_inside)
                st.rerun()

            if st.session_state.get("custom_subset", False) and st.session_state.get("indices"):
                _apply_subset(st.session_state.indices, shuffle=True)
                st.rerun()

            st.session_state.custom_subset = False
            order_local = list(df.index)
            if st.session_state.shuffle_q:
                random.shuffle(order_local)
            st.session_state.indices = order_local
            st.session_state.current_lot_id = None
            st.rerun()

    if show_timer:
        timer_placeholder.write("⏳ Temps indicatif en cours…")

    # Feedback persistant (si pas exam blanc)
    if not st.session_state.exam_mode and row_idx in st.session_state.answers:
        given_now = st.session_state.answers[row_idx]
        if not isinstance(given_now, list):
            given_now = [] if given_now is None else [given_now]

        ok_here = sorted(given_now) == sorted(shuffled_answer_list)
        _ = st.success("Bonne réponse ✅") if ok_here else st.error("Mauvaise réponse ❌")

        correct_letters_local = ", ".join(
            letters_local[i] for i in sorted(shuffled_answer_list)
        )
        st.info(f"Bonne(s) réponse(s) : **{correct_letters_local}**")
        if str(row.get("explanation", "")).strip():
            st.info(row.get("explanation"))

    return shuffled_answer_list


if current_pos >= total_questions:
    # ===== FIN DU QCM =====
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

        # récupérer l'ordre d'affichage utilisé pour CETTE question
        perm_disp = st.session_state.choice_shuffle.get(
            i, list(range(len(row["choices_list"])))
        )
        disp_choices = [row["choices_list"][k] for k in perm_disp]
        map_old_to_new_disp = {old: new for new, old in enumerate(perm_disp)}
        true_disp_list = [map_old_to_new_disp[a] for a in row["answer_parsed"]]

        given_disp = st.session_state.answers.get(i, [])
        if not isinstance(given_disp, list):
            given_disp = [] if given_disp is None else [given_disp]

        ok = sorted(given_disp) == sorted(true_disp_list)

        n = order_map[i] + 1
        badge = "🟩 Correct" if ok else "🟥 Faux"
        st.markdown(f"**Q{n}.** {row['question']}  \n_{badge}_")

        letters_g = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for j, opt in enumerate(disp_choices):
            ok_correct = (j in true_disp_list)
            ok_given = (j in given_disp)
            prefix = "✅" if ok_correct else ("❌" if ok_given else "•")
            st.markdown(f"{prefix} {opt}")

        if str(row.get("explanation", "") or "").strip():
            with st.expander("Explication"):
                st.markdown(row["explanation"])

        st.divider()

        if not ok:
            recap_rows.append({
                "question": row["question"],
                "choices": "||".join(disp_choices),
                "correct": true_disp_list,
                "given": given_disp,
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
            _apply_subset(fixed_ids, shuffle=shuffle_fixed_inside)
            st.rerun()

        if st.session_state.get("custom_subset", False) and st.session_state.get("indices"):
            _apply_subset(st.session_state.indices, shuffle=True)
            st.rerun()

        st.session_state.custom_subset = False
        order_reset = list(df.index)
        if st.session_state.shuffle_q:
            random.shuffle(order_reset)
        st.session_state.indices = order_reset
        st.rerun()

else:
    st.write(f"**Question {current_pos + 1}/{total_questions}**")
    _current_qcm_question_block()

st.caption(
    "Astuce : LaTeX accepté. Un JSON par matière → data/École/Année/Matière/questions.json. "
    "Seul 'qcm' est filtrable. Les lots figés sont dans lots_figes.json.\n\n"
    "Mode Exercices guidés : chaque exercice dans exercices.json peut avoir soit "
    "'intro' (ancien format texte uniquement), soit le duo 'intro_text' + 'intro_image' "
    "pour afficher un bloc de contexte commun (avec éventuelle image) en haut de TOUTES "
    "les questions de l'exercice. Les questions peuvent avoir une ou plusieurs bonnes réponses."
)
