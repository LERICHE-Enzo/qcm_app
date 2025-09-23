import ast
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "Révise tes QCM"
DATA_FILE = Path(__file__).with_name("questions.csv")
ERRORS_FILE = Path(__file__).with_name("erreurs.json")

# --------- Utilitaires ---------
@st.cache_data
def load_questions(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Fichier introuvable : {path}")
        st.stop()

    # 1) Lecture permissive (auto-détection du séparateur) + UTF-8
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

    # 2) Normalisation des noms d'en-têtes (trim, minuscules, retrait BOM)
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]

    # 3) Si tout est dans une seule colonne (souvent CSV FR), retente en ';'
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=";", engine="python", encoding="utf-8")
        df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]

    # 4) Alias tolérants
    alias = {
        "question": "question", "questions": "question",
        "choices": "choices", "propositions": "choices", "choix": "choices",
        "answer": "answer", "reponse": "answer", "réponse": "answer",
        "explanation": "explanation", "explication": "explanation",
        "tags": "tags", "tag": "tags",
    }
    df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)

    # 5) Colonnes requises
    required_cols = {"question", "choices", "answer"}
    if not required_cols.issubset(df.columns):
        st.error(f"Colonnes manquantes. Requis: {required_cols}. Trouvé: {set(df.columns)}")
        st.stop()

    # 6) Dé-quotage léger si nécessaire
    def _dequote(s):
        if pd.isna(s):
            return s
        s = str(s)
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        return s.replace('""', '"')
    for col in ["question", "choices", "explanation", "tags"]:
        if col in df.columns:
            df[col] = df[col].apply(_dequote)

    # 7) Split des choix sur '||'
    df["choices_list"] = df["choices"].astype(str).apply(
        lambda s: [c.strip() for c in s.split("||")]
    )

    # 8) Parse de la colonne answer : entier ou liste [i,j]
    def parse_answer(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                lst = ast.literal_eval(s)
                return [int(v) for v in lst]
            except Exception:
                return None
        try:
            return int(float(s))
        except Exception:
            return None

    df["answer_parsed"] = df["answer"].apply(parse_answer)
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
        path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Impossible d'enregistrer les erreurs : {e}")


# --------- UI ---------
st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="centered")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Paramètres")
    mode = st.selectbox("Mode", ["Entraînement", "Examen blanc", "Révisions ciblées"])
    df = load_questions(DATA_FILE)

    # Tags
    all_tags = sorted({
        t.strip()
        for ts in df.get("tags", pd.Series(dtype=str)).fillna("")
        for t in str(ts).split(",")
        if t.strip()
    })
    selected_tags = st.multiselect("Filtrer par tags (optionnel)", all_tags)

    show_timer = st.toggle("Afficher un minuteur (indicatif)", value=False)
    shuffle_q = st.toggle("Mélanger l'ordre des questions", value=True)

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

# Filtrage par tags
if selected_tags:
    mask = df.get("tags", "").fillna("").apply(
        lambda s: any(t.strip() in [x.strip() for x in str(s).split(",")] for t in selected_tags)
    )
    df = df[mask].reset_index(drop=True)

# --------- Ordre des questions FIGÉ ---------
def _freeze_order(df_: pd.DataFrame, shuffle_flag: bool):
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

_freeze_order(df, shuffle_q)
indices = st.session_state.indices

# --------- État de session ---------
if "idx_ptr" not in st.session_state:
    st.session_state.idx_ptr = 0
if "answers" not in st.session_state:
    # map: row_index -> int | list[int]
    st.session_state.answers = {}
if "choice_shuffle" not in st.session_state:
    # map: row_index -> list of original indices in the NEW order (e.g., [2,0,1,3])
    st.session_state.choice_shuffle = {}
st.session_state.exam_mode = (mode == "Examen blanc")

# --------- Minuteur indicatif ---------
if show_timer:
    st.caption("⏱️ Le minuteur est indicatif (ne bloque rien).")
    timer_placeholder = st.empty()

# Aucune question ?
if df.empty:
    st.warning("Aucune question ne correspond au filtre.")
    st.stop()

# --------- Helpers ---------
def _map_true_answer(row_i: int, row_series: pd.Series):
    """Retourne (choices_shuffled_text, true_ans_in_shuffled_index)"""
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
    """Score basé sur les questions répondues (pour 'Terminer maintenant')."""
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

# --------- Affichage principal ---------
current_pos = st.session_state.idx_ptr

if current_pos >= len(indices):
    # ================= FIN DE SESSION =================
    correct, answered = _compute_score(df)
    # Si fin anticipée, affiche score sur répondues; sinon, sur tout
    total = answered if 0 < answered < len(df) else len(df)
    score_pct = 100 * correct / total if total > 0 else 0.0

    st.success(f"Terminé ✅ — Score: {correct}/{total} ({score_pct:.1f}%)")
    if answered < len(df):
        st.caption(f"Questions répondues : {answered}/{len(df)} — {len(df) - answered} non répondues.")

    # ===== Récapitulatif =====
    st.subheader("Récapitulatif")

    # ordre exactement comme présenté pendant le test
    presented_order = st.session_state.indices
    order_map = {idx: pos for pos, idx in enumerate(presented_order)}  # idx -> position 0-based

    recap_rows = []
    answered_set = set(st.session_state.answers.keys())

    # fin anticipée -> seulement répondues, sinon -> toutes, dans l'ordre présenté
    if answered < len(df):
        iter_indices = [i for i in presented_order if i in answered_set]
        st.caption("Affichage des **questions répondues uniquement** (fin anticipée).")
    else:
        iter_indices = list(presented_order)

    for i in iter_indices:
        row = df.loc[i]
        # Respecte le mélange des options de cette question
        choices, true_ans = _map_true_answer(i, row)
        given = st.session_state.answers.get(i, None)

        # Évaluation
        if isinstance(true_ans, list):
            ok = isinstance(given, list) and sorted(given) == sorted(true_ans)
        else:
            ok = (given == true_ans)
        status = "Correct" if ok else "Faux"

        # En-tête avec numéro vu pendant le test
        n = order_map[i] + 1
        badge = "🟩 Correct" if status == "Correct" else "🟥 Faux"
        st.markdown(f"**Q{n}.** {row['question']}  \n_{badge}_")

        # Marquage des options : ✅ bonne, ❌ choisie fausse, • sinon
        for j, opt in enumerate(choices):
            ok_correct = (j in true_ans) if isinstance(true_ans, list) else (j == true_ans)
            ok_given = (j in given) if isinstance(given, list) else (j == given)
            prefix = "✅" if ok_correct else ("❌" if ok_given else "•")
            st.markdown(f"{prefix} {opt}")

        # Explication
        if str(row.get("explanation", "")).strip():
            with st.expander("Explication"):
                st.markdown(row["explanation"])

        st.divider()

        # Pour export erreurs (uniquement celles non correctes affichées)
        if not ok:
            recap_rows.append({
                "question": row["question"],
                "choices": "||".join(choices),
                "correct": true_ans,
                "given": given,
                "explanation": str(row.get("explanation", "")),
                "tags": str(row.get("tags", "")),
            })

    # Export erreurs + Recommencer
    if recap_rows:
        if st.button("Enregistrer ces erreurs"):
            save_errors(ERRORS_FILE, recap_rows)
            st.toast("Erreurs enregistrées.")

    if st.button("Recommencer"):
        st.session_state.idx_ptr = 0
        st.session_state.answers = {}
        st.session_state.choice_shuffle = {}
        order = list(df.index)
        if shuffle_q:
            random.shuffle(order)
        st.session_state.indices = order
        st.rerun()

else:
    # ================= QUESTION EN COURS =================
    row_idx = indices[current_pos]
    row = df.loc[row_idx]
    st.write(f"**Question {current_pos + 1}/{len(df)}**")

    # --- MÉLANGE DES RÉPONSES (par question, mémorisé en session) ---
    if row_idx not in st.session_state.choice_shuffle:
        shuffle_order = list(range(len(row["choices_list"])))
        random.shuffle(shuffle_order)
        st.session_state.choice_shuffle[row_idx] = shuffle_order
    else:
        shuffle_order = st.session_state.choice_shuffle[row_idx]

    options = [row["choices_list"][k] for k in shuffle_order]
    map_old_to_new = {old: new for new, old in enumerate(shuffle_order)}
    true_ans_orig = row["answer_parsed"]
    if isinstance(true_ans_orig, list):
        shuffled_answer = [map_old_to_new[a] for a in true_ans_orig]
    else:
        shuffled_answer = map_old_to_new[true_ans_orig]

    # --- Badge "Choix unique / Choix multiples" bien visible ---
    is_multi = isinstance(shuffled_answer, list)  # (on le calcule ici)
    label = "Choix multiples" if is_multi else "Choix unique"
    color = "#9333EA" if is_multi else "#2563EB"  # violet pour multi, bleu pour unique
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
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    st.markdown("\n".join([f"- **{letters[i]}**. {opt}" for i, opt in enumerate(options)]))

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
        chosen_letter = st.radio(
            "Ta réponse :",
            options=[letters[i] for i in range(len(options))],
            index=(prev if isinstance(prev, int) else None),
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
            st.session_state.idx_ptr = len(indices)  # saute à l'écran de fin (récap sur répondues)
            st.rerun()

    with col4:
        if st.button("⟲ Réinitialiser"):
            st.session_state.idx_ptr = 0
            st.session_state.answers = {}
            st.session_state.choice_shuffle = {}
            order = list(df.index)
            if shuffle_q:
                random.shuffle(order)
            st.session_state.indices = order
            st.rerun()

    if show_timer:
        timer_placeholder.write("⏳ Temps indicatif en cours…")

    # --------- Feedback persistant (unique)
    if not st.session_state.exam_mode and row_idx in st.session_state.answers:
        given = st.session_state.answers[row_idx]
        ok = False
        if isinstance(shuffled_answer, list):
            ok = isinstance(given, list) and sorted(given) == sorted(shuffled_answer)
            correct_letters = ", ".join(letters[i] for i in shuffled_answer)
        else:
            ok = (given == shuffled_answer)
            correct_letters = letters[shuffled_answer]

        if ok:
            st.success("Bonne réponse ✅")
        else:
            st.error("Mauvaise réponse ❌")

        st.info(f"Bonne(s) réponse(s) : **{correct_letters}**")

        if str(row.get("explanation", "")).strip():
            st.info(row.get("explanation"))

st.caption("Astuce : tu peux coller du LaTeX ($\\LaTeX$). Le séparateur CSV est auto-détecté (',' ou ';'). Les réponses sont mélangées par question et mémorisées.")
