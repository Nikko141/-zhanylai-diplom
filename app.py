import json, numpy as np, pandas as pd, plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

st.set_page_config(page_title="Кыргыз тили — Dataset Explorer", page_icon="🇰🇬", layout="wide")
st.title("🇰🇬 Кыргыз тили — Dataset Explorer")
st.caption("Антонимы + диалоги · Sentence Transformers (MiniLM) · Векторный анализ")

@st.cache_data
def load_data():
    with open(Path(__file__).parent / "dataset.json", encoding="utf-8") as f:
        return json.load(f)

raw = load_data()
antonyms  = raw["antonyms"]
dialogues = raw["dialogues"]

@st.cache_resource(show_spinner="Загружаю модель...")
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        mdl = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return mdl, "minilm"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Фитируем на ВСЁМ корпусе сразу — чтобы vocab одинаковый для KY и RU
        with open(Path(__file__).parent / "dataset.json", encoding="utf-8") as _f:
            _d = json.load(_f)
        all_texts = []
        for a in _d["antonyms"]:
            all_texts += [a["kyrgyz"]["pair"], a["kyrgyz"]["word1"], a["kyrgyz"]["word2"],
                          a["kyrgyz"]["example"], a["russian"]["pair"],
                          a["russian"]["word1"], a["russian"]["word2"], a["russian"]["example"]]
        for dlg in _d["dialogues"]:
            for lang in ["kyrgyz", "english", "russian"]:
                all_texts += [ln["text"] for ln in dlg["lines"].get(lang, [])]
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf.fit([t for t in all_texts if t])
        return tfidf, "tfidf"

model_obj, backend = load_model()
if backend == "tfidf":
    st.warning("torch не найден — используется TF-IDF char n-gram (fallback). "
               "Запустите `pip install -r requirements.txt` для полного MiniLM.")
else:
    st.success("MiniLM загружен — полный семантический анализ доступен.")

def _l2(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms

@st.cache_data
def encode(texts):
    t = list(texts)
    if backend == "minilm":
        return np.array(model_obj.encode(t, show_progress_bar=False, normalize_embeddings=True))
    # TF-IDF: transform (не fit_transform) — vocab уже зафиксирован
    return _l2(model_obj.transform(t).toarray().astype(float))

LC = {"kyrgyz":"#4FC3F7","english":"#A5D6A7","russian":"#EF9A9A",
      "Кыргызча":"#4FC3F7","Орусча":"#EF9A9A"}
LL = {"kyrgyz":"🇰🇬 Кыргызча","english":"🇬🇧 English","russian":"🇷🇺 Орусча"}

tab1, tab2, tab3, tab4 = st.tabs(["📚 Антонимы","💬 Диалоги","📐 Векторный анализ","📊 Статистика"])

# TAB1 ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Антонимдер / Антонимы")
    rows = [{"№":a["id"],"Кыргызча (пара)":a["kyrgyz"]["pair"],
             "Слово 1 (KY)":a["kyrgyz"]["word1"],"Слово 2 (KY)":a["kyrgyz"]["word2"],
             "Орусча (пара)":a["russian"]["pair"],
             "Слово 1 (RU)":a["russian"]["word1"],"Слово 2 (RU)":a["russian"]["word2"],
             "Мисал (KY)":a["kyrgyz"]["example"],"Пример (RU)":a["russian"]["example"]}
            for a in antonyms]
    df = pd.DataFrame(rows)
    if not st.checkbox("Показать примеры", value=True):
        df = df.drop(columns=["Мисал (KY)","Пример (RU)"])
    st.dataframe(df.set_index("№"), use_container_width=True, height=520)
    with st.expander("Скачать CSV"):
        st.download_button("Скачать", df.to_csv(index=False, encoding="utf-8-sig"),
                           "antonyms.csv", "text/csv")

# TAB2 ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Диалогдор / Диалоги / Dialogues")
    c1, c2 = st.columns([3,1])
    opts = {f"Диалог {d['id']}: {d['title']['kyrgyz'] or d['title']['russian']}": d for d in dialogues}
    with c1:
        sel = st.selectbox("Выберите диалог", list(opts.keys()))
    with c2:
        langs = st.multiselect("Языки", ["kyrgyz","english","russian"],
                               default=["kyrgyz","english","russian"])
    d = opts[sel]
    st.markdown(f"**Тема:** {d['title']['kyrgyz']} / {d['title']['english']} / {d['title']['russian']}")
    max_n = max((len(d["lines"].get(l,[])) for l in langs), default=0)
    rows_d = []
    for i in range(max_n):
        row = {"Реплика": i+1}
        for lang in langs:
            ll = d["lines"].get(lang, [])
            row[LL[lang]] = f"**{ll[i]['speaker']}:** {ll[i]['text']}" if i < len(ll) else ""
        rows_d.append(row)
    st.dataframe(pd.DataFrame(rows_d).set_index("Реплика"), use_container_width=True, height=420)
    st.markdown("---")
    st.markdown("#### Все диалоги")
    ov = [{"ID":d["id"],"Тема KY":d["title"]["kyrgyz"],"Тема EN":d["title"]["english"],
           "Тема RU":d["title"]["russian"],
           "Реплик":max(len(d["lines"].get(l,[])) for l in ["kyrgyz","english","russian"])}
          for d in dialogues]
    st.dataframe(pd.DataFrame(ov).set_index("ID"), use_container_width=True, height=360)

# TAB3 ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Векторный анализ — Sentence Transformers (MiniLM)")
    mode = st.radio("Что анализируем?", ["Антонимы (пары слов)","Диалоги (реплики)"], horizontal=True)

    if mode == "Антонимы (пары слов)":
        @st.cache_data
        def ant_emb():
            txts, lbls, lngs = [], [], []
            for a in antonyms:
                for w, lg in [(a["kyrgyz"]["word1"],"Кыргызча"),(a["kyrgyz"]["word2"],"Кыргызча"),
                              (a["russian"]["word1"],"Орусча"),(a["russian"]["word2"],"Орусча")]:
                    txts.append(w); lbls.append(f"{'KY' if lg=='Кыргызча' else 'RU'}: {w}"); lngs.append(lg)
            return encode(txts), lbls, lngs, txts

        embs, lbls, lngs, txts = ant_emb()
        coords = PCA(2, random_state=42).fit_transform(embs)
        fig1 = px.scatter(pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],
                                        "Язык":lngs,"Метка":lbls,"Слово":txts}),
                          x="PC1", y="PC2", color="Язык", hover_data=["Метка","Слово"],
                          title="PCA — слова-антонимы в векторном пространстве",
                          template="plotly_dark", color_discrete_map=LC)
        fig1.update_traces(marker=dict(size=11, opacity=0.88))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Косинусная близость: кыргызская пара vs русский перевод")

        @st.cache_data
        def pair_sims():
            ky = encode([a["kyrgyz"]["pair"] for a in antonyms])
            ru = encode([a["russian"]["pair"] for a in antonyms])
            return ([float(cosine_similarity([ky[i]],[ru[i]])[0][0]) for i in range(len(antonyms))],
                    [a["kyrgyz"]["pair"] for a in antonyms])

        sims, plbls = pair_sims()
        df_s = pd.DataFrame({"Пара":plbls,"Сходство":sims}).sort_values("Сходство")
        fig2 = px.bar(df_s, x="Сходство", y="Пара", orientation="h",
                      color="Сходство", color_continuous_scale="RdYlGn", range_color=[0.3,1.0],
                      title="KY vs RU: семантическое сходство пар антонимов", template="plotly_dark")
        fig2.update_layout(height=540, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
        ca,cb,cc = st.columns(3)
        ca.metric("Среднее", f"{np.mean(sims):.3f}")
        cb.metric("Мин", f"{np.min(sims):.3f}")
        cc.metric("Макс", f"{np.max(sims):.3f}")

        st.markdown("---")
        st.markdown("#### Матрица сходств: все KY × все RU пары")

        @st.cache_data
        def full_mat():
            ky = encode([a["kyrgyz"]["pair"] for a in antonyms])
            ru = encode([a["russian"]["pair"] for a in antonyms])
            return cosine_similarity(ky, ru)

        mat = full_mat()
        fig3 = px.imshow(mat, x=[a["russian"]["pair"] for a in antonyms],
                         y=[a["kyrgyz"]["pair"] for a in antonyms],
                         color_continuous_scale="Blues",
                         title="Матрица косинусных сходств KY x RU",
                         template="plotly_dark", zmin=0, zmax=1)
        fig3.update_layout(height=560, xaxis_tickangle=-35)
        st.plotly_chart(fig3, use_container_width=True)

    else:
        sel_id = st.selectbox("Диалог", [d["id"] for d in dialogues],
            format_func=lambda x: f"Диалог {x} — " +
                next(d["title"]["kyrgyz"] or d["title"]["russian"] for d in dialogues if d["id"]==x))

        @st.cache_data
        def dial_emb(did):
            d = next(d for d in dialogues if d["id"]==did)
            txts, lngs, spks = [], [], []
            for lang in ["kyrgyz","english","russian"]:
                for ln in d["lines"].get(lang,[]):
                    txts.append(ln["text"]); lngs.append(lang); spks.append(ln["speaker"])
            if not txts:
                return None, None, None, None
            return encode(txts), txts, lngs, spks

        embs_d, txts_d, lngs_d, spks_d = dial_emb(sel_id)
        if embs_d is not None and len(embs_d) >= 3:
            coords2 = PCA(2, random_state=42).fit_transform(embs_d)
            fig4 = px.scatter(pd.DataFrame({"PC1":coords2[:,0],"PC2":coords2[:,1],
                                            "Язык":lngs_d,"Спикер":spks_d,"Реплика":txts_d}),
                              x="PC1", y="PC2", color="Язык", symbol="Спикер",
                              hover_data=["Реплика"], title=f"Диалог {sel_id} — PCA реплик",
                              template="plotly_dark", color_discrete_map=LC)
            fig4.update_traces(marker=dict(size=13))
            st.plotly_chart(fig4, use_container_width=True)

            sim_d = cosine_similarity(embs_d)
            short = [f"{l[:2].upper()}-{s}" for l,s in zip(lngs_d, spks_d)]
            fig5 = px.imshow(sim_d, x=short, y=short, color_continuous_scale="Blues",
                             title="Матрица косинусных сходств реплик",
                             template="plotly_dark", zmin=0, zmax=1)
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)

            st.markdown("#### Среднее сходство между языками")
            lp = []
            for l1, l2 in [("kyrgyz","english"),("kyrgyz","russian"),("english","russian")]:
                i1 = [i for i,l in enumerate(lngs_d) if l==l1]
                i2 = [i for i,l in enumerate(lngs_d) if l==l2]
                if i1 and i2:
                    lp.append({"Пара":f"{l1[:2].upper()} vs {l2[:2].upper()}",
                               "Сходство":float(sim_d[np.ix_(i1,i2)].mean())})
            if lp:
                fig6 = px.bar(pd.DataFrame(lp), x="Пара", y="Сходство",
                              color="Сходство", color_continuous_scale="Teal", range_color=[0,1],
                              template="plotly_dark", title="Среднее сходство между языками")
                st.plotly_chart(fig6, use_container_width=True)
        else:
            st.warning("Недостаточно реплик для PCA.")

# TAB4 ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Статистика датасета")
    total = sum(max(len(d["lines"].get(l,[])) for l in ["kyrgyz","english","russian"]) for d in dialogues)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Антонимов", len(antonyms))
    c2.metric("Диалогов", len(dialogues))
    c3.metric("Реплик всего", total)
    c4.metric("Реплик / диалог", f"{total/len(dialogues):.1f}")

    st.markdown("---")
    st.markdown("#### Длина реплик по языкам (символов)")
    len_rows = [{"Язык":lang,"Длина":len(ln["text"])}
                for d in dialogues for lang in ["kyrgyz","english","russian"]
                for ln in d["lines"].get(lang,[])]
    fig7 = px.box(pd.DataFrame(len_rows), x="Язык", y="Длина", color="Язык",
                  color_discrete_map=LC, title="Распределение длины реплик",
                  template="plotly_dark", points="all")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("#### Количество реплик в каждом диалоге")
    cnt = [{"Диалог":f"D{d['id']}",
            "Реплик":max(len(d["lines"].get(l,[])) for l in ["kyrgyz","english","russian"]),
            "Тема":d["title"]["russian"] or d["title"]["kyrgyz"]}
           for d in dialogues]
    fig8 = px.bar(pd.DataFrame(cnt), x="Диалог", y="Реплик", hover_data=["Тема"],
                  color="Реплик", color_continuous_scale="Blues",
                  title="Реплик в каждом диалоге", template="plotly_dark")
    fig8.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Косинусное сходство KY vs RU: примеры предложений")

    @st.cache_data
    def ex_sims():
        pairs = [(a["kyrgyz"]["example"],a["russian"]["example"],a["kyrgyz"]["pair"])
                 for a in antonyms if a["kyrgyz"]["example"] and a["russian"]["example"]]
        ky = encode([p[0] for p in pairs])
        ru = encode([p[1] for p in pairs])
        return ([float(cosine_similarity([ky[i]],[ru[i]])[0][0]) for i in range(len(pairs))],
                [p[2] for p in pairs])

    es, el = ex_sims()
    fig9 = px.bar(pd.DataFrame({"Пара":el,"Сходство":es}).sort_values("Сходство",ascending=False),
                  x="Пара", y="Сходство", color="Сходство",
                  color_continuous_scale="Tealrose", range_color=[0.3,1.0],
                  title="KY vs RU: косинусное сходство примеров", template="plotly_dark")
    fig9.update_layout(height=420, xaxis_tickangle=-40)
    st.plotly_chart(fig9, use_container_width=True)
    ca,cb,cc = st.columns(3)
    ca.metric("Среднее", f"{np.mean(es):.3f}")
    cb.metric("Мин", f"{np.min(es):.3f}")
    cc.metric("Макс", f"{np.max(es):.3f}")

    st.markdown("---")
    st.markdown("#### Длина слов-антонимов: KY vs RU")
    wl = [{"Пара":a["kyrgyz"]["pair"],"Слово":lbl,"Длина":ln}
          for a in antonyms
          for lbl,ln in [("W1-KY",len(a["kyrgyz"]["word1"])),("W2-KY",len(a["kyrgyz"]["word2"])),
                         ("W1-RU",len(a["russian"]["word1"])),("W2-RU",len(a["russian"]["word2"]))]]
    fig10 = px.bar(pd.DataFrame(wl), x="Пара", y="Длина", color="Слово", barmode="group",
                   title="Длина слов: KY vs RU", template="plotly_dark")
    fig10.update_layout(xaxis_tickangle=-40, height=400)
    st.plotly_chart(fig10, use_container_width=True)

    st.caption("Модель: paraphrase-multilingual-MiniLM-L12-v2 | dim=384 | cosine similarity")
