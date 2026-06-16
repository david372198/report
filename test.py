import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
import requests
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================== 模型載入 ==================
model = tf.keras.models.load_model("pokemon_model.keras")

# ================== class_names ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "class_names.json")

with open(json_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)


# ================== PokéAPI ==================
def get_pokemon_info(name):
    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    res = requests.get(url)

    if res.status_code != 200:
        return None

    data = res.json()

    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

    return {
        "name": data["name"],
        "types": [t["type"]["name"] for t in data["types"]],
        "sprite": data["sprites"]["front_default"],
        # 單位修正
        "height_m": data["height"] / 10,
        "weight_kg": data["weight"] / 10,
        # ⭐ 能力值
        "hp": stats.get("hp"),
        "attack": stats.get("attack"),
        "defense": stats.get("defense"),
        "sp_attack": stats.get("special-attack"),
        "sp_defense": stats.get("special-defense"),
        "speed": stats.get("speed"),
    }


def get_pokemon_species(name):
    url = f"https://pokeapi.co/api/v2/pokemon-species/{name.lower()}"
    res = requests.get(url)

    if res.status_code != 200:
        return None

    data = res.json()

    return {
        "habitat": data["habitat"]["name"] if data["habitat"] else "unknown",
        "is_legendary": data["is_legendary"],
        "is_mythical": data["is_mythical"],
    }


# ================== 預測函數 ==================
def predict_image(img):
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    top_idx = np.argsort(preds)[::-1][:3]
    results = [(class_names[i], float(preds[i])) for i in top_idx]

    return results


# ================== UI ==================
st.title("🔥 Pokémon AI 圖鑑系統")

uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="輸入圖片", use_column_width=True)

    # ================== AI 預測 ==================
    results = predict_image(image)

    st.subheader("🔮 AI 預測結果 (Top 3)")
    for name, score in results:
        st.write(f"👉 {name} : {score:.4f}")

    top_name = results[0][0]

    st.success(f"🎯 最可能結果：{top_name}")

    # ================== PokéAPI ==================
    info = get_pokemon_info(top_name)
    species = get_pokemon_species(top_name)

    if info:
        st.subheader("📘 Pokémon 基本資料")

        col1, col2 = st.columns(2)

        with col1:
            st.write("🔥 屬性：", ", ".join(info["types"]))
            st.write(f"📏 身高：{info['height_m']} m")
            st.write(f"⚖️ 體重：{info['weight_kg']} kg")
            st.subheader("⚔️ 能力值 (Stats)")
            st.write(f"❤️ HP：{info['hp']}")
            st.write(f"💥 攻擊：{info['attack']}")
            st.write(f"🛡️ 防禦：{info['defense']}")
            st.write(f"✨ 特攻：{info['sp_attack']}")
            st.write(f"🔰 特防：{info['sp_defense']}")
            st.write(f"⚡ 速度：{info['speed']}")

        with col2:
            if info["sprite"]:
                st.image(info["sprite"], caption=top_name)

    if species:
        st.subheader("🌍 世界觀資訊")

        st.write("🌿 棲息地：", species["habitat"])
        st.write("⭐ 傳說寶可夢：", species["is_legendary"])
        st.write("✨ 幻之寶可夢：", species["is_mythical"])
