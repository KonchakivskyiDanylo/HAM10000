import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import json
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
from huggingface_hub import hf_hub_download
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== ЗМІНИ НА СВІЙ USERNAME! =====
HF_REPO = "ardangerus/skin-ai-models"

MODEL_FILES = {
    "segmentation_unet.keras": "segmentation_unet.keras",
    "v4_efficientnet.keras": "v4_efficientnet.keras",
    "v5_convnext_seg.keras": "v5_convnext_seg.keras",
    "v6_densenet_4ch.keras": "v6_densenet_4ch.keras",
}

def download_models():
    for local_name, repo_name in MODEL_FILES.items():
        local_path = os.path.join(MODELS_DIR, local_name)
        if not os.path.exists(local_path):
            print(f"Downloading {repo_name}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=repo_name,
                local_dir=MODELS_DIR,
            )

download_models()
# ============================================================
# Config
# ============================================================
MODELS_DIR = "models"
SEG_IMG_SIZE = 256
CLF_IMG_SIZE_V4 = 260   # EfficientNetV2B2
CLF_IMG_SIZE_V5 = 224   # ConvNeXtTiny
CLF_IMG_SIZE_V6 = 224   # DenseNet121

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
LABEL_NAMES = {
    'akiec': 'Актинічний кератоз',
    'bcc':   'Базальноклітинна карцинома',
    'bkl':   'Доброякісний кератоз',
    'df':    'Дерматофіброма',
    'mel':   'Меланома',
    'nv':    'Меланоцитарний невус',
    'vasc':  'Судинне ураження'
}
LABEL_NAMES_EN = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}
RISK_LEVELS = {
    'akiec': ('⚠️ Передракове', '#f39c12'),
    'bcc':   ('🔴 Злоякісне', '#e74c3c'),
    'bkl':   ('🟢 Доброякісне', '#27ae60'),
    'df':    ('🟢 Доброякісне', '#27ae60'),
    'mel':   ('🔴 Злоякісне', '#e74c3c'),
    'nv':    ('🟢 Доброякісне', '#27ae60'),
    'vasc':  ('🟡 Доброякісне', '#3498db'),
}

RECOMMENDATIONS = {
    'akiec': {
        'title': 'Актинічний кератоз — Передракове ураження',
        'action': '🏥 Зверніться до дерматолога найближчим часом',
        'details': [
            'Це передракове ураження, яке може перерости в плоскоклітинну карциному',
            'Зазвичай виникає на ділянках шкіри, що піддаються сонячному впливу',
            'Лікування: кріотерапія, місцеві препарати (5-фторурацил, імікімод), фотодинамічна терапія',
            'Регулярний огляд дерматолога кожні 6 місяців',
            'Використовуйте сонцезахисний крем SPF 50+',
        ],
        'urgency': 'середня',
    },
    'bcc': {
        'title': 'Базальноклітинна карцинома — Злоякісне',
        'action': '🚨 Терміново зверніться до онкодерматолога',
        'details': [
            'Найпоширеніший тип раку шкіри, але з низьким ризиком метастазування',
            'Росте повільно, але може руйнувати навколишні тканини',
            'Лікування: хірургічне видалення (операція Мооса), кріотерапія, променева терапія',
            'Після лікування — регулярні огляди кожні 3-6 місяців',
            'Прогноз зазвичай сприятливий при ранньому виявленні',
        ],
        'urgency': 'висока',
    },
    'bkl': {
        'title': 'Доброякісний кератоз — Не небезпечно',
        'action': '✅ Спостереження, лікування не обов\'язкове',
        'details': [
            'Доброякісне утворення, не переростає в рак',
            'Включає себорейний кератоз, сонячне лентиго та ліхеноїдний кератоз',
            'Видалення лише з косметичних причин або при подразненні',
            'Методи видалення: кріотерапія, кюретаж, лазер',
            'Якщо змінюється колір, форма або розмір — покажіть дерматологу',
        ],
        'urgency': 'низька',
    },
    'df': {
        'title': 'Дерматофіброма — Не небезпечно',
        'action': '✅ Спостереження, лікування зазвичай не потрібне',
        'details': [
            'Доброякісне щільне утворення в шкірі, часто на ногах',
            'Безпечне, не переростає в злоякісне',
            'Може виникати після травми або укусу комахи',
            'Видалення лише якщо заважає або з косметичних причин',
            'Зверніться до лікаря якщо швидко росте або болить',
        ],
        'urgency': 'низька',
    },
    'mel': {
        'title': 'Меланома — Небезпечне злоякісне утворення',
        'action': '🚨 ТЕРМІНОВО зверніться до онколога!',
        'details': [
            'Найнебезпечніший тип раку шкіри з високим ризиком метастазування',
            'Раннє виявлення критично важливе для прогнозу',
            'Правило ABCDE: Асиметрія, нерівні Борди, нерівномірний Колір, Діаметр >6мм, Еволюція',
            'Лікування: хірургічне видалення з широкими краями, імунотерапія, таргетна терапія',
            'НЕ ЗВОЛІКАЙТЕ з візитом до лікаря — час має значення!',
        ],
        'urgency': 'критична',
    },
    'nv': {
        'title': 'Меланоцитарний невус (родимка) — Не небезпечно',
        'action': '✅ Звичайна родимка, спостерігайте за змінами',
        'details': [
            'Доброякісне пігментне утворення, є у більшості людей',
            'Безпечне за умови відсутності змін',
            'Слідкуйте за правилом ABCDE: зверніться до лікаря при будь-яких змінах',
            'Фотографуйте родимки раз на рік для порівняння',
            'Захищайте від прямого сонця, не травмуйте',
        ],
        'urgency': 'низька',
    },
    'vasc': {
        'title': 'Судинне ураження — Зазвичай не небезпечно',
        'action': '✅ Спостереження або косметичне видалення',
        'details': [
            'Включає ангіоми, ангіокератоми та піогенні гранульоми',
            'Зазвичай доброякісне, не переростає в рак',
            'Видалення лазером або електрокоагуляцією при необхідності',
            'Зверніться до лікаря якщо кровоточить або швидко росте',
            'Піогенна гранульома потребує видалення',
        ],
        'urgency': 'низька',
    },
}

# ============================================================
# Custom objects for model loading
# ============================================================
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def call(self, y_true, y_pred):
        nc = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / nc
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1 - y_pred, self.gamma) * ce, axis=-1)
    def get_config(self):
        c = super().get_config()
        c.update({'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return c

def dice_coeff(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1.0):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred_bin)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

CUSTOM_OBJECTS = {
    'FocalLoss': FocalLoss,
    'bce_dice_loss': bce_dice_loss,
    'dice_coeff': dice_coeff,
    'iou_metric': iou_metric,
}

# ============================================================
# Model loading (cached)
# ============================================================
@st.cache_resource
def load_seg_model():
    path = os.path.join(MODELS_DIR, "segmentation_unet.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)

@st.cache_resource
def load_clf_v4():
    path = os.path.join(MODELS_DIR, "v4_efficientnet.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)

@st.cache_resource
def load_clf_v5():
    path = os.path.join(MODELS_DIR, "v5_convnext_seg.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)

@st.cache_resource
def load_clf_v6():
    path = os.path.join(MODELS_DIR, "v6_densenet_4ch.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)

# ============================================================
# Inference functions
# ============================================================
def predict_mask(model, img_rgb, img_size=SEG_IMG_SIZE):
    img_resized = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    pred = model.predict(img_resized[np.newaxis, ...], verbose=0)[0, ..., 0]
    mask = (pred > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_full

def predict_v4(model, img_rgb):
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    img = cv2.resize(img_rgb, (CLF_IMG_SIZE_V4, CLF_IMG_SIZE_V4))
    img = preprocess_input(img.astype(np.float32))
    return model.predict(img[np.newaxis, ...], verbose=0)[0]

def predict_v5(model, img_rgb, mask):
    from tensorflow.keras.applications.convnext import preprocess_input
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
    mask_soft = np.clip(mask_soft, 0, 1)
    img_masked = (img_rgb.astype(np.float32) * mask_soft[..., np.newaxis]).astype(np.uint8)
    img = cv2.resize(img_masked, (CLF_IMG_SIZE_V5, CLF_IMG_SIZE_V5))
    img = preprocess_input(img.astype(np.float32))
    return model.predict(img[np.newaxis, ...], verbose=0)[0]

def predict_v6(model, img_rgb, mask):
    from tensorflow.keras.applications.densenet import preprocess_input
    img = cv2.resize(img_rgb, (CLF_IMG_SIZE_V6, CLF_IMG_SIZE_V6))
    mask_resized = cv2.resize(mask, (CLF_IMG_SIZE_V6, CLF_IMG_SIZE_V6))
    img_preprocessed = preprocess_input(img.astype(np.float32))
    mask_norm = mask_resized.astype(np.float32)
    img_4ch = np.concatenate([img_preprocessed, mask_norm[..., np.newaxis]], axis=-1)
    return model.predict(img_4ch[np.newaxis, ...], verbose=0)[0]

def create_overlay(img_rgb, mask, alpha=0.4):
    overlay = img_rgb.copy()
    green = np.zeros_like(img_rgb)
    green[:, :, 1] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(img_rgb, 1 - alpha, green, alpha, 0)[mask_bool]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay

def create_heatmap_overlay(img_rgb, mask):
    h, w = img_rgb.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

# ============================================================
# UI
# ============================================================
st.set_page_config(
    page_title="SkinAI — Аналіз шкірних уражень",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid;
        margin: 0.5rem 0;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.3rem 0;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🔬 SkinAI</h1><p style='font-size: 1.2rem; color: #666;'>Аналіз шкірних уражень за допомогою ШІ</p></div>", unsafe_allow_html=True)

st.markdown("""<div class='disclaimer'>
⚠️ <b>Застереження:</b> Цей інструмент створено для навчальних цілей і НЕ є медичним діагнозом. 
Завжди звертайтесь до дерматолога для професійної консультації.
</div>""", unsafe_allow_html=True)

# Load models
with st.spinner("Завантаження моделей..."):
    seg_model = load_seg_model()
    clf_v4 = load_clf_v4()
    clf_v5 = load_clf_v5()
    clf_v6 = load_clf_v6()

loaded = []
if seg_model: loaded.append("U-Net")
if clf_v4: loaded.append("v4")
if clf_v5: loaded.append("v5")
if clf_v6: loaded.append("v6")

if not loaded:
    st.error("❌ Жодна модель не знайдена! Покладіть файли моделей в папку `models/`")
    st.stop()

st.sidebar.markdown("### Моделі")
st.sidebar.success(f"Завантажено: {', '.join(loaded)}")

# Upload
uploaded_file = st.file_uploader(
    "📷 Завантажте фото шкірного ураження",
    type=["jpg", "jpeg", "png", "bmp", "heic", "heif"],
    help="Підтримуються формати: JPG, PNG, BMP, HEIC"
)

if uploaded_file is not None:
    # Read image (supports HEIC via Pillow)
    pil_img = Image.open(uploaded_file).convert('RGB')
    img_rgb = np.array(pil_img)

    st.markdown("---")

    # ===== Segmentation =====
    if seg_model:
        with st.spinner("Сегментація..."):
            mask = predict_mask(seg_model, img_rgb)
    else:
        mask = np.ones(img_rgb.shape[:2], dtype=np.uint8)

    # ===== Classification =====
    predictions = {}
    with st.spinner("Класифікація..."):
        if clf_v4:
            predictions['v4'] = predict_v4(clf_v4, img_rgb)
        if clf_v5 and seg_model:
            predictions['v5'] = predict_v5(clf_v5, img_rgb, mask)
        if clf_v6 and seg_model:
            predictions['v6'] = predict_v6(clf_v6, img_rgb, mask)

    # Ensemble (simple average)
    if predictions:
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
    else:
        st.error("Жодна класифікаційна модель не завантажена!")
        st.stop()

    top_class = CLASSES[np.argmax(ensemble_pred)]
    top_prob = np.max(ensemble_pred)
    risk_label, risk_color = RISK_LEVELS[top_class]

    # ===== Display =====
    # Row 1: Images
    st.markdown("### 📸 Візуалізація")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_rgb, caption="Оригінал", use_container_width=True)

    with col2:
        if seg_model:
            overlay = create_overlay(img_rgb, mask)
            st.image(overlay, caption="Сегментація", use_container_width=True)
        else:
            st.info("U-Net не завантажено")

    with col3:
        if seg_model:
            mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
            seg_img = (img_rgb * mask_soft[..., np.newaxis]).astype(np.uint8)
            st.image(seg_img, caption="Ізольоване ураження", use_container_width=True)
        else:
            st.info("U-Net не завантажено")

    st.markdown("---")

    # Row 2: Diagnosis
    st.markdown("### 🩺 Результат")

    col_diag, col_probs = st.columns([1, 1])

    with col_diag:
        st.markdown(f"""
        <div class='risk-card' style='border-color: {risk_color}; background: {risk_color}11;'>
            <h2 style='margin: 0; color: {risk_color};'>{risk_label}</h2>
            <h3 style='margin: 0.5rem 0;'>{LABEL_NAMES[top_class]}</h3>
            <p style='color: #666; margin: 0;'>{LABEL_NAMES_EN[top_class]}</p>
            <h1 style='margin: 0.5rem 0; color: {risk_color};'>{top_prob*100:.1f}%</h1>
            <p style='color: #888; font-size: 0.9rem;'>Впевненість ensemble ({len(predictions)} моделі)</p>
        </div>
        """, unsafe_allow_html=True)

        # Coverage info
        if seg_model:
            coverage = mask.sum() / mask.size * 100
            st.markdown(f"""
            <div class='metric-box'>
                <b>Площа ураження:</b> {coverage:.1f}% зображення
            </div>
            """, unsafe_allow_html=True)

    with col_probs:
        st.markdown("**Ймовірності по класах:**")
        # Sort by probability
        sorted_idx = np.argsort(ensemble_pred)[::-1]
        for idx in sorted_idx:
            cls = CLASSES[idx]
            prob = ensemble_pred[idx]
            risk_tag, color = RISK_LEVELS[cls]

            bar_color = color if prob > 0.1 else '#ddd'
            st.markdown(f"""
            <div style='margin: 4px 0;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.9rem;'>
                    <span><b>{cls}</b> — {LABEL_NAMES[cls]}</span>
                    <span><b>{prob*100:.1f}%</b></span>
                </div>
                <div style='background: #eee; border-radius: 4px; height: 8px; margin-top: 2px;'>
                    <div style='background: {bar_color}; width: {prob*100:.1f}%; height: 8px; border-radius: 4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Row 3: Recommendations
    st.markdown("---")
    st.markdown("### 💊 Рекомендації")

    rec = RECOMMENDATIONS[top_class]

    urgency_colors = {
            'низька': '#27ae60',
            'середня': '#f39c12',
            'висока': '#e74c3c',
            'критична': '#c0392b',
        }
    urg_color = urgency_colors[rec['urgency']]

    st.markdown(f"""
        <div class='risk-card' style='border-color: {urg_color}; background: {urg_color}11;'>
            <h3 style='margin: 0;'>{rec['action']}</h3>
            <p style='color: #666; margin: 0.5rem 0;'>Терміновість: 
                <b style='color: {urg_color};'>{rec['urgency'].upper()}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    for point in rec['details']:
            st.markdown(f"• {point}")

    if top_prob < 0.5:
            st.warning(
                "⚠️ Впевненість моделі низька (<50%). Результат може бути неточним — обов'язково зверніться до дерматолога.")
    # Row 3: Per-model breakdown
    if len(predictions) > 1:
        st.markdown("---")
        st.markdown("### 🔍 Деталі по моделях")

        model_names = {
            'v4': 'EfficientNetV2B2 (raw)',
            'v5': 'ConvNeXtTiny (segmented)',
            'v6': 'DenseNet121 (4-channel)',
        }

        cols = st.columns(len(predictions))
        for i, (key, pred) in enumerate(predictions.items()):
            with cols[i]:
                top_cls = CLASSES[np.argmax(pred)]
                top_p = np.max(pred)
                r_label, r_color = RISK_LEVELS[top_cls]
                st.markdown(f"""
                <div class='metric-box'>
                    <p style='font-size: 0.8rem; color: #888;'>{model_names.get(key, key)}</p>
                    <h4 style='margin: 0.3rem 0; color: {r_color};'>{top_cls} — {top_p*100:.1f}%</h4>
                    <p style='font-size: 0.75rem; color: #aaa;'>{LABEL_NAMES[top_cls]}</p>
                </div>
                """, unsafe_allow_html=True)

        # Agreement check
        model_preds = [CLASSES[np.argmax(p)] for p in predictions.values()]
        if len(set(model_preds)) == 1:
            st.success("✅ Всі моделі згодні з діагнозом")
        else:
            st.warning(f"⚠️ Моделі не згодні: {', '.join([f'{k}→{CLASSES[np.argmax(v)]}' for k, v in predictions.items()])}")

    # Disclaimer bottom
    st.markdown("---")
    st.caption("🔬 SkinAI | HAM10000 Dataset | Ensemble: EfficientNetV2B2 + ConvNeXtTiny + DenseNet121 | U-Net Segmentation")
    st.caption("⚠️ Не є медичним діагнозом. Зверніться до лікаря.")
