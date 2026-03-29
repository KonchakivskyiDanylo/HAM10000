# 🔬 SkinAI — Аналіз шкірних уражень

Веб-застосунок для класифікації шкірних уражень за допомогою ensemble з 3 моделей + U-Net сегментації.

## Встановлення

### 1. Клонувати/скопіювати проект

```bash
cd skin_app
```

### 2. Встановити залежності

```bash
pip install -r requirements.txt
```

### 3. Покласти моделі в папку `models/`

```
skin_app/
├── app.py
├── requirements.txt
├── models/
│   ├── segmentation_unet.keras        ← U-Net сегментація
│   ├── v4_efficientnet.keras           ← EfficientNetV2B2
│   ├── v5_convnext_seg.keras           ← ConvNeXtTiny + Seg
│   └── v6_densenet_4ch.keras           ← DenseNet121 4-channel
```

**Назви файлів повинні бути точно такі!** Перейменуй свої файли:

| Твій файл на Drive | Перейменувати в |
|---------------------|-----------------|
| `ham10000_segmentation_unet.keras` | `segmentation_unet.keras` |
| `ham10000_efficientnetv2b2.keras` або `best_model.keras` | `v4_efficientnet.keras` |
| `ham10000_v5_convnext_seg.keras` | `v5_convnext_seg.keras` |
| `ham10000_v6_densenet_4ch.keras` | `v6_densenet_4ch.keras` |

### 4. Запустити

```bash
streamlit run app.py
```

Відкриється в браузері на `http://localhost:8501`

## Як працює

1. Користувач завантажує фото шкірного ураження
2. **U-Net** генерує сегментаційну маску (виділяє ураження від фону)
3. **3 моделі** класифікують (кожна по-своєму використовує зображення):
   - v4: EfficientNetV2B2 на raw RGB
   - v5: ConvNeXtTiny на segmented image
   - v6: DenseNet121 на 4-channel (RGB + mask)
4. **Simple Average Ensemble** об'єднує ймовірності
5. Відображається діагноз + маска + ймовірності

## 7 класів HAM10000

| Клас | Назва | Ризик |
|------|-------|-------|
| akiec | Актинічний кератоз | ⚠️ Передракове |
| bcc | Базальноклітинна карцинома | 🔴 Злоякісне |
| bkl | Доброякісний кератоз | 🟢 Доброякісне |
| df | Дерматофіброма | 🟢 Доброякісне |
| mel | Меланома | 🔴 Злоякісне |
| nv | Меланоцитарний невус | 🟢 Доброякісне |
| vasc | Судинне ураження | 🟡 Доброякісне |
