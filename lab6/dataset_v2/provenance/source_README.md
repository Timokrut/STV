---
annotations_creators:
- manual
language:
- en
license: mit
multilinguality: monolingual
pretty_name: Car Parts and Damages Polygon Dataset
size_categories:
- 1K<n<10K
source_datasets: []
task_categories:
- image-segmentation
- object-detection
---

# Car Parts and Damages Polygon Dataset

## Dataset Summary

The **Car Parts and Damages Polygon Dataset** consists of **1,812 high-resolution images**, each annotated with polygon-based segmentation masks for either **car parts** or **car damages**. The dataset is designed to support training and evaluation of deep learning models for fine-grained object detection, instance segmentation, and automotive inspection tasks.

### ✅ Key Stats:
- **Total images**: 1,812  
  - **Car parts**: 998 images  
  - **Car damages**: 814 images  
- **Total polygons**: 24,851  

This dataset supports advanced computer vision tasks such as automated damage detection, insurance assessment, vehicle inspection, and part localization.

## Classes

### 🧩 Car Parts (998 Images)
Annotated polygon classes include:

- Windshield  
- Back-windshield  
- Front-window  
- Back-window  
- Front-door  
- Back-door  
- Front-wheel  
- Back-wheel  
- Front-bumper  
- Back-bumper  
- Headlight  
- Tail-light  
- Hood  
- Trunk  
- License-plate  
- Mirror  
- Roof  
- Grille  
- Rocker-panel  
- Quarter-panel  
- Fender  

### 💥 Car Damages (814 Images)
Annotated polygon classes include:

- Dent  
- Cracked  
- Scratch  
- Flaking  
- Broken part  
- Paint chip  
- Missing part  
- Corrosion  

## Use Cases

This dataset is ideal for developing:
- Instance segmentation models for automotive inspection  
- Damage classification and severity assessment tools  
- Insurance and repair estimate systems  
- Multi-class part detection and vehicle structure understanding  

## Dataset Format

- Annotations are provided in **polygon format**, compatible with tools like COCO JSON or VIA/VGG annotations.
- Each image is labeled with one of two categories: **car parts** or **car damages**, with corresponding polygon masks.