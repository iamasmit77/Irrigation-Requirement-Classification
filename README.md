# Irrigation Requirement Classification

## Project Overview

This project is developed as part of the **Learn Depth Machine Learning Internship – Track 1**.

The project focuses on predicting the irrigation requirement of agricultural fields using machine learning. The model classifies irrigation requirements into three categories:

- Low
- Medium
- High

The project includes data preprocessing, exploratory data analysis (EDA), machine learning model training, evaluation, and a Streamlit-based application for prediction.

---

## Objective

The main objective of this project is to build a machine learning classification system that can determine the irrigation requirement of a field based on agricultural and environmental features such as:

- Soil moisture
- Soil pH
- Temperature
- Humidity
- Rainfall
- Crop type
- Crop growth stage
- Season
- Previous irrigation
- Field area
- Other soil and environmental factors

---

## Dataset

The original dataset is from the Kaggle competition **Predicting Irrigation Need – Playground Series Season 6 Episode 4**.

For this project, a **10,000-row stratified sample** was created from the original training dataset to make the project easier to work with on a beginner-level system.

### Target Variable

`Irrigation_Need`

The target contains three classes:

- `Low`
- `Medium`
- `High`

### Dataset File

```text
irrigation_capstone_10000.csv