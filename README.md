# ==============================
# SLEEP INTELLIGENCE SYSTEM
# TERMINAL + BAR + PIE + CSV
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# DATASET
# ==============================

sleep_data = {
    "sleep_hours": [4, 5, 6, 7, 8, 6, 5, 9],
    "screen_time": [130, 110, 90, 60, 30, 80, 120, 20],
    "exercise": [0, 10, 20, 30, 45, 15, 5, 60],
    "stress": [9, 8, 7, 5, 3, 6, 8, 2],
    "caffeine": [3, 3, 2, 1, 0, 2, 3, 0],
    "sleep_quality": ["Poor", "Poor", "Average", "Average",
                      "Good", "Average", "Poor", "Good"]
}

df = pd.DataFrame(sleep_data)

# ==============================
# MODEL TRAINING
# ==============================

X = df.drop("sleep_quality", axis=1)
y = df["sleep_quality"]

sleep_model = RandomForestClassifier(random_state=42)
sleep_model.fit(X, y)

# ==============================
# EMOTION NLP MODEL
# ==============================

texts = [
    "I am stressed",
    "I am calm",
    "I am anxious",
    "I feel happy"
]
labels = ["Negative", "Positive", "Negative", "Positive"]

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(texts)

emotion_model = LogisticRegression()
emotion_model.fit(X_text, labels)

def predict_emotion(text):
    return emotion_model.predict(vectorizer.transform([text]))[0]

# ==============================
# HELPER FUNCTIONS
# ==============================

def calculate_sleep_debt(past_sleep_hours):
    debt = 0
    for h in past_sleep_hours:
        if h < 8:
            debt += (8 - h)
    return debt

def explain_sleep(user):
    reasons = []
    impact = {}

    impact["Sleep Hours"] = max(0, 8 - user["sleep_hours"])
    impact["Screen Time"] = max(0, user["screen_time"] - 60)
    impact["Stress"] = user["stress"]
    impact["Exercise"] = max(0, 30 - user["exercise"])
    impact["Caffeine"] = user["caffeine"] * 10

    if user["sleep_hours"] < 6:
        reasons.append("Low sleep hours")
    if user["screen_time"] > 90:
        reasons.append("High screen time")
    if user["stress"] > 7:
        reasons.append("High stress")
    if user["exercise"] < 20:
        reasons.append("Low exercise")
    if user["caffeine"] > 2:
        reasons.append("High caffeine")

    return reasons, impact

def habit_swap_suggestions(reasons):
    tips = []
    if "Low sleep hours" in reasons:
        tips.append("Sleep earlier")
    if "High screen time" in reasons:
        tips.append("Reduce phone usage before bed")
    if "High stress" in reasons:
        tips.append("Practice meditation")
    if "Low exercise" in reasons:
        tips.append("Add 20 minutes walking")
    if "High caffeine" in reasons:
        tips.append("Avoid caffeine at night")
    return tips

def show_charts(impact):
    labels = list(impact.keys())
    values = list(impact.values())

    plt.figure()
    plt.bar(labels, values)
    plt.title("Sleep Factors - Bar Chart")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Sleep Factors - Pie Chart")
    plt.tight_layout()
    plt.show()

def save_to_csv(user, prediction, emotion, reasons, tips, debt, risk,
                filename="sleep_history.csv"):

    row = {
        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sleep_Hours": user["sleep_hours"],
        "Screen_Time": user["screen_time"],
        "Exercise": user["exercise"],
        "Stress": user["stress"],
        "Caffeine": user["caffeine"],
        "Prediction": prediction,
        "Emotion": emotion,
        "Sleep_Debt": debt,
        "Future_Risk": risk,
        "Reasons": ", ".join(reasons),
        "Suggestions": ", ".join(tips)
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(filename):
        df_row.to_csv(filename, mode="a", header=False, index=False)
    else:
        df_row.to_csv(filename, index=False)

# ==============================
# MAIN EXECUTION
# ==============================

user_input = {
    "sleep_hours": 5,
    "screen_time": 120,
    "exercise": 10,
    "stress": 9,
    "caffeine": 3
}

mood_text = "I am stressed about work"
past_sleep = [5, 6, 5, 4, 6, 5, 6]

prediction = sleep_model.predict(pd.DataFrame([user_input]))[0]
emotion = predict_emotion(mood_text)

debt = calculate_sleep_debt(past_sleep)

reasons, impact = explain_sleep(user_input)
tips = habit_swap_suggestions(reasons)

risk = "LOW"
if debt > 5:
    risk = "HIGH RISK OF FUTURE POOR SLEEP"

# TERMINAL OUTPUT
print("\n==============================")
print(" SLEEP INTELLIGENCE RESULT")
print("==============================\n")
print("Predicted Sleep Quality :", prediction)
print("Detected Emotion        :", emotion)

print("\nReasons:")
for i, r in enumerate(reasons, 1):
    print(f"{i}. {r}")

print("\nSuggestions:")
for i, t in enumerate(tips, 1):
    print(f"{i}. {t}")

print("\nSleep Debt (Hours)      :", debt)
print("Future Sleep Risk       :", risk)

# SAVE CSV
save_to_csv(
    user_input,
    prediction,
    emotion,
    reasons,
    tips,
    debt,
    risk
)

# SHOW CHARTS
show_charts(impact)
