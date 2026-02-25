import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("⚽ توقعات مباريات اليوم")

# ===== نموذج الذكاء الاصطناعي =====
data = {
    "home_attack":[1.9,1.5,2.1,1.3,1.8,2.0,1.4,1.7],
    "home_defense":[0.9,1.1,0.8,1.3,1.0,0.9,1.2,1.1],
    "away_attack":[1.4,1.7,1.2,1.6,1.3,1.5,1.8,1.4],
    "away_defense":[1.2,1.4,1.1,1.0,1.3,1.2,1.5,1.2],
    "result":[2,0,2,1,2,2,0,1]
}

df = pd.DataFrame(data)
X = df.drop("result", axis=1)
y = df["result"]
model = RandomForestClassifier()
model.fit(X, y)

# ===== مباريات اليوم =====
matches_today = [
    ("Real Madrid","Benfica",[1.9,0.9,1.4,1.2]),
    ("PSG","Monaco",[2.0,1.1,1.6,1.3]),
    ("Juventus","Galatasaray",[1.5,1.0,1.7,1.4]),
]

st.subheader("المباريات القادمة")

for home, away, stats in matches_today:
    result = model.predict([stats])[0]
    winner = home if result==2 else "تعادل" if result==1 else away
    st.write(f"**{home} vs {away}** ➜ التوقع: {winner}")