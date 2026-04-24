import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------- SAMPLE DATA ----------------
# Features:
# [Budget, Distance, Rating, Amenities, Food Quality]

X = np.array([
    [2000, 2, 4.5, 5, 5],
    [1500, 5, 4.0, 4, 4],
    [3000, 1, 5.0, 5, 5],
    [1000, 10, 3.5, 3, 3],
    [2500, 3, 4.2, 4, 5],
    [1800, 4, 4.0, 4, 4],
    [2200, 2, 4.8, 5, 5],
    [1200, 8, 3.8, 3, 3]
])

# Satisfaction score (target variable)
y = np.array([9, 7, 10, 5, 8, 7, 9, 6])

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------
print("🏨 HOTEL RECOMMENDATION SYSTEM 🏨")

budget = int(input("Enter your budget (₹): "))
distance = float(input("Enter distance from city center (km): "))
rating = float(input("Enter preferred rating (1 to 5): "))
amenities = int(input("Enter number of amenities (1 to 5): "))
food = float(input("Enter food quality preference (1 to 5): "))

# ---------------- PREDICTION ----------------
user_input = np.array([[budget, distance, rating, amenities, food]])

score = model.predict(user_input)[0]

print("\n📊 HOTEL ANALYSIS RESULT 📊")
print(f"⭐ Predicted Satisfaction Score: {round(score, 2)} / 10")

# ---------------- RECOMMENDATION ----------------
if score >= 8:
    print("🏆 Recommendation: EXCELLENT HOTEL – Book it immediately!")
elif score >= 6:
    print("👍 Recommendation: GOOD HOTEL – Worth considering.")
else:
    print("⚠️ Recommendation: NOT RECOMMENDED – Look for better options.")
