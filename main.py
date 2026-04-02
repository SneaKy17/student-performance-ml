import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student_data.csv")

X = data[['Hours', 'Attendance']]
y = data['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)


new_data = pd.DataFrame([[6, 75]], columns=['Hours', 'Attendance'])
prediction = model.predict(new_data)[0]

print("Predicted Marks:", prediction)

if prediction > 80:
    print("Excellent! Keep it up 🔥")
elif prediction > 60:
    print("Good, try increasing study hours slightly")
else:
    print("Focus more on studies and attendance")
while True:
    hours = float(input("Enter hours: "))
    attendance = float(input("Enter attendance: "))

    new_data = pd.DataFrame([[hours, attendance]], columns=['Hours', 'Attendance'])
    prediction = model.predict(new_data)[0]

    print("Predicted Marks:", prediction)
    break


import matplotlib.pyplot as plt 

plt.scatter(data['Hours'], data['Marks'])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

from sklearn.metrics import r2_score

y_pred = model.predict(X_test)
print("Accuracy:", r2_score(y_test, y_pred))


import joblib

joblib.dump(model, "student_model.pkl")
joblib.load("student_model.pkl")

try:
    hours = float(input("Enter hours: "))
except:
    print("Invalid input")
