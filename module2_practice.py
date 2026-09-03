# Polynomial Regression Practice
# This example shows how polynomial regression can model curved data.
# The dots represent the actual data, and the curve represents the model's predictions.



import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# داده‌های x
x = np.linspace(1, 100, 10).reshape(-1, 1)

# داده‌های واقعی y
y = np.array([20, 34, 46, 10, 15, 17, 46, 66, 77, 82])


# ساخت Polynomial Transformer
transformer = PolynomialFeatures(
    degree=2,
    include_bias=False
)

# تبدیل x به x و x²
x_trans = transformer.fit_transform(x)


# ساخت و آموزش مدل
model = LinearRegression()

model.fit(x_trans, y)


# پیدا کردن کوچک‌ترین و بزرگ‌ترین x
min_value = np.min(x_trans[:, 0])
max_value = np.max(x_trans[:, 0])


# ساخت 100 نقطه بین min و max
# برای اینکه منحنی صاف رسم شود
X_seq = np.linspace(
    min_value,
    max_value,
    100
).reshape(-1, 1)


# تبدیل 100 نقطه به Polynomial Features
X_seq_trans = transformer.transform(X_seq)


# پیش‌بینی y برای 100 نقطه
y_pred = model.predict(X_seq_trans)



# ساخت مدل Linear Regression
linear_model = LinearRegression()

# آموزش مدل Linear Regression با داده‌های x و y
linear_model.fit(x, y)

# پیش‌بینی مقدار y برای هر مقدار x با استفاده از مدل Linear
linear_y_pred = linear_model.predict(x)

# ساخت یک Figure شامل دو نمودار کنار هم
# 1 یعنی یک ردیف و 2 یعنی دو ستون
fig, ax = plt.subplots(1, 2)

# رسم نقاط واقعی x و y در نمودار اول
ax[0].scatter(x, y)

# رسم خط پیش‌بینی Linear Regression در نمودار اول
ax[0].plot(x, linear_y_pred)

# قرار دادن عنوان برای نمودار اول
ax[0].set_title("Linear Regression")

# رسم نقاط واقعی x و y در نمودار دوم
ax[1].scatter(x, y)

# رسم منحنی پیش‌بینی Polynomial Regression در نمودار دوم
ax[1].plot(X_seq, y_pred)

# قرار دادن عنوان برای نمودار دوم
ax[1].set_title("Polynomial Regression")

# نمایش هر دو نمودار
plt.show()