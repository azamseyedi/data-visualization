# تمرین Confusion Matrix
# در این تمرین ابتدا یک مدل Logistic Regression می‌سازیم و آن را با داده‌ها آموزش می‌دهیم.
# سپس مدل برای داده‌ها پیش‌بینی انجام می‌دهد.
# بعد با مقایسه جواب‌های واقعی و جواب‌های پیش‌بینی‌شده، Confusion Matrix ساخته می‌شود.
# در آخر Confusion Matrix را به صورت Heatmap نمایش می‌دهیم تا نتیجه پیش‌بینی مدل را بهتر ببینیم.




import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# داده‌های ورودی
x = np.arange(10).reshape(-1, 1)

# جواب‌های واقعی
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# ساخت مدل Logistic Regression
model = LogisticRegression(
    solver="liblinear",
    random_state=0,
    C=10.0
)

# آموزش مدل
model.fit(x, y)

# پیش‌بینی
y_pred = model.predict(x)

# ساخت Confusion Matrix
cm = confusion_matrix(y, y_pred)

# ساخت نمودار
fig, ax = plt.subplots()

# تبدیل Confusion Matrix به Heatmap
im = ax.imshow(cm, cmap="seismic")

# اسم محورهای نمودار
ax.xaxis.set(
    ticks=(0, 1),
    ticklabels=("Predicted 0s", "Predicted 1s")
)

ax.yaxis.set(
    ticks=(0, 1),
    ticklabels=("Actual 0s", "Actual 1s")
)

ax.set_ylim(1.5, -0.5)

# نوشتن اعداد داخل خانه‌های Heatmap
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white"
        )

# اضافه کردن Colorbar
fig.colorbar(im)

# نمایش نمودار
plt.show()