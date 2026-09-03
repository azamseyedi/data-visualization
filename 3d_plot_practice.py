# تمرین رسم نمودار سه‌بعدی
# در این تمرین با استفاده از Matplotlib یک نمودار خطی سه‌بعدی می‌سازیم.
# برای نمودار سه‌بعدی به سه مختصات x، y و z نیاز داریم.
# با sin و cos مقادیر x و y را به شکل منظم تولید می‌کنیم.

import matplotlib.pyplot as plt
import numpy as np

# ساخت 100 مقدار از 0 تا 50 برای محور z
z = np.linspace(0, 50, 100)

# ساخت مقادیر x با استفاده از sin
x = np.sin(z)

# ساخت مقادیر y با استفاده از cos
y = np.cos(z)

# ساخت Figure
fig = plt.figure()

# ساخت محور سه‌بعدی
ax = plt.axes(projection="3d")

# رسم خط سه‌بعدی
ax.plot3D(x, y, z, "blue", label="The line")

# نمایش legend
ax.legend()

# نمایش نمودار
plt.show()