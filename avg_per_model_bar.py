import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_order = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]

#loading the files
cpu_df = pd.read_csv("time_CPU.csv")
gpu_df = pd.read_csv("time_GPU_CUDA.csv")

# average inference time per model
cpu_avg = cpu_df.groupby("model")["inference_ms"].mean()
gpu_avg = gpu_df.groupby("model")["inference_ms"].mean()

cpu_avg = cpu_avg.reindex(model_order)
gpu_avg = gpu_avg.reindex(model_order)

models = model_order
x = np.arange(len(models))
width = 0.35

# plot 1
plt.figure(figsize=(10, 6))

bars_cpu = plt.bar(x - width/2, cpu_avg.values, width, label="CPU")
bars_gpu = plt.bar(x + width/2, gpu_avg.values, width, label="CUDA GPU")

plt.xlabel("YOLOv8 Models")
plt.ylabel("Average Inference Time (ms)")
plt.title("CPU vs CUDA Inference Time Across YOLOv8 Models")
plt.xticks(x, models)
plt.grid(axis="y", linestyle="--", alpha=0.3)

# ---- annotate bars ----
def label_bars(bars, text):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

label_bars(bars_cpu, "CPU")
label_bars(bars_gpu, "GPU")

plt.legend()
plt.tight_layout()
plt.show()

# plot 2
plt.figure(figsize=(10, 6))

line_cpu, = plt.plot(models, cpu_avg.values, marker="o", label="CPU")
line_gpu, = plt.plot(models, gpu_avg.values, marker="o", label="CUDA GPU")

plt.xlabel("YOLOv8 Model Size")
plt.ylabel("Average Inference Time (ms)")
plt.title("Inference Time Scaling Across YOLOv8 Models")

plt.grid(True, linestyle="--", alpha=0.3)

# ---- inline labels near the middle of the lines ----
mid_idx = len(models) // 2

plt.text(
    mid_idx + 0.1,
    cpu_avg.values[mid_idx],
    "CPU",
    color=line_cpu.get_color(),
    fontsize=10,
    fontweight="bold",
    va="center"
)

plt.text(
    mid_idx + 0.1,
    gpu_avg.values[mid_idx] * 2.5, # small upward offset
    "CUDA GPU",
    color=line_gpu.get_color(),
    fontsize=10,
    fontweight="bold",
    va="center"
)


plt.legend().remove()
plt.tight_layout()
plt.show()

# Plot 2.5
# plot 2: scaling behavior (corrected)
plt.figure(figsize=(10, 6))

# numeric x-axis representing increasing model complexity
x_idx = np.arange(len(model_order))
model_labels = ["n", "s", "m", "l", "x"]

plt.plot(
    x_idx,
    cpu_avg.values,
    marker="o",
    linewidth=2,
    label="CPU"
)

plt.plot(
    x_idx,
    gpu_avg.values,
    marker="o",
    linewidth=2,
    label="CUDA GPU"
)

plt.xlabel("YOLOv8 Model Variant (Increasing Complexity)")
plt.ylabel("Average Inference Time (ms)")
plt.title("Inference Time Scaling with YOLOv8 Model Complexity")

plt.xticks(x_idx, model_labels)
plt.yscale("log")

plt.text(3.3, cpu_avg.values[3], "Rapid CPU scaling", fontsize=9)
plt.text(3.3, gpu_avg.values[3], "Efficient GPU scaling", fontsize=9)

plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Plot 3
speedup = cpu_avg.values / gpu_avg.values

plt.figure(figsize=(10, 6))


bars = plt.barh(
    models,
    speedup,
    height=0.5,
    color="tab:green",
)

# reference line at 1×
# plt.axvline(1, color="gray", linestyle="--", linewidth=1)

plt.xlabel("Speed-up (× faster than CPU)")
plt.ylabel("YOLOv8 Models")
plt.title("GPU Speed-up over CPU")

# annotate bars
for bar, val in zip(bars, speedup):
    plt.text(
        val,
        bar.get_y() + bar.get_height() / 2,
        f" {val:.1f}×",
        va="center",
        ha="left",
        fontsize=9,
        # fontweight="bold"
    )

plt.tight_layout()
plt.show()

# plot 4 (pipeline overhead)

# average pre-process time per model
cpu_avg_pre = cpu_df.groupby("model")["preprocess_ms"].mean()
gpu_avg_pre = gpu_df.groupby("model")["preprocess_ms"].mean()

# average post-process time per model
cpu_avg_post = cpu_df.groupby("model")["postprocess_ms"].mean()
gpu_avg_post = gpu_df.groupby("model")["postprocess_ms"].mean()

# pre plot
plt.figure(figsize=(10, 5))

bars_cpu = plt.bar(
    x - width/2,
    cpu_avg_pre.reindex(models).values,
    width,
    label="CPU"
)

bars_gpu = plt.bar(
    x + width/2,
    gpu_avg_pre.reindex(models).values,
    width,
    label="CUDA GPU"
)

plt.xlabel("YOLOv8 Models")
plt.ylabel("Pre-processing Time (ms)")
plt.title("Pre-processing Overhead: CPU vs GPU")
plt.xticks(x, models)
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

bars_cpu = plt.bar(
    x - width/2,
    cpu_avg_post.reindex(models).values,
    width,
    label="CPU"
)

bars_gpu = plt.bar(
    x + width/2,
    gpu_avg_post.reindex(models).values,
    width,
    label="CUDA GPU"
)

plt.xlabel("YOLOv8 Models")
plt.ylabel("Post-processing Time (ms)")
plt.title("Post-processing Overhead: CPU vs GPU")
plt.xticks(x, models)
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()



# -------------------------------------------------------------------

# plt.close("all")
# cpu_pre  = np.array(cpu_avg_pre.reindex(model_order).values)
# cpu_inf  = np.array(cpu_avg.reindex(model_order).values)
# cpu_post = np.array(cpu_avg_post.reindex(model_order).values)
#
# gpu_pre  = np.array(gpu_avg_pre.reindex(model_order).values)
# gpu_inf  = np.array(gpu_avg.reindex(model_order).values)
# gpu_post = np.array(gpu_avg_post.reindex(model_order).values)
#
#
# # Data
# x = np.arange(len(model_order))
# width = 0.35
#
# # Stack values
# cpu_stack = [cpu_pre, cpu_inf, cpu_post]
# gpu_stack = [gpu_pre, gpu_inf, gpu_post]
#
# # Colors
# cpu_colors = ["#a6cee3", "#1f78b4", "#b2df8a"]
# gpu_colors = ["#fdbf6f", "#ff7f00", "#cab2d6"]
#
# # Labels
# cpu_labels = ["CPU Pre", "CPU Inference", "CPU Post"]
# gpu_labels = ["GPU Pre", "GPU Inference", "GPU Post"]
#
# plt.figure(figsize=(12, 6))
#
# # --- CPU bars ---
# bottom = np.zeros(len(model_order))
# for i in range(3):
#     plt.bar(x - width/2, cpu_stack[i], width, bottom=bottom, color=cpu_colors[i], label=cpu_labels[i])
#     bottom += cpu_stack[i]  # increment bottom for stacking
#
# # --- GPU bars ---
# bottom = np.zeros(len(model_order))
# for i in range(3):
#     plt.bar(x + width/2, gpu_stack[i], width, bottom=bottom, color=gpu_colors[i], label=gpu_labels[i])
#     bottom += gpu_stack[i]
#
# # --- Axes ---
# plt.xticks(x, model_order)
# plt.xlabel("YOLOv8 Models")
# plt.ylabel("Time (ms)")
# plt.title("Full Pipeline Time: CPU vs GPU")
# plt.grid(axis="y", linestyle="--", alpha=0.3)
#
# # --- Legend ---
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.show()