import json
import matplotlib.pyplot as plt

# Load both result files
with open("results_model_using_scikit.json", "r") as f1:
    scikit_results = json.load(f1)

with open("results_model_without_scikit.json", "r") as f2:
    manual_resault = json.load(f2)
    
# Extract data
k_values = scikit_results["k_values"]
scikit_train = scikit_results["training_accuracy"]
scikit_val = scikit_results["validation_accuracy"]

manual_train = manual_resault["training_accuracy"]
manual_val = manual_resault["validation_accuracy"]

# --- Plot comparison ---
plt.figure(figsize=(10, 6))

# Training accuracy comparison
plt.plot(k_values, scikit_train, marker='o', label="Scikit (Training)", linestyle='--')
plt.plot(k_values, manual_train, marker='o', label="Manually (Training)", linestyle='--')

# Validation accuracy comparison
plt.plot(k_values, scikit_val, marker='s', label="Scikit (Validation)", linewidth=2)
plt.plot(k_values, manual_val, marker='s', label="Manually (Validation)", linewidth=2)

# Highlight best points
plt.scatter(scikit_results["best_k"], scikit_results["best_val_accuracy"],
            color='blue', s=100, label=f"Scikit Best k={scikit_results['best_k']}")
plt.scatter(manual_resault["best_k"], manual_resault["best_val_accuracy"],
            color='orange', s=100, label=f"Manually Best k={manual_resault['best_k']}")

# Labels and title
plt.title("KNN Comparison: Scikit vs Manually Scaling", fontsize=14)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary
print("=== Comparison Summary ===")
print(f"Scikit:  Best k={scikit_results['best_k']}, Accuracy={scikit_results['best_val_accuracy']:.4f}")
print(f"Manually:  Best k={manual_resault['best_k']}, Accuracy={manual_resault['best_val_accuracy']:.4f}")
