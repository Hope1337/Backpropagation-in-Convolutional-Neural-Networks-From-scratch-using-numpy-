import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Tải lại mảng từ tệp .npz
loaded_arrays = np.load('arrays.npz')

# Truy cập các mảng
predicts = loaded_arrays['predictions']
truths   = loaded_arrays['truths'][:,0]

print(predicts.shape)
print(truths.shape)
print((predicts == truths).sum())

accuracy = (predicts == truths).sum() / truths.shape[0]
cm       = confusion_matrix(truths, predicts)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
