import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ======================================================
# 📊 رسم مصفوفة الالتباس مع تدرج ألوان واضح
# ======================================================
def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    cmatrix = confusion_matrix(y_true, y_pred)
    
    # Normalize to % if تريد
    # cmatrix_norm = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    
    # نستخدم تدرج أبيض → أحمر
    cmap = sns.light_palette("red", as_cmap=True)

    sns.heatmap(
        cmatrix,
        annot=True,
        fmt='d',
        xticklabels=classes,
        yticklabels=classes,
        cmap=cmap,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('True', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # بدل plt.close() لتظهر مباشرة

# # ======================================================
# # 📈 رسم منحنى التدريب
# # ======================================================
def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
