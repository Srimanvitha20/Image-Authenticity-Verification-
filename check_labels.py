from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r"C:\Users\suren\OneDrive\Desktop\image authenticity\dataset\ai-vs-real-images\Data Set 1\Data Set 1\train"

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

print("\n Class Index Mapping:")
print(train_gen.class_indices)
