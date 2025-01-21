from NPset import Create_NPset
from plugin_model import Fake_Or_Real_Dog_Model
from pred_model import Check_Image

################################################################
# Create a new npy files
################################################################
Np_fake = Create_NPset('data/fake_dogs', 'data/synthetic_dog.npy')
Np_real = Create_NPset('data/real_dogs', 'data/real_dog.npy')

Np_fake.save_synthetic_images_to_file()
Np_real.save_synthetic_images_to_file()

print("Files success")
################################################################
# Train and learn model
################################################################

model = Fake_Or_Real_Dog_Model()
model.train_and_save()

print("Model success")
################################################################
# Check results of training
################################################################
print("Check results of training")

check = Check_Image()
print("Fake image")
check.Fake_or_Real('source/fake_dog.jpg')
print("Real image")
check.Fake_or_Real('source/real_dog.jpg')