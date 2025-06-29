import torch

# Model
model=torch.load('best.pt')

# Images
imgs=['50.jpg']

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()
results.show()
results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]