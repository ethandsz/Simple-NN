import imageio
import os

# Create a video from saved images
def create_video():
    images = []
    for epoch in range(1000):  # Adjust this if you have a different number of epochs
        img_path = f'figs/epoch{epoch}.png'
        images.append(imageio.imread(img_path))

    # Save the video
    imageio.mimwrite('training_process.mp4', images, fps=60)  # Adjust fps as needed

# Run the video creation
create_video()