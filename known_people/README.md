# Known People

This directory contains the images that the peogram will match to during runtime.

# Content

This directory should be populated with images of people that you want to recognize. At runtime, `facial_recognition.py` will extract faces from the images, calculate an embedding vector for them, and store them in memory. When a new face comes in to be recognized, it will be compared to each one of the facial embeddings calculated previously, thus reducing the computation time.

