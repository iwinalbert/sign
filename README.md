```powershell
# Enter the project directory.
cd edge-sign

# Create a lightweight isolated Python environment.
py -m venv .venv

# Activate the virtual environment on Windows.
.\.venv\Scripts\Activate.ps1

# Install Windows runtime dependencies.
pip install -r requirements-windows.txt

# Download the selected sign-language video dataset.
python -m scripts.download_hf_dataset    

# Extract hand landmarks from the downloaded videos.
python -m scripts.extract_landmarks    

# Train and save the gesture classifier model.
python -m scripts.train_model    
 
# Evaluate the trained gesture classifier.
python -m scripts.evaluate_model    

# Start the local sign-language translator.
python run.py
```
