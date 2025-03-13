# diagnostic.py
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import sklearn
    print("Scikit-learn version:", sklearn.__version__)
    
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error: {e}")