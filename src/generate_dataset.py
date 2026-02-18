import pandas as pd
import numpy as np
import os

def generate_dataset():
    np.random.seed(42)

    samples = 300

    data = {
        "Worker_ID": np.random.randint(1, 61, samples),
        "Age": np.random.randint(20, 60, samples),
        "Temperature": np.random.uniform(28, 45, samples),
        "Humidity": np.random.uniform(35, 85, samples),
        "Heart_Rate": np.random.uniform(60, 130, samples),
        "Working_Hours": np.random.uniform(4, 12, samples)
    }

    df = pd.DataFrame(data)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/farm_worker_health.csv", index=False)

    print("✅ Dataset generated successfully with", len(df), "records")

if __name__ == "__main__":
    generate_dataset()
