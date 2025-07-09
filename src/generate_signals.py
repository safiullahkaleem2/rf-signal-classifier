import numpy as np
import os

def generate_am(freq=1000, rate=10000, duration=1.0):
    t = np.arange(0, duration, 1/rate)
    carrier = np.cos(2 * np.pi * freq * t)
    mod = 1 + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    return mod * carrier

def generate_fm(freq=1000, rate=10000, duration=1.0):
    t = np.arange(0, duration, 1/rate)
    mod_signal = np.sin(2 * np.pi * 2 * freq * t)
    return np.cos(2 * np.pi * freq * t + 5 * mod_signal)

def generate_noise(rate=10000, duration=1.0):
    return np.random.normal(0, 1, int(rate * duration))

def save_signals():
    os.makedirs("data/signals", exist_ok=True)
    for label, gen_fn in zip(["AM", "FM", "Noise"], [generate_am, generate_fm, generate_noise]):
        for i in range(100):  # 100 samples each
            signal = gen_fn()
            np.save(f"data/signals/{label}_{i}.npy", signal)

if __name__ == "__main__":
    save_signals()
