import time
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INGEST_URL = "http://localhost:8080/ingest"

def send_telemetry(biomass: float, glucose: float, lactate: float, url: str = INGEST_URL, timeout: float = 0.05) -> bool:
    """
    Sends clean bioreactor state variables to the Go Edge Ingestion Microservice.
    
    Parameters:
        biomass (float): Biomass concentration (g/L)
        glucose (float): Glucose concentration (g/L)
        lactate (float): Lactate concentration (mmol/L)
        url (str): Target HTTP endpoint
        timeout (float): Max allowed wait time (seconds) to prevent blocking simulation loops
        
    Returns:
        bool: True if server acknowledged ingestion (202 Accepted / 200 OK), False otherwise.
    """
    payload = {
        "biomass": float(biomass),
        "glucose": float(glucose),
        "lactate": float(lactate)
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code in (200, 202):
            logger.debug(f"Telemetry sent successfully: {response.json()}")
            return True
        else:
            logger.warning(f"Ingestion server returned status {response.status_code}: {response.text}")
            return False
    except requests.exceptions.Timeout:
        logger.warning(f"Telemetry POST timed out (> {timeout}s). Continuing simulation loop.")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to ingestion microservice: {e}")
        return False

if __name__ == "__main__":
    import numpy as np
    from bioreactor_gym_env import BioreactorTwinEnv

    print("--- OmniCell-AI Ingestion Client Verification ---")
    print("Initializing Bio-Twin Gym Environment (dFBA + Euler physics)...")
    env = BioreactorTwinEnv()
    obs, _ = env.reset()

    print("Simulating telemetry dispatch from Bio-Twin over 5 time steps...")
    action = np.array([0.0, -1.0, -1.0], dtype=np.float32)

    for step in range(1, 6):
        obs, reward, terminated, truncated, _ = env.step(action)
        biomass, glucose, lactate = float(obs[0]), float(obs[1]), float(obs[2])
        
        success = send_telemetry(biomass, glucose, lactate)
        print(f"Step {step}: Biomass={biomass:.2f} g/L, Glucose={glucose:.2f} g/L, Lactate={lactate:.2f} mmol/L -> Sent: {success}")
        time.sleep(0.1)

