import gymnasium as gym
from gymnasium import spaces
import numpy as np
from cobra_metabolism import BiologicalEngine

class BioreactorTwinEnv(gym.Env):
    """
    OpenAI Gym interface for the Bio-Twin simulation.
    Maps RL actions to microfluidic pumps and calculates dFBA kinetics.
    """
    def __init__(self):
        super().__init__()
        self.bio_engine = BiologicalEngine("textbook")
        
        # Action Space: Defines the AI's "hands." Restricts the AI to outputting a 
        # continuous float between -1.0 and 1.0, which step() later translates into 
        # a physical feed pump rate (Liters/hour).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: Defines the AI's "eyes." Bounds the physical concentrations 
        # to realistic limits [Biomass (g/L), Glucose (g/L), Lactate (mmol/L)], representing 
        # what a physical Raman spectrometer could actually measure.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 100.0, 50.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.dt = 0.1 # Simulation time-step delta (hours)
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initial Macroscopic Tank State: Acts as the physical memory of the tank. 
        # Tracks the exact macroscopic concentrations of Biomass (X), Glucose (S), 
        # and Lactate (L) at any given millisecond.
        self.state = {
            "X": 0.1,    # Initial Biomass (g/L)
            "S": 20.0,   # Initial Glucose (g/L)
            "L": 0.0     # Initial Lactate (mmol/L)
        }
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Translate Action to Physics (Feed Pump L/h)
        # Map [-1, 1] to [0, 0.5] Liters/hour
        feed_rate = (action[0] + 1.0) * 0.25 
        feed_concentration = 100.0 # g/L in the feed tank
        
        # 2. Formulate Biological Constraints (Michaelis-Menten Kinetics proxy)
        # Substrate availability restricts maximum cellular uptake
        max_uptake = 10.0 * (self.state["S"] / (0.5 + self.state["S"]))
        constraints = {"EX_glc__D_e": max_uptake}
        
        # 3. Execute Biological FBA
        bio_rates = self.bio_engine.solve_fba(constraints)
        
        # 4. Execute Macroscopic Mass Balance (Euler Integration)
        # Applies differential mass-balance equations (dX, dS, dL), calculating how 
        # the tank's overall volume and chemical makeup change over time delta (self.dt) 
        # based on pump input and cellular metabolic rates.
        mu = bio_rates["mu"]
        q_s = bio_rates["q_glucose"]
        q_l = bio_rates["q_lactate"]
        
        dX = (mu * self.state["X"]) * self.dt
        dS = (-q_s * self.state["X"] + feed_rate * feed_concentration) * self.dt
        dL = (q_l * self.state["X"]) * self.dt
        
        self.state["X"] = np.clip(self.state["X"] + dX, 0.0, 50.0)
        self.state["S"] = np.clip(self.state["S"] + dS, 0.0, 100.0)
        self.state["L"] = np.clip(self.state["L"] + dL, 0.0, 50.0)
        
        # 5. Calculate Reward Function (The Incentive)
        # The mathematical incentive scorecard: grants points for high growth rate (mu) 
        # and deducts points for toxic lactate accumulation (self.state["L"]).
        reward = (mu * 10.0) - (self.state["L"] * 2.0)
        
        # Regulatory Guardrails (QbD Failure)
        # Catastrophic penalty trigger if tank breaches Quality by Design (QbD) boundaries.
        terminated = False
        if self.state["L"] > 25.0:
            reward -= 1000.0 # Catastrophic failure penalty
            terminated = True
            
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.state["X"], self.state["S"], self.state["L"]], dtype=np.float32)
