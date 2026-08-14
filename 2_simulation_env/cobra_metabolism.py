import cobra
from cobra.io import load_model

class BiologicalEngine:
    def __init__(self, model_name="textbook"):
        #Loads the genome-scale metabolic network 
        #(e.g., the textbook core model or a proprietary iCHOv1 file).
        self.model = load_model(model_name)
        
    def solve_fba(self, constraints: dict) -> dict:
        """
        Runs Flux Balance Analysis based on current macroscopic environmental constraints
        (how much food is available) and runs a linear programming optimization to maximize
        cell growth.

        constraints: Dictionary mapping reaction IDs to upper bounds.
        """
        with self.model: # Context manager prevents permanent mutation of the base model
            # Apply dynamic environmental bounds
            for rxn_id, bound in constraints.items():
                if rxn_id in self.model.reactions:
                    # Negative bound denotes uptake from the environment
                    self.model.reactions.get_by_id(rxn_id).lower_bound = -abs(bound)
            
            # Maximize biomass production
            solution = self.model.optimize()
            
            if solution.status == 'optimal':
                return {
                    "mu": solution.objective_value, # Specific growth rate (1/h)
                    "q_glucose": abs(solution.fluxes.get('EX_glc__D_e', 0.0)), # Uptake rate
                    "q_lactate": solution.fluxes.get('EX_lac__D_e', 0.0)       # Excretion rate
                }
            else:
                return {"mu": 0.0, "q_glucose": 0.0, "q_lactate": 0.0}

