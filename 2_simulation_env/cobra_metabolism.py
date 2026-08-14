import cobra
from cobra.io import load_model

class BiologicalEngine:
    def __init__(self, model_name="textbook"):
        # Loads the genome-scale metabolic network 
        # (e.g., the textbook core model or a proprietary iCHOv1 file).
        self.model_name = model_name
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

    def induce_metabolic_drift(self, target_reaction: str = "PDH", knock_down_fraction: float = 0.2):
        """
        Simulates genetic drift / epigenetic silencing by artificially restricting 
        an enzyme's maximum flux capacity.
        
        target_reaction: e.g., 'PDH' (Pyruvate Dehydrogenase) or 'PYK' (Pyruvate Kinase)
        knock_down_fraction: 0.0 (completely silenced) to 1.0 (fully active)
        """
        if target_reaction in self.model.reactions:
            rxn = self.model.reactions.get_by_id(target_reaction)
            # Scale upper bound to simulate enzyme suppression
            rxn.upper_bound = rxn.upper_bound * knock_down_fraction
            return True
        return False

    def reset_metabolic_drift(self):
        """Resets all reaction bounds to standard wild-type defaults."""
        self.model = load_model(self.model_name)


