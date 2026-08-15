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
        import warnings
        
        with self.model: # Context manager prevents permanent mutation of the base model
            # Apply dynamic environmental bounds
            for rxn_id, bound in constraints.items():
                if rxn_id in self.model.reactions:
                    # Negative bound denotes uptake from the environment
                    self.model.reactions.get_by_id(rxn_id).lower_bound = -abs(bound)
            
            # Catch and suppress the CobraPy infeasible warnings cleanly
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                solution = self.model.optimize()
            
            if solution.status == 'optimal':
                return {
                    "mu": solution.objective_value, # Specific growth rate (1/h)
                    "q_glucose": abs(solution.fluxes.get('EX_glc__D_e', 0.0)), # Uptake rate
                    # FIX: Sum the E. coli organic acid overflow pathways (Acetate, Formate, Lactate)
                    "q_lactate": (
                        solution.fluxes.get('EX_ac_e', 0.0) +      
                        solution.fluxes.get('EX_for_e', 0.0) +     
                        solution.fluxes.get('EX_lac__D_e', 0.0)    
                    )
                }
            else:
                # If the solver is infeasible, the cell cannot meet baseline ATP maintenance.
                # It is starving. We return a negative growth rate (cellular decay).
                return {"mu": -0.1, "q_glucose": 0.0, "q_lactate": 0.0}

    def induce_metabolic_drift(self, target_reaction: str = "PDH", knock_down_fraction: float = 0.2):
        """
        Simulates genetic drift / epigenetic silencing by artificially restricting 
        an enzyme's maximum flux capacity.
        
        target_reaction: e.g., 'PDH' (Pyruvate Dehydrogenase) or 'PYK' (Pyruvate Kinase)
        knock_down_fraction: 0.0 (completely silenced) to 1.0 (fully active)
        """
        if target_reaction in self.model.reactions:
            rxn = self.model.reactions.get_by_id(target_reaction)
            # FIX: Do not scale the default 1000.0 bound. 
            # Apply an absolute ceiling based on standard glucose flux (~10.0).
            # 10.0 * 0.15 = 1.5. This will severely choke the cell and force fermentation!
            rxn.upper_bound = 10.0 * knock_down_fraction
            return True
        return False

    def reset_metabolic_drift(self):
        """Resets all reaction bounds to standard wild-type defaults."""
        self.model = load_model(self.model_name)


