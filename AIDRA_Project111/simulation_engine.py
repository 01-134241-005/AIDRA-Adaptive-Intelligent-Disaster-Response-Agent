
# simulation_engine.py — step-based engine for Streamlit live animation
from search import astar

class SimulationEngine:
    def __init__(self, grid, victims, ambulances=2):
        self.grid    = grid
        # victims is list of ((x,y), severity)
        self.victims = list(victims)
        self.time    = 0
        self.rescued = 0
        self.ambulances = {
            f"A{i}": {"pos": (0, 0), "path": [], "carrying": 0}
            for i in range(ambulances)
        }

    def step(self, search_fn=None):
        if search_fn is None:
            search_fn = astar
        self.time += 1

        for amb_id, amb in self.ambulances.items():
            # Assign new victim if idle and victims remain
            if not amb["path"] and self.victims:
                target = self.victims.pop(0)
                path = search_fn(self.grid, amb["pos"], target[0])
                amb["path"] = path if path else []

            # Move one step along path
            if amb["path"]:
                next_pos = amb["path"].pop(0)
                amb["pos"] = next_pos
                # If reached end of path (destination), count rescue
                if not amb["path"]:
                    self.rescued += 1

    def grid_view(self):
        """Returns a 2-D list of display characters for Streamlit rendering."""
        view = [[cell for cell in row] for row in self.grid]

        # Mark victim positions
        for v in self.victims:
            x, y = v[0]
            if 0 <= x < len(view) and 0 <= y < len(view[0]):
                view[x][y] = 'V'

        # Mark ambulance positions
        for amb in self.ambulances.values():
            x, y = amb["pos"]
            if 0 <= x < len(view) and 0 <= y < len(view[0]):
                view[x][y] = '🚑'

        return view