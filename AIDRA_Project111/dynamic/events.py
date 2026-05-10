
import random

def block_road(grid, victim_locs=None):
    """Block a random free cell. Avoids victim locations if provided."""
    protected = set(victim_locs) if victim_locs else set()
    for _ in range(20):
        x = random.randint(0, len(grid) - 1)
        y = random.randint(0, len(grid[0]) - 1)
        if grid[x][y] == '.' and (x, y) not in protected:
            grid[x][y] = 'X'
            return (x, y)
    return None  # No suitable cell to block