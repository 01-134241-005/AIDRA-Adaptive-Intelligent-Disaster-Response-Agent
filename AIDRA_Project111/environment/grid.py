import csv
import random
import os

def create_grid():
    # S = start, X = blocked, R = hazard, H = hospital
    grid = [
        ['S', '.', '.', 'R', '.'],
        ['.', 'X', '.', 'R', '.'],
        ['.', '.', '.', '.', '.'],
        ['R', '.', '.', 'X', '.'],
        ['.', '.', '.', '.', 'H']
    ]
    return grid


def get_victims():
    grid = create_grid()
    victims = []

    # Required victim counts per project spec
    required_counts = {
        "Critical": 2,
        "Moderate": 2,
        "Minor": 1
    }

    current_counts = {
        "Critical": 0,
        "Moderate": 0,
        "Minor": 0
    }

    used_positions = set()

    # Bug 1 Fix: use relative path so it works on any machine
    file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'synthetic_medical_triage.csv')

    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:
                severity = row["triage"]

                # Stop when enough victims collected
                if current_counts == required_counts:
                    break

                # Skip severities we don't need
                if severity not in required_counts:
                    continue

                if current_counts[severity] >= required_counts[severity]:
                    continue

                # Generate valid random grid position
                attempts = 0
                while attempts < 100:
                    x = random.randint(0, 4)
                    y = random.randint(0, 4)
                    # Avoid blocked/start/hospital/repeated cells
                    if grid[x][y] not in ['X', 'S', 'H'] and (x, y) not in used_positions:
                        used_positions.add((x, y))
                        break
                    attempts += 1
                else:
                    continue  # couldn't place this victim, skip

                victims.append(((x, y), severity))
                current_counts[severity] += 1

    except FileNotFoundError:
        pass  # fall through to fallback below

    # Fallback: if CSV couldn't fill all slots, pad with random placements
    needed = [sev for sev, cnt in required_counts.items()
              for _ in range(cnt - current_counts[sev])]
    random.shuffle(needed)
    for sev in needed:
        attempts = 0
        while attempts < 200:
            x = random.randint(0, 4)
            y = random.randint(0, 4)
            if grid[x][y] not in ['X', 'S', 'H'] and (x, y) not in used_positions:
                used_positions.add((x, y))
                victims.append(((x, y), sev))
                break
            attempts += 1

    return victims
