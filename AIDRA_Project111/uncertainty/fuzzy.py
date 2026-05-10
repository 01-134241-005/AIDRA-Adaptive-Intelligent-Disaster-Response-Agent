import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyRiskSystem:
    def __init__(self):
        self.risk = ctrl.Antecedent(np.arange(0, 11, 1), 'risk')
        self.time = ctrl.Antecedent(np.arange(0, 61, 1), 'time')
        self.priority = ctrl.Consequent(np.arange(0, 11, 1), 'priority')
        
        self.risk['low'] = fuzz.trimf(self.risk.universe, [0,0,4])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [2,5,8])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [6,10,10])
        
        self.time['short'] = fuzz.trimf(self.time.universe, [0,0,20])
        self.time['medium'] = fuzz.trimf(self.time.universe, [10,30,50])
        self.time['long'] = fuzz.trimf(self.time.universe, [40,60,60])
        
        self.priority['low'] = fuzz.trimf(self.priority.universe, [0,0,4])
        self.priority['medium'] = fuzz.trimf(self.priority.universe, [2,5,8])
        self.priority['high'] = fuzz.trimf(self.priority.universe, [6,10,10])
        
        rules = [
            ctrl.Rule(self.risk['high'] & self.time['short'], self.priority['high']),
            ctrl.Rule(self.risk['high'] & self.time['long'], self.priority['high']),
            ctrl.Rule(self.risk['medium'] & self.time['medium'], self.priority['medium']),
            ctrl.Rule(self.risk['low'] & self.time['long'], self.priority['low']),
            ctrl.Rule(self.risk['low'] & self.time['short'], self.priority['medium']),
            ctrl.Rule(self.risk['high'], self.priority['high']),
            ctrl.Rule(self.risk['low'], self.priority['low'])
        ]
        self.ctrl_sys = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_sys)

    def compute(self, risk_val, time_val):
        self.sim.input['risk'] = risk_val
        self.sim.input['time'] = time_val
        self.sim.compute()
        return self.sim.output.get('priority', 5.0)   # FIXED: removed trailing 's'