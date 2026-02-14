Component	Description	                         Key Parameters
Bus	Nodes   (junctions)	                         v_nom (kV), x/y (coords for plotting) 
​
Line	    Transmission links	                 bus0/bus1, x (reactance pu), s_nom (MVA) 
​
Generator	Power sources	                     bus, p_nom (MW), marginal_cost ($/MWh), p_max_pu (availability) 
​
Load	    Demand	                             bus, p_set (MW demand) 
​
Link	    One-port devices (HVDC, batteries)	 bus0/bus1, p_nom, efficiency 
​
StorageUnit	Time-coupled storage	             bus, p_nom, max_hours (energy/MW), cyclic state-of-charge