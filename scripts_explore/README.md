# Changelog

v5.2: modification in importation function calculation, and calculation fixes to post-simulation script
- Importation function calculation: we rearranged the calculation steps of infectious Omicron persons, such that the number of Omicron-infectious persons are first determined from CSSE's SA daily reported cases, and then the 'infectious presence' rolling window calculation is applied. Originally, the rolling window operation is added to the CSSE daily reported cases. 
- Calculation fixes to post-simulation script: 
v5.1: tweak in importation share calculation.
- Originally, we scale (ie. 15 Omicron positive on 26 Nov 2021) only for direct importation, with the indirect importation as an add-on. This creates a situation that the importation rate prior to flight bans are not consistent. The updated version considers all importation (direct and indirect streams) to be part of the scaling target, so for an import of 15 pax on 26 November 2021, eg. ~90% would be for direct flights, 10% would be from indirect flights. This may be inconsistent with the actual case though, given that the determined 15 scaling is for the direct flights from SA to NL. 
v5.0: addition of multiprocessing script, sensitivity analysis module, and tweaks to indirect importation math
- (TODO: elaborate)

v4.1: addition of extra model output metrics to determine contributing factor of indirect/direct imports on epidemic 

v4: added indirect and extra importation

v3: added (serial approach) 


v1:
