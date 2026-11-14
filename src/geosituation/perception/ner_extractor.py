import re
from typing import Dict, List, Set

class IncidentEntityExtractor:
    """
    Extracts situational contextual entities:
    - Involved Organizations / Public Services
    - Key Roles / Groups mentioned
    - Vehicles / Transport Modalities
    - Hazard Sources or Weapons
    - Core Action / Event Verb
    """

    def __init__(self):
        # Controlled dictionaries of key incident indicators
        self.organizations = [
            "noida police", "traffic police", "fire department", "fire tenders",
            "district administration", "noida authority", "fortis hospital", "felix hospital",
            "kailash hospital", "jaypee hospital", "metro rail", "nmrc", "dtc",
            "cyber crime cell", "special task force", "stf", "bomb disposal squad"
        ]
        
        self.vehicles = [
            "motorcycle", "bike", "scooter", "dumper truck", "dumper", "truck",
            "car", "suv", "sedan", "auto-rickshaw", "auto", "e-rickshaw",
            "tractor-trolley", "tractor", "bus", "van", "container truck"
        ]
        
        self.weapons_and_hazards = [
            "knife", "gun", "pistol", "firearm", "bullet", "iron rod", "stick", "lathi",
            "gas cylinder", "transformer", "chemical blaze", "ammonia gas", "short circuit",
            "blaze", "fire", "smoke", "explosive"
        ]
        
        self.roles_and_actors = [
            "assailants", "miscreants", "thieves", "robbers", "snatchers", "gang",
            "protesters", "farmers", "workers", "driver", "motorcyclist", "cyclist",
            "pedestrian", "victim", "delivery executive", "businessman", "senior citizen",
            "shopkeeper", "guard", "security personnel", "sub-inspector", "constable"
        ]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        
        found_orgs = set()
        for org in self.organizations:
            if re.search(rf'\b{re.escape(org)}\b', text_lower):
                found_orgs.add(org.title())
                
        found_vehicles = set()
        for veh in self.vehicles:
            if re.search(rf'\b{re.escape(veh)}\b', text_lower):
                found_vehicles.add(veh.title())
                
        found_instruments = set()
        for inst in self.weapons_and_hazards:
            if re.search(rf'\b{re.escape(inst)}\b', text_lower):
                found_instruments.add(inst.title())
                
        found_roles = set()
        for role in self.roles_and_actors:
            if re.search(rf'\b{re.escape(role)}\b', text_lower):
                found_roles.add(role.title())

        # Extract number of casualties / victims mentioned (regex pattern)
        casualties = []
        casualty_patterns = [
            r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:people|persons|victims|passengers)?\s*(?:dead|killed|succumbed|injured|hospitalized)',
            r'killed\s*(\d+|one|two|three|four|five)',
            r'injured\s*(\d+|one|two|three|four|five)'
        ]
        for pat in casualty_patterns:
            matches = re.findall(pat, text_lower)
            if matches:
                casualties.extend(matches)

        return {
            "organizations": sorted(list(found_orgs)),
            "vehicles": sorted(list(found_vehicles)),
            "instruments_or_hazards": sorted(list(found_instruments)),
            "roles_and_actors": sorted(list(found_roles)),
            "casualty_mentions": list(set(casualties))
        }
