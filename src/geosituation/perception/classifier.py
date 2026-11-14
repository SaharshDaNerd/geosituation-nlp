import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, f1_score

from geosituation.config import MODELS_DIR, INCIDENT_CATEGORIES

class IncidentClassifier:
    """
    Level 1 Perception: Categorizes raw news articles into standardized incident classes
    with calibrated probability estimates using a Word+Subword Feature Union and Modified Huber Loss.
    """

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = MODELS_DIR / "incident_classifier.joblib"
            
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.categories = list(INCIDENT_CATEGORIES.keys())
        
        if self.model_path.exists():
            self.load_model()
        else:
            self.train_seed_model()

    def train_seed_model(self) -> None:
        """
        Trains a high-precision calibrated classifier using Word (1-2) + Character-WB (3-5)
        n-gram feature union and SGD with modified huber loss for smooth calibrated probabilities.
        """
        seed_data, seed_labels = self._get_seed_training_data()
        
        feature_union = FeatureUnion([
            ('word_tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=4000
            )),
            ('char_tfidf', TfidfVectorizer(
                ngram_range=(3, 5),
                analyzer='char_wb',
                sublinear_tf=True,
                max_features=6000
            ))
        ])

        self.pipeline = Pipeline([
            ('features', feature_union),
            ('clf', SGDClassifier(
                loss='modified_huber',
                penalty='l2',
                alpha=1e-4,
                max_iter=1500,
                random_state=42
            ))
        ])
        
        self.pipeline.fit(seed_data, seed_labels)
        self.save_model()

    def predict(self, text: str) -> Dict:
        """
        Predicts incident class and returns the winning category, probability, and full distribution.
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded or trained.")
            
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        
        best_idx = np.argmax(probs)
        predicted_class = classes[best_idx]
        confidence = float(probs[best_idx])
        
        dist = {cls: float(prob) for cls, prob in zip(classes, probs)}
        
        return {
            "predicted_category": predicted_class,
            "category_name": INCIDENT_CATEGORIES.get(predicted_class, {}).get("name", predicted_class),
            "confidence": round(confidence, 4),
            "distribution": dist,
            "severity_weight": INCIDENT_CATEGORIES.get(predicted_class, {}).get("severity_weight", 1.0)
        }

    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluates model performance and returns metrics dictionary."""
        preds = self.pipeline.predict(texts)
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro')
        report = classification_report(true_labels, preds, output_dict=True)
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "detailed_report": report
        }

    def save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def load_model(self) -> None:
        self.pipeline = joblib.load(self.model_path)

    def _get_seed_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Curated domain seed dataset capturing Indian English journalistic conventions
        across the 5 incident classes and 1 non-incident control class.
        """
        samples = [
            # VIOLENT_CRIME (Assault, armed robbery, homicide, gunfire, stabbing, kidnap)
            ("A 32-year-old delivery executive was stabbed to death by three bike-borne assailants during an attempted robbery", "VIOLENT_CRIME"),
            ("Gunfire erupted outside a commercial complex leaving two security guards critically injured with gunshot wounds", "VIOLENT_CRIME"),
            ("Police arrested a gang of four miscreants who abducted a businessman and demanded a ransom of fifty lakh rupees", "VIOLENT_CRIME"),
            ("A woman was brutally assaulted and attacked with an iron rod near the metro station late Monday night", "VIOLENT_CRIME"),
            ("Double murder mystery: elderly couple found dead with slit throats inside their duplex apartment", "VIOLENT_CRIME"),
            ("Shooting reported near the market plaza as rival groups clashed over parking disputes, three hospitalized", "VIOLENT_CRIME"),
            ("A minor girl was rescued after being kidnapped from outside her tuition centre in an auto-rickshaw", "VIOLENT_CRIME"),
            ("Armed robbers stormed into a jewelry showroom at gunpoint, threatening the staff before looting gold", "VIOLENT_CRIME"),
            ("Man beaten to death with sticks and lathis outside restaurant following heated brawl", "VIOLENT_CRIME"),
            ("Police encounter: wanted criminal injured in exchange of fire after refusing to stop at barricade", "VIOLENT_CRIME"),
            ("Homicide squad investigates murder of factory supervisor found with multiple stab wounds", "VIOLENT_CRIME"),
            ("Three armed men barged into bank branch brandishing pistols and held staff hostage", "VIOLENT_CRIME"),
            ("Youth attacked with sharp-edged weapons outside mall over old rivalry, condition critical", "VIOLENT_CRIME"),
            ("Gang war on highway leaves two gangsters dead after twenty rounds fired from pistols", "VIOLENT_CRIME"),
            ("Contract killer arrested in connection with targeted assassination of local contractor", "VIOLENT_CRIME"),
            ("Kidnappers arrested after chasing vehicle near toll plaza, victim rescued safely", "VIOLENT_CRIME"),

            # PROPERTY_CRIME (Theft, burglary, snatching, scam, fraud, pickpocketing)
            ("Bike-borne snatchers grabbed a gold chain and iPhone from a woman on her evening walk before speeding off", "PROPERTY_CRIME"),
            ("Burglaries on the rise: thieves broke into three locked flats in the high-rise society and stole cash and ornaments", "PROPERTY_CRIME"),
            ("Police busted an inter-state car theft gang and recovered twelve stolen luxury SUVs with forged number plates", "PROPERTY_CRIME"),
            ("Cyber crime cell arrested two men operating a fake call centre defrauding senior citizens of their pension savings", "PROPERTY_CRIME"),
            ("Shoplifter caught red-handed attempting to steal expensive electronics from the mall basement store", "PROPERTY_CRIME"),
            ("A retired professor fell prey to an electricity bill update scam and lost three lakh rupees within minutes", "PROPERTY_CRIME"),
            ("Two men on a motorcycle snatched a handbag containing laptop and wallet outside the IT park gate", "PROPERTY_CRIME"),
            ("Thieves decamped with copper wires and transformer components from an electrical sub-station", "PROPERTY_CRIME"),
            ("Pickpockets active at crowded metro station: three arrested with ten stolen mobile phones", "PROPERTY_CRIME"),
            ("Car lifters stole motorcycle parked outside market using master key within minutes", "PROPERTY_CRIME"),
            ("House break-in reported in residential colony while family was attending wedding out of town", "PROPERTY_CRIME"),
            ("ATM fraud: cyber criminals installed skimming device to clone debit cards and withdraw cash", "PROPERTY_CRIME"),
            ("Two women held for snatching gold mangalsutra from elderly devotee inside temple premises", "PROPERTY_CRIME"),
            ("Factory inventory stolen overnight as night guards tied up by masked burglary gang", "PROPERTY_CRIME"),
            ("Laptop bag snatched from commuter waiting for cab on expressway service road", "PROPERTY_CRIME"),
            ("High-tech gang stealing wheels and catalytic converters of parked luxury cars busted", "PROPERTY_CRIME"),

            # TRAFFIC_INCIDENT (Accident, collision, pileup, hit-and-run, rollover)
            ("A speeding dumper truck collided with a hatchback on the expressway, killing two passengers and injuring three", "TRAFFIC_INCIDENT"),
            ("Major hit and run: an overspeeding luxury sedan rammed into a cyclist and fled towards the toll plaza", "TRAFFIC_INCIDENT"),
            ("Dense morning fog triggered a seven-vehicle pileup on the arterial highway, causing severe traffic paralysis", "TRAFFIC_INCIDENT"),
            ("A DTC bus overturned after losing control while navigating a sharp turn near the roundabout, 15 injured", "TRAFFIC_INCIDENT"),
            ("Massive traffic jam stretching over four kilometers reported due to a broken-down container truck on the flyover", "TRAFFIC_INCIDENT"),
            ("Motorcyclist succumbed to head injuries after being hit by a commercial tractor-trolley at the intersection", "TRAFFIC_INCIDENT"),
            ("Multi-car collision on the link road leaves five vehicles badly crushed during rush hour commute", "TRAFFIC_INCIDENT"),
            ("Auto-rickshaw rammed into a road divider after brake failure, driver and two commuters rushed to hospital", "TRAFFIC_INCIDENT"),
            ("Overspeeding truck overturned on central verge, spilling cargo and blocking two lanes of traffic", "TRAFFIC_INCIDENT"),
            ("Pedestrian crushed to death by speeding private bus while crossing arterial road near flyover", "TRAFFIC_INCIDENT"),
            ("Head-on collision between school van and delivery canter leaves seven children with minor injuries", "TRAFFIC_INCIDENT"),
            ("Biker skidded on wet road surface during morning drizzle and was hit by trailing car", "TRAFFIC_INCIDENT"),
            ("Drunk driving accident: SUV rammed into stationary police checkpost vehicle late Saturday night", "TRAFFIC_INCIDENT"),
            ("Traffic halted for three hours after oil tanker leaked diesel across both carriageways of the bridge", "TRAFFIC_INCIDENT"),
            ("Commercial tempo hit electric pole and toppled over near junction, driver trapped inside cabin", "TRAFFIC_INCIDENT"),
            ("Expressway collision: driver fell asleep at wheel and rammed rear of stationary container", "TRAFFIC_INCIDENT"),

            # CIVIC_HAZARD (Fire, explosion, structural collapse, gas leak, short-circuit)
            ("A massive fire broke out on the third floor of an industrial packaging unit, sending thick plumes of toxic smoke", "CIVIC_HAZARD"),
            ("Ten fire tenders rushed to douse a chemical blaze in an electronic manufacturing factory warehouse", "CIVIC_HAZARD"),
            ("A portion of a newly constructed boundary wall collapsed following heavy monsoon rains, crushing two parked cars", "CIVIC_HAZARD"),
            ("Gas cylinder blast inside a local restaurant kitchen injured four cooks and shattered adjacent window panes", "CIVIC_HAZARD"),
            ("Severe waterlogging and electrical short circuit submerged the basement parking of a residential society", "CIVIC_HAZARD"),
            ("Underground transformer caught fire with loud explosions, triggering power outages across five sectors", "CIVIC_HAZARD"),
            ("Ammonia gas leak detected in an ice cold-storage facility, prompting precautionary evacuation of nearby residents", "CIVIC_HAZARD"),
            ("Fire department rescued twelve workers trapped inside a garment warehouse fire that took six hours to control", "CIVIC_HAZARD"),
            ("Blaze engulfed plastic recycling plant near industrial area, thick smoke visible from five kilometers", "CIVIC_HAZARD"),
            ("Roof of dilapidated godown collapsed during heavy thunderstorms, trapping three labourers under debris", "CIVIC_HAZARD"),
            ("Commercial building evacuated after fire started in server room due to electrical short circuit", "CIVIC_HAZARD"),
            ("Boiler explosion at food processing factory leaves five workers with severe burn injuries", "CIVIC_HAZARD"),
            ("Chemical fumes cause breathing difficulty for school students, hazardous waste dumping suspected", "CIVIC_HAZARD"),
            ("Fire in slum cluster gutting thirty jhuggis, eight fire engines deployed to douse raging flames", "CIVIC_HAZARD"),
            ("Pipeline burst flooded major underpass, disrupting vehicle movement and submerging cars", "CIVIC_HAZARD"),
            ("Transformer caught fire after lightning strike, electricity department working on emergency restoration", "CIVIC_HAZARD"),

            # PUBLIC_DISRUPTION (Protests, rallies, dharnas, strikes, road blockades)
            ("Farmers staged a massive tractor march and blocked both carriageways of the expressway demanding crop compensation", "PUBLIC_DISRUPTION"),
            ("Section 144 imposed across the district following widespread protests by factory workers over unpaid wages", "PUBLIC_DISRUPTION"),
            ("Commuters stranded as hundreds of transport union drivers staged a flash strike and sat on dharna at the junction", "PUBLIC_DISRUPTION"),
            ("Protesters burned effigies and blocked arterial roads for three hours, leading to severe diversions", "PUBLIC_DISRUPTION"),
            ("Police deployed water cannons to disperse an unruly demonstration outside the development authority office", "PUBLIC_DISRUPTION"),
            ("Rally of two hundred supporters caused major bottleneck near the collectorate gate during morning hours", "PUBLIC_DISRUPTION"),
            ("Citizens held a candlelight vigil and protest march against municipal negligence regarding open drains", "PUBLIC_DISRUPTION"),
            ("Trade unions announced a 24-hour total shutdown, halting commercial deliveries and auto services", "PUBLIC_DISRUPTION"),
            ("Homebuyers stage protest outside builder office demanding immediate delivery of pending flats", "PUBLIC_DISRUPTION"),
            ("Heavy police deployment along border checkpoint as agitating union threatens road blockade", "PUBLIC_DISRUPTION"),
            ("Anganwadi workers protest near collectorate seeking minimum wage hike and medical benefits", "PUBLIC_DISRUPTION"),
            ("Sanitation workers go on indefinite strike, garbage piles up across markets and residential sectors", "PUBLIC_DISRUPTION"),
            ("Students demonstrate outside university administrative block alleging irregularities in fee structure", "PUBLIC_DISRUPTION"),
            ("Traffic diverted as religious procession draws thousand devotees along main road corridor", "PUBLIC_DISRUPTION"),
            ("Unauthorized demonstration dispersed by security forces, sixteen protesters detained under police act", "PUBLIC_DISRUPTION"),
            ("Traders association shuts down commercial market for half day to protest municipal sealing drive", "PUBLIC_DISRUPTION"),

            # NON_INCIDENT (Normal news, governance, culture, sports, development)
            ("The development authority inaugurated a new multi-level parking facility with capacity for 1000 four-wheelers", "NON_INCIDENT"),
            ("Real estate summit discusses upcoming green commercial buildings and sustainable architectural standards", "NON_INCIDENT"),
            ("Local school students won national science exhibition with their solar-powered water purification model", "NON_INCIDENT"),
            ("Cultural festival celebrating regional folk art and culinary traditions kicked off with large attendance", "NON_INCIDENT"),
            ("Metro authority announced extended train timings for the upcoming festive weekend to ease passenger transit", "NON_INCIDENT"),
            ("Municipal corporation conducted an extensive tree plantation drive across green belts and public parks", "NON_INCIDENT"),
            ("Health department organizes a free blood donation and eye checkup camp for senior citizens", "NON_INCIDENT"),
            ("New public library and youth reading centre opened to promote digital literacy among students", "NON_INCIDENT"),
            ("Police commissioner reviewed security arrangements ahead of Independence Day celebrations in the district", "NON_INCIDENT"),
            ("Weather department forecasts partly cloudy skies with light breeze across the National Capital Region", "NON_INCIDENT"),
            ("City hospital introduces advanced robotic surgery unit to provide minimally invasive treatment options", "NON_INCIDENT"),
            ("Annual flower show and gardening competition begins at regional botanical park with fifty exhibits", "NON_INCIDENT"),
            ("District sports stadium hosts inter-college badminton tournament with participation from forty teams", "NON_INCIDENT"),
            ("Development authority launches online portal for hassle-free property map approval and utility connections", "NON_INCIDENT"),
            ("Solar rooftop initiative expands to cover government schools and community centers in rural belt", "NON_INCIDENT"),
            ("Heritage walk organized to educate young citizens on local historical monuments and archaeological sites", "NON_INCIDENT")
        ]
        
        texts = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        return texts, labels
