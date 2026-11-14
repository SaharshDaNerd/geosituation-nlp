import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from bs4 import BeautifulSoup

class TextPreprocessor:
    """
    Cleans raw news articles and extracts normalized temporal cues.
    """

    def __init__(self):
        # Patterns for relative temporal indicators
        self.time_of_day_patterns = {
            "late night": (23, 0),
            "midnight": (0, 0),
            "early morning": (5, 30),
            "morning": (9, 0),
            "noon": (12, 0),
            "afternoon": (14, 30),
            "evening": (18, 30),
            "night": (21, 0),
        }

    def clean_text(self, raw_html_or_text: str) -> str:
        """Removes HTML markup, extra whitespaces, boilerplate advertising tokens."""
        if not raw_html_or_text:
            return ""
        
        # Strip HTML if present
        if "<" in raw_html_or_text and ">" in raw_html_or_text:
            soup = BeautifulSoup(raw_html_or_text, "html.parser")
            # Remove scripts and styles
            for elem in soup(["script", "style", "aside", "nav", "footer"]):
                elem.extract()
            text = soup.get_text(separator=" ")
        else:
            text = raw_html_or_text

        # Remove typical Indian news dateline prefixes (e.g., "NOIDA: ", "GREATER NOIDA: ", "TNN | ")
        text = re.sub(r'^[A-Z\s]{2,15}\s*[:\-–|]\s*', '', text.strip())
        # Replace repetitive whitespaces/newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_temporal_anchor(self, text: str, ref_date: Optional[datetime] = None) -> Dict:
        """
        Parses relative temporal phrases ('yesterday', 'Monday night', 'late evening')
        relative to the publication date (ref_date).
        """
        if ref_date is None:
            ref_date = datetime.now()

        text_lower = text.lower()
        incident_time = ref_date
        confidence = 0.65
        indicator_found = "ARTICLE_PUBLICATION_TIMESTAMP"

        # Check relative day words
        if re.search(r'\byesterday\b', text_lower):
            incident_time = ref_date - timedelta(days=1)
            confidence = 0.88
            indicator_found = "YESTERDAY"
        elif re.search(r'\bday before yesterday\b', text_lower):
            incident_time = ref_date - timedelta(days=2)
            confidence = 0.90
            indicator_found = "DAY_BEFORE_YESTERDAY"
        elif re.search(r'\blast night\b', text_lower):
            incident_time = (ref_date - timedelta(days=1)).replace(hour=22, minute=0)
            confidence = 0.92
            indicator_found = "LAST_NIGHT"
        elif re.search(r'\bearly today\b|\bthis morning\b', text_lower):
            incident_time = ref_date.replace(hour=7, minute=30)
            confidence = 0.85
            indicator_found = "THIS_MORNING"

        # Check explicit hour/minute mention (e.g. "at 10:30 pm", "around 4 am")
        clock_match = re.search(r'\b(?:at|around|approx)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', text_lower)
        if clock_match:
            hours = int(clock_match.group(1))
            minutes = int(clock_match.group(2)) if clock_match.group(2) else 0
            meridiem = clock_match.group(3)
            
            if meridiem == "pm" and hours < 12:
                hours += 12
            elif meridiem == "am" and hours == 12:
                hours = 0
                
            incident_time = incident_time.replace(hour=hours, minute=minutes, second=0)
            confidence = min(0.95, confidence + 0.1)
            indicator_found += f"_CLOCK_TIME({hours:02d}:{minutes:02d})"
        else:
            # Check coarse time of day
            for tod_label, (h, m) in self.time_of_day_patterns.items():
                if tod_label in text_lower:
                    incident_time = incident_time.replace(hour=h, minute=m, second=0)
                    indicator_found += f"_{tod_label.upper().replace(' ', '_')}"
                    break

        return {
            "incident_timestamp": incident_time.isoformat(),
            "temporal_confidence": confidence,
            "temporal_indicator": indicator_found
        }
