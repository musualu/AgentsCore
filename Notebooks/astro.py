import os
import swisseph as swe
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

JOSEFS_DATEN = {
    "name": "Josef Wertschulte",
    "geburtsdatum": (1948, 4, 16, 16, 47, 50),
    "ort": "Warstein"
}

def berechne_aspekte(jd1, jd2):
    planeten = [swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS]
    score = 0
    aspekt_text = []
    for planet in planeten:
        p1 = swe.calc_ut(jd1, planet)[0]
        p2 = swe.calc_ut(jd2, planet)[0]
        winkel = abs(p1 - p2) % 360

        if 0 <= winkel < 6:  # Konjunktion
            score += 3 if planet in [swe.VENUS, swe.JUPITER] else 1
            aspekt_text.append(f"Konjunktion {swe.get_planet_name(planet)}")
        elif 60 <= winkel < 66:  # Sextil
            score += 2
            aspekt_text.append(f"Sextil {swe.get_planet_name(planet)}")
        elif 120 <= winkel < 126:  # Trigon
            score += 3
            aspekt_text.append(f"Trigon {swe.get_planet_name(planet)}")
        elif 90 <= winkel < 96:  # Quadrat
            score -= 2
            aspekt_text.append(f"Quadrat {swe.get_planet_name(planet)}")

    return score, ", ".join(aspekt_text)

astrologe = Agent(
    role="Meisterastrologe",
    goal="Präzise astrologische Analysen erstellen",
    backstory="Mit 40 Jahren Erfahrung in klassischer und moderner Astrologie",
    verbose=True
)

analyst = Agent(
    role="Beziehungsanalyst",
    goal="Psychologische und strategische Beziehungsanalysen",
    backstory="Experte für zwischenmenschliche Dynamiken und Konfliktlösung",
    verbose=True
)

def astro_analyse(context):
    target_date = tuple(map(int, context["zielperson"]["geburtsdatum"]))
    jd_josef = swe.julday(*JOSEFS_DATEN["geburtsdatum"])
    jd_target = swe.julday(*target_date)
    score, aspekte = berechne_aspekte(jd_josef, jd_target)
    return {"score": score, "aspekte": aspekte}

def generiere_bericht(context):
    """
    Wir erzwingen ein ganz bestimmtes Format: 
      *Keine* Einleitung
      *Keine* extra Überschriften
      Nur 3 Punkte:
       1. Grundsätzlich
       2. Alltag
       3. Aktuell
    """
    astro_data = context["astro_results"]
    ziel = context["zielperson"]

    # Geburtsdaten (falls du sie anzeigen möchtest)
    geb_string = (
        f"{ziel['geburtsdatum'][2]:02d}.{ziel['geburtsdatum'][1]:02d}.{ziel['geburtsdatum'][0]}, "
        f"{ziel['geburtsdatum'][3]:02d}:{ziel['geburtsdatum'][4]:02d} Uhr"
    )

    prompt = f"""
Bitte gib deine astrologische Analyse *ausschließlich* in diesem Format zurück:

Analyse der Beziehung zwischen {JOSEFS_DATEN["name"]} \
({JOSEFS_DATEN["geburtsdatum"][2]:02d}.{JOSEFS_DATEN["geburtsdatum"][1]:02d}.{JOSEFS_DATEN["geburtsdatum"][0]}, \
{JOSEFS_DATEN["geburtsdatum"][3]:02d}:{JOSEFS_DATEN["geburtsdatum"][4]:02d} Uhr, {JOSEFS_DATEN["ort"]}) \
und {ziel["name"]} ({geb_string}, {ziel["ort"]})

1. Grundsätzlich:
   Schreibe hier 3-5 Sätze.

2. Alltag:
   Schreibe hier 2-4 Sätze.

3. Aktuell:
   Schreibe hier 2-4 Sätze.

WICHTIG:
- Schreibe *keine* zusätzliche Zusammenfassung, keine anderen Überschriften.
- Verwende keine Einleitung oder Schluss außerhalb der 3 Abschnitte.
- Verwende *genau* die Überschriften wie oben, und nummeriere sie (1.), (2.), (3.).
- Erwähne höchstens *einmal* das Wort 'Analyse' im Einleitungssatz, sonst nirgends.

Astrologische Aspekte: {astro_data["aspekte"]}
Score: {astro_data["score"]}/15
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,          # Kein kreatives Ausschweifen
        max_tokens=500,
        frequency_penalty=1.0,  # Strengere Wiederholungs-Kontrolle
        presence_penalty=1.0    # Strengere Wiederholungs-Kontrolle
    )
    return response["choices"][0]["message"]["content"]

astro_task = Task(
    name="astro_task",
    description="Astrologische Aspektanalyse durchführen",
    agent=astrologe,
    function=astro_analyse,
    expected_output="{'score': int, 'aspekte': str}"
)

analysis_task = Task(
    name="analysis_task",
    description="Kompletten Beziehungsbericht generieren",
    agent=analyst,
    function=generiere_bericht,
    expected_output="str",
    context=[astro_task]
)

crew = Crew(
    agents=[astrologe, analyst],
    tasks=[astro_task, analysis_task],
    verbose=True
)

if __name__ == "__main__":
    zielperson = {
        "name": input("Name der Person: "),
        "geburtsdatum": input("Geburtsdatum (YYYY-MM-DD): ").split("-"),
        "geburtszeit": input("Geburtszeit (HH:MM): ").split(":"),
        "ort": input("Geburtsort: ")
    }

    zielperson["geburtsdatum"] = (
        [int(x) for x in zielperson["geburtsdatum"]]
        + [int(zielperson["geburtszeit"][0]),
           int(zielperson["geburtszeit"][1]),
           0]
    )

    ergebnis = crew.kickoff(inputs={"zielperson": zielperson})

    print("\n=== Ergebnis ===")
    print(ergebnis)
