# ⚽ European Football Analysis (2012-2023)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![StatsModels](https://img.shields.io/badge/StatsModels-0.13%2B-red)

Análisis avanzado de partidos de fútbol europeo con modelos estadísticos y recomendaciones tácticas basadas en datos.

## 📁 Estructura del Repositorio


## 🔧 Instalación
```bash
git clone https://github.com/Theog-go/euro-football-analysis.git
cd euro-football-analysis
pip install -r requirements.txt

# Cargar datos limpios
df = pd.read_csv('data/processed/cleaned_matches.csv')

# Generar recomendaciones para un equipo
from src.models import TeamAnalyzer
analyzer = TeamAnalyzer(df)
analyzer.recommend_for_team('Barcelona')


---

### ✨ Bonus: Archivo requirements.txt
