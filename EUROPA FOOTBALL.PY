# Europe-Football# Europe-Football

import pandas as pd
from pathlib import Path

# Ruta exacta del archivo
ruta_archivo = Path(r"C:\Users\theog\Downloads\Euro-Football_2012-2023.csv")

# Carga el archivo con pandas
try:
    df = pd.read_csv(ruta_archivo)
    print("✅ Archivo cargado correctamente")
    display(df.head(3))  # Muestra las primeras filas
except Exception as e:
    print(f"❌ Error al cargar: {str(e)}")

# Estadísticas básicas
print(df.describe(include='all'))

# Valores nulos
print(df.isnull().sum())

# Distribución de resultados
df['FTR'].value_counts().plot(kind='pie', autopct='%1.1f%%')

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Seleccionar variables predictoras
variables = df_clean[['HS', 'HST', 'HC', 'HY']].dropna()

# Calcular VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = variables.columns
vif_data["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
print("Factores de Inflación de Varianza (VIF):")
print(vif_data)

from statsmodels.formula.api import negativebinomial

modelo_nb = negativebinomial(
    formula='FTHG ~ HST + HY',  # Usar solo variables significativas y no colineales
    data=df_clean
).fit()
print(modelo_nb.summary())

from statsmodels.formula.api import negativebinomial

# Modelo con variables seleccionadas (eliminamos HS por colinealidad con HST)
modelo_nb = negativebinomial(
    formula='FTHG ~ HST + HC + HY',  # Fórmula ajustada
    data=df_clean.dropna(subset=['HST', 'HC', 'HY'])
).fit()

print("\nResultados del Modelo Binomial Negativo:")
print(modelo_nb.summary())

# Interpretación de coeficientes
coeficientes = modelo_nb.params
for var, coef in coeficientes.items():
    efecto = (np.exp(coef) - 1) * 100
    print(f"\n{var}:")
    print(f"  Coeficiente: {coef:.4f}")
    print(f"  Efecto porcentual: {efecto:.1f}%")

import numpy as np
import statsmodels.api as sm

# 1. Verificar y limpiar datos faltantes/infinitos
print("Valores nulos antes de limpiar:")
print(df_clean[['HST', 'HY']].isnull().sum())

print("\nValores infinitos antes de limpiar:")
print(df_clean[['HST', 'HY']].apply(lambda x: np.isinf(x).sum()))

# 2. Eliminar filas con problemas
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)  # Convertir infinitos a NaN
df_clean = df_clean.dropna(subset=['HST', 'HY', 'FTHG'])  # Eliminar filas con NaN

# 3. Verificar que los datos estén limpios
print("\nValores nulos después de limpiar:")
print(df_clean[['HST', 'HY']].isnull().sum())

# 4. Ajustar el modelo GLM con datos limpios
try:
    modelo_poisson = sm.GLM(
        df_clean['FTHG'],
        sm.add_constant(df_clean[['HST', 'HY']]),
        family=sm.families.Poisson()
    ).fit()
    
    print("\n" + "═"*50)
    print("MODELO AJUSTADO CORRECTAMENTE")
    print("═"*50)
    print(modelo_poisson.summary())
    
except Exception as e:
    print(f"\nError al ajustar el modelo: {str(e)}")
    print("\nRevisa que:")
    print("1. df_clean['HST'] y df_clean['HY'] no contengan nulos/infinitos")
    print("2. Los valores sean numéricos (no strings)")
    print("3. Las columnas existan en el DataFrame")


# Crear dataset con variables de ambos equipos
df_advanced = df_clean.copy()

# Variables del rival (visitante)
df_advanced['AST_rival'] = df_advanced['AST']  # Tiros al arco rival
df_advanced['AY_rival'] = df_advanced['AY']    # Tarjetas rival
df_advanced['AF_rival'] = df_advanced['AF']    # Faltas rival

# Eliminar filas con nulos en nuevas variables
df_advanced = df_advanced.dropna(subset=['AST_rival', 'AY_rival'])

print(f"Datos disponibles para análisis avanzado: {len(df_advanced):,} partidos")

modelo_avanzado = sm.GLM.from_formula(
    '''FTHG ~ HST + HY + AST_rival + AY_rival + 
        HST:HY + HST:AST_rival''',  # Interacciones clave
    data=df_advanced,
    family=sm.families.Poisson()
).fit()

print("\n" + "═"*80)
print("MODELO AVANZADO CON VARIABLES DEL RIVAL")
print("═"*80)
print(modelo_avanzado.summary())

# Extraer coeficientes y efectos
coefs = modelo_avanzado.params
effects = np.exp(coefs) - 1

print("\nEFECTOS PRINCIPALES (cambio porcentual):")
print(f"- Cada tiro al arco local (HST): {effects['HST']*100:.1f}%")
print(f"- Cada tarjeta amarilla local (HY): {effects['HY']*100:.1f}%")
print(f"- Cada tiro al arco rival (AST_rival): {effects['AST_rival']*100:.1f}%")
print(f"- Cada tarjeta amarilla rival (AY_rival): {effects['AY_rival']*100:.1f}%")

print("\nINTERACCIONES:")
print(f"- Efecto HST*HY: {effects['HST:HY']*100:.1f}%")
print(f"- Efecto HST*AST_rival: {effects['HST:AST_rival']*100:.1f}%")

from mpl_toolkits.mplot3d import Axes3D

# Configurar grid de valores
hst_values = np.linspace(0, 10, 5)
ast_rival_values = np.linspace(0, 10, 5)
hst_grid, ast_grid = np.meshgrid(hst_values, ast_rival_values)

# Calcular predicciones (con HY=0 y AY_rival=0 para simplificar)
goles_pred = np.exp(
    coefs['Intercept'] + 
    coefs['HST']*hst_grid + 
    coefs['AST_rival']*ast_grid +
    coefs['HST:AST_rival']*hst_grid*ast_grid
)

# Gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(hst_grid, ast_grid, goles_pred, cmap='viridis')
ax.set_xlabel('Tiros Locales (HST)')
ax.set_ylabel('Tiros Rival (AST_rival)')
ax.set_zlabel('Goles Predichos')
ax.set_title('Interacción HST-AST_rival en Goles Locales')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Calcular eficiencia ofensiva/defensiva por equipo
equipo_stats = df_advanced.groupby('HomeTeam').agg({
    'FTHG': 'mean',
    'HST': 'mean',
    'AST_rival': 'mean',
    'HY': 'mean'
}).sort_values('FTHG', ascending=False)

# Clasificación táctico
equipo_stats['Tipo'] = np.where(
    (equipo_stats['HST'] > equipo_stats['HST'].median()) & 
    (equipo_stats['AST_rival'] < equipo_stats['AST_rival'].median()),
    'Ofensivo',
    np.where(
        (equipo_stats['HST'] < equipo_stats['HST'].median()) & 
        (equipo_stats['AST_rival'] > equipo_stats['AST_rival'].median()),
        'Defensivo',
        'Mixto'
    )
)

print("\nTOP 10 EQUIPOS POR ESTRATEGIA:")
print(equipo_stats.head(10))

def recomendaciones_tacticas(equipo, df_stats):
    # Obtener estadísticas del equipo (el índice ahora contiene los nombres)
    stats = df_stats.loc[equipo]
    
    print(f"\nRECOMENDACIONES PARA {equipo.upper()} (Estilo: {stats['Tipo']})")
    print("═"*50)
    
    if stats['Tipo'] == 'Ofensivo':
        print(f"- Mantener presión ofensiva (Actual: {stats['HST']:.1f} tiros al arco/partido)")
        print(f"- Reducir tarjetas (Actual: {stats['HY']:.1f} amarillas/partido)")
    elif stats['Tipo'] == 'Defensivo':
        print(f"- Mejorar bloqueo de tiros rivales (Actual: {stats['AST_rival']:.1f} concedidos/partido)")
    else:
        print(f"- Optimizar equilibrio ofensivo/defensivo")
    
    print(f"\nObjetivo goles locales: {stats['FTHG']:.2f} → {stats['FTHG']*1.1:.2f} (+10%)")

# Versión alternativa si prefieres resetear el índice
equipo_stats_reset = equipo_stats.reset_index()

# Ejemplo de uso con el primer equipo (usando el índice)
recomendaciones_tacticas(equipo_stats.index[0], equipo_stats)

# O usando la versión con reset_index()
# recomendaciones_tacticas(equipo_stats_reset.iloc[0]['HomeTeam'], equipo_stats_reset)
def recomendaciones_tacticas_interactivo(df_stats):
    """
    Muestra recomendaciones tácticas para un equipo seleccionado por el usuario
    """
    # Mostrar lista de equipos disponibles
    print("Equipos disponibles:")
    print("-" * 30)
    for i, equipo in enumerate(df_stats.index.unique(), 1):
        print(f"{i}. {equipo}")
    print("-" * 30)
    
    # Solicitar selección
    while True:
        try:
            seleccion = int(input("\nIngrese el número del equipo a analizar (1-{}): ".format(len(df_stats))))
            equipo = df_stats.index[seleccion-1]
            break
        except (ValueError, IndexError):
            print("¡Entrada inválida! Por favor ingrese un número de la lista.")
    
    # Generar recomendaciones
    stats = df_stats.loc[equipo]
    
    print(f"\nRECOMENDACIONES PARA {equipo.upper()} (Estilo: {stats['Tipo']})")
    print("═"*50)
    
    if stats['Tipo'] == 'Ofensivo':
        print(f"- Mantener presión ofensiva (Actual: {stats['HST']:.1f} tiros al arco/partido)")
        print(f"- Reducir tarjetas (Actual: {stats['HY']:.1f} amarillas/partido)")
    elif stats['Tipo'] == 'Defensivo':
        print(f"- Mejorar bloqueo de tiros rivales (Actual: {stats['AST_rival']:.1f} concedidos/partido)")
    else:
        print(f"- Optimizar equilibrio ofensivo/defensivo")
    
    print(f"\nObjetivo goles locales: {stats['FTHG']:.2f} → {stats['FTHG']*1.1:.2f} (+10%)")

# Ejemplo de uso:
recomendaciones_tacticas_interactivo(equipo_stats)

# Fórmula para calcular tiros necesarios
def tiros_necesarios(goles_deseados, hy=0, ast_rival=0):
    return (np.log(goles_deseados) + 0.2496 + 0.0276*hy + 0.0275*ast_rival) / 0.1505

print(f"Tiros requeridos para 2 goles (con 3 HY y 8 AST_rival): {tiros_necesarios(2, 3, 8):.1f}")

