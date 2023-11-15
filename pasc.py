import pandas as pd
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Información de estaciones
station_data = pd.read_csv("station_data.csv")

# Lista con valores de los meses
months = [f"0{m}" if m < 10 else str(m) for m in range(1, 13)]

# Variables
var_list = ["EMXP", "PRCP", "TAVG", "TMAX", "TMIN", "WDFG", "WSFG"]

def fill_in_missing_dates(df):
    """
    Añade fechas no registradas en las observaciones que se encuentran en el
    rango de 1950 a 2020.
    """
    for station_ID in station_data.ID:
        st = df[df.STATION == station_ID]
        # Se obtienen las fechas faltantes (MD)
        MD = [f"{y}-{m}" for y in range(1950,2021) for m in months if len(st[st.DATE == f"{y}-{m}"]) == 0]
        # Se añaden fechas faltantes a df
        for date in MD:
            df.loc[len(df)] = [station_ID, date, None, None, None, None, None, None, None]

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calcula la distancia entre dos puntos de coordenadas dados en grados decimales.
    
    Modificada del código: https://stackoverflow.com/a/19412565
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    return 2 * 6378.137 * np.arcsin(np.sqrt(a))

def distance(row1, row2):
    """
    Calcula la distancia entre las estaciones correspondientes a las dos filas
    de DataFrame dadas como entrada.
    """
    return np.linalg.norm([
        haversine_np(
            float(row1.LON),
            float(row1.LAT),
            float(row2.LON),
            float(row2.LAT),
        ),
        np.abs(float(row2.ELE)-float(row1.ELE)),
    ])

def idw(dist, vals):
    """
    Estima un valor segun interpolacion IDW tomando dist como el arreglo de las
    distancias al punto desconocido y vals como el arreglo de valores
    correspondientes con dichas distancias.
    """
    weight = dist/2
    weight /= weight.sum(axis=0)
    return np.dot(weight.T, vals)

def interpolate(df):
    """
    Rellena los datos faltantes del DataFrame por interpolacion.
    """
    for ir, a_row in df.iterrows():
        for col in var_list:
            if pd.isnull(a_row[col]) and (col not in ["WDFG", "WSFG"] or 1970 <= int(a_row.DATE[:4]) <= 1994):
                dist = []
                vals = []
                a_info = station_data[station_data.ID==a_row.STATION]
                for b_id in station_data.ID:
                    if b_id != a_row.STATION:
                        b_row = df[(df["STATION"]==b_id) & (df["DATE"]==a_row.DATE)][col]
                        if not pd.isnull(float(pd.to_numeric(b_row))):
                            dist.append(distance(a_info, station_data[station_data.ID==b_id]))
                            vals.append(float(b_row))
                df.loc[ir,col] = idw(np.array(dist), np.array(vals))
                
def MLR(df, var):
    """
    Crea un modelo MLR con las observaciones de df para predecir la variable
    indicada.

    Retorna los coeficientes de interés: determinación y correlación
    de Spearman.
    """
    df['YEAR'] = pd.to_numeric(df['DATE'].str[:4])
    df = pd.merge(df, station_data[['ID', 'LAT', 'LON', 'ELE']], left_on='STATION', right_on='ID', how='left')

    X = df[['LAT', 'LON', 'ELE', 'YEAR']]
    y = df[var]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    return np.array([
        model.score(X, y),
        spearmanr(y, y_pred).statistic
    ])

# Observaciones sin interpolar
df = pd.read_csv("observations.csv") #*

# Llena datos faltantes
fill_in_missing_dates(df) #*
interpolate(df) #*

# Si desea probar el resto del código sin tener que esperar la interpolación
# (proceso tardado), comente las líneas señaladas "#*" y en su lugar ejecute
# la línea siguiente (debe contar con el archivo observations_interpolated.csv)
# df = pd.read_csv("observations_interpolated.csv")

# eval_metrics almacenará estadísticos de coef. de determinación y de correlación
# para el modelo de cada variable para cada rango de años
eval_metrics = {v: np.zeros(shape=(7,2)) for v in var_list[:-2]}
eval_metrics["WDFG"], eval_metrics["WSFG"] = np.zeros(shape=(5,2)), np.zeros(shape=(5,2))

print("\n\nEVALUACIÓN DE MODELOS DE REG.")
for v in var_list:
    print("\n"+v+":\nRango de años\tC.Det.\tC.Corr.")
    
    if v in ["WDFG", "WSFG"]:
        # "WDFG" y "WSFG" se evaluarán en rangos de 5 años por la escasez de datos
        li_start = 1970
        li_end = 1995
        jump = 5
    else:
        # Las demás variables se evaluarán en rangos de 10 años
        li_start = 1950
        li_end = 2020
        jump = 10

    for li in range(li_start, li_end, jump):
        # ind es el índice del rango de años en el arreglo eval_metrics[v]
        ind = (li-li_start)//jump

        # subdf contiene las filas de df del rango de años actual
        subdf = df[(pd.to_numeric(df.DATE.str[:4]) <= li+jump) & (pd.to_numeric(df.DATE.str[:4]) >= li)]
        if v in ["WDFG", "WSFG"]:
            subdf = subdf.dropna(axis=0)

        # Se crea un modelo por cada mes para ser más precisos; los estadísticos
        # del rango de años serán el promedio de los estadísticos de cada mes
        for m in months:
            eval_metrics[v][ind] = eval_metrics[v][ind] + MLR(subdf[(subdf.DATE.str[-2:] == m)].copy(), v)
        eval_metrics[v][ind] = eval_metrics[v][ind]/12
        print(f"{li}-{li+jump}\t{round(eval_metrics[v][ind][0],3)}\t{round(eval_metrics[v][ind][1],3)}")

# Para los coeficientes obtenidos, se grafica por cada variable
for ci in range(2):
    fig, ax = plt.subplots(4, 2, figsize=(6,8))
    fig.tight_layout(pad=4.0)
    ax[3, 1].remove()

    li_start = 1950
    li_end = 2020
    jump = 10
    for i in range(4):
        for j in range(2):
            if 2*i+j > 6:
                break
            elif 2*i+j > 4:
                li_start = 1970
                li_end = 1995
                jump = 5
            ax[i, j].plot(eval_metrics[var_list[2*i+j]][:,ci])
            ax[i, j].set_xticks(
                range((li_end-li_start)//jump),
                labels=[f"{li}-{li+jump}" for li in range(li_start, li_end, jump)],
                rotation=45,
            )
            ax[i, j].set_title(var_list[2*i+j])

    fig.subplots_adjust(hspace=0.98)

    if ci == 0:
        fig.suptitle("Coeficiente de determinación vs. Rango de años", fontsize="medium")
        fig.savefig("Coeficiente_determinación_VS_Rango_años")
    else:
        fig.suptitle("Coeficiente de correlación de Spearman vs. Rango de años", fontsize="medium")
        fig.savefig("Coeficiente_correlación_Spearman_VS_Rango_años")
plt.show()

# Se aplica la prueba de MK sobre las series temporales de los coeficientes 
print("\n\nTEST MANN-KENDALL")
for v in var_list:
    print("\n"+v+":")
    print("Coef.Det.",round(mk.original_test(eval_metrics[v][:,0]).z,3))
    print("Coef.Corr.",round(mk.original_test(eval_metrics[v][:,1]).z,3))
