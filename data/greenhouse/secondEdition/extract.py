import datetime
import pandas as pd
import torch

def load_data():
    train_teams = [
        "Reference",
        "Digilog",
        "IUACAAS",
        "Automatoes",
        "TheAutomators",
    ]

    val_teams = [
        "AICU"
    ]

    test_teams = [
        "Reference"
    ]

    train_X = None
    train_Y = None
    val_X = None
    val_Y = None
    test_X = None
    test_Y = None
    for team in train_teams:
        if train_Y is None:
            train_X, train_Y = extract_2nd_edition_train_data(team)
        else:
            X, Y = extract_2nd_edition_train_data(team)
            train_X = torch.cat([train_X,X])
            train_Y = torch.cat([train_Y,Y]) 
    for team in val_teams:
        if val_Y is None:
            val_X, val_Y = extract_2nd_edition_train_data(team)
        else:
            X, Y = extract_2nd_edition_train_data(team)
            val_X = torch.cat([val_X,X])
            val_Y = torch.cat([val_Y,Y])
    for team in test_teams:
        if test_Y is None:
            test_X, test_Y = extract_2nd_edition_train_data(team)
        else:
            X, Y = extract_2nd_edition_train_data(team)
            test_X = torch.cat([test_X,X])
            test_Y = torch.cat([test_Y,Y])
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def convert_excel_timestamp(excel_timestamp):
    """
    Converts an Excel serial date (days since 1899-12-30) 
    to a timezone-aware datetime object in UTC.
    """
    excel_epoch = datetime.datetime(1899, 12, 30, tzinfo=datetime.timezone.utc)
    dt_object = excel_epoch + datetime.timedelta(days=excel_timestamp)
    return dt_object


def load_climate_data(fp):
    cols = {"%time":"Time",
            "Tair":"Tair",
            "Rhair":"RH",
            "Tot_PAR":"PAR", # Total PAR includes sun. alternative column -> Tot_PAR_Lamps 
            "HumDef": "HumidityDeficit",
            "CO2air": "CO2air", 
            # "EC_drain_PC": "ECdrain", 
            # "pH_drain_PC": "pHdrain"
        }

    climate_df = pd.read_csv(fp, 
                             usecols=lambda x: x in cols.keys(),
                             low_memory=False,
                             )
    climate_df["%time"] = climate_df["%time"].transform(convert_excel_timestamp)
    climate_df.rename(cols,axis=1, inplace=True)

    df = climate_df
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype('float64')
    df.dropna(inplace=True)
    df["Time"] = df["Time"].astype("datetime64[ns]")
    df.index = df["Time"]
    df.drop('Time', axis=1, inplace=True)
    return df

def load_prod_data(fp):
    cols = {"%time":"Time",
            "ProdA": "ProdA", # kg/m2
            "ProdB": "ProdB", # kg/m2
            "Nr_fruits_ClassA":"nClassA",
            "Nr_fruits_ClassB":"nClassB",
            "Weight_fruits_ClassA":"gClassA", # total weight in grams for 10 samples
            "Weight_fruits_ClassB":"gClassB", # total weight in grams for 10 samples
        }

    prod_df = pd.read_csv(fp, usecols=lambda x: x in cols.keys())
    prod_df["%time"] = prod_df["%time"].transform(convert_excel_timestamp)
    prod_df.rename(cols,axis=1, inplace=True)

    prod_df.index = prod_df["Time"].dt.date.astype("datetime64[ns]")
    prod_df = prod_df[1:]

    #TODO: Verify this for other teams
    #Reference dataset is dirty, filling in values manually. 
    prod_df.fillna(0, inplace=True)

    prod_df["DAP"] = (prod_df.index - prod_df.index.min()).days
    return prod_df

def load_tomato_data(fp):
    dmc_col = "dryMatterPercent"
    cols = {"%time":"Time",
            "Weight":"avgFruitWeight",
            "DMC_fruit": dmc_col
        }
    
    df = pd.read_csv(fp, usecols=lambda x: clean(x) in cols.keys())
    df.columns = [clean(c) for c in df.columns]
    df["%time"] = df["%time"].transform(convert_excel_timestamp)
    df.rename(cols,axis=1, inplace=True)
    df.replace(r'\s+','', regex=True)
    df[dmc_col] = pd.to_numeric(df[dmc_col], errors='coerce')
    df.dropna(inplace=True)
    df.index = df["Time"].dt.date.astype("datetime64[ns]")
    df.drop('Time', axis=1, inplace=True)
    return df

def load_parameter_data(fp):
    cols={
        "%Time":"Time",
        "stem_dens":"stem_density",
        "plant_dens":"plant_density"
    }

    df = pd.read_csv(fp, usecols=lambda x: clean(x) in cols.keys())
    df.columns = [clean(c) for c in df.columns]
    df["%Time"] = df["%Time"].transform(convert_excel_timestamp)
    df.rename(cols,axis=1, inplace=True)
    df.replace(r'\s+','', regex=True)
    df.dropna(inplace=True)
    df.index = df["Time"].dt.date.astype("datetime64[ns]")
    df.drop('Time', axis=1, inplace=True)
    return df

def clean(str):
    return str.strip().replace(r'\s+','')

def extract_2nd_edition_climate_data(fp):
    df = load_climate_data(fp)
    TBase = 10
    n_total = df["PAR"].resample('D').count()
    # n_total = df["PAR"].groupby(df["Time"].dt.date).count()
    # n_day = df["PAR"].where(df["PAR"]>0).groupby(df["Time"].dt.date).count() # readings when crop were in the light (day)
    n_day = df["PAR"].where(df["PAR"]>0).resample('D').count()
    light_hours = (n_day / n_total) * 24
    # PPFD_mean = df["PAR"].groupby(df["Time"].dt.date).mean()
    PPFD_mean = df["PAR"].resample('D').mean()
    dli = 3.6 * 10**-3 * PPFD_mean * light_hours

    daily_temp = df["Tair"].resample('D').mean()
    gdd_daily = (daily_temp - TBase).clip(lower=0)

    cumulative_dli = dli.cumsum()
    cumulative_gdd = gdd_daily.cumsum()
    
    climate_df = pd.DataFrame({"GDD":gdd_daily,"GDD_Sum":cumulative_gdd,"DLI":dli, "DLI_Sum":cumulative_dli})
    climate_df["DAP"] = range(1, len(climate_df)+1)

    return climate_df

def extract_2nd_edition_production_data(fp):
    prod_df = load_prod_data(fp)

    df = pd.DataFrame({
        "N": prod_df["nClassA"],
        "N_Sum":prod_df["nClassA"].cumsum(),
        "Yield":prod_df["gClassA"], 
        "Yield_Sum":prod_df["gClassA"].cumsum(),
        })
    
    return df

def extract_2nd_edition_data(team="Reference"):
    data_dir = "data/greenhouse/secondEdition"
    climate_file = f"{team}/GreenhouseClimate.csv"
    production_file = f"{team}/Production.csv"

    climate_df = extract_2nd_edition_climate_data(f"{data_dir}/{climate_file}")
    prod_df = extract_2nd_edition_production_data(f"{data_dir}/{production_file}")

    df = pd.merge(climate_df, prod_df, on="Time", how="inner")

    return df

def extract_2nd_edition_train_data(team="Reference"):
    df = extract_2nd_edition_data(team)

    train_X = torch.tensor(df[["DAP", "GDD_Sum", "DLI_Sum"]].values)
    train_Y = torch.tensor(df[["Yield_Sum"]].values)
    return train_X, train_Y
