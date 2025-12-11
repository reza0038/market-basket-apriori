import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# load dataset
df = pd.read_csv("bread basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

df["month"] = df["date_time"].dt.month
df["day"] = df["date_time"].dt.weekday

df["month"].replace(
    [i for i in range(1, 13)],
    ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
     "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
    inplace=True
)

df["day"].replace(
    [i for i in range(7)],
    ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
    inplace=True
)

st.title("Analisis Market Basket Menggunakan Algoritma Apriori")

# ----------------------------
# FUNCTIONS
# ----------------------------

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) &
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result"


def user_input_features():
    item = st.selectbox(
        "Item",
        ['Afternoon with the baker', 'Alfajores','Argentina Night','Art Tray','Bacon',
         'Baguette','Bakewell','Bare Popcorn','Basket','Bowl Nic Pitt','Bread',
         'Bread Pudding','Brioche and salami','Brownie','Cake','Caramel bites',
         'Cherry me Dried fruit','Chicken sand','Chicken Stew','Chimichurri Oil',
         'Chocolates','Christmas common','Coffee','Coffee granules','Coke','Cookies',
         'Crepes','Crisps','Drinking chocolate spoons','Duck egg','Dulce de Leche',
         'Eggs','Ella Kitchen Pouches','Empanadas','Extra Salami or Feta','Fairy Doors',
         'Farm House','Focaccia','Frittata','Fudge','Gift voucher','Gingerbread syrup',
         'Granola','Hack the stack','Half slice Monster','Hearty & Seasonal','Honey',
         'Hot chocolate','Jam','Jammie Dodgers','Juice','Keeping It Local',
         'Kids biscuit','Lemon and coconut','Medialuna','Mighty Protein',
         'Mineral water','Mortimer','Muesli','Muffin','My-5 Fruit Shoot','Nomad bag',
         'Olum & polenta','Panatone','Pastry','Pick and Mix Bowls','Pintxos',
         'Polenta','Postcard','Raspberry shortbread sandwich','Raw bars','Salad',
         'Sandwich','Scandinavian','Scone','Siblings','Smoothies','Soup',
         'Spanish Brunch','Spread','Tacos/Fajita','Tartine','Tea','The BART',
         'The Nomad','Tiffin','Toast','Truffles','Tshirt','Valentine card',
         'Vegan Feast','Vegan mincepie', 'Victorian Sponge']
    )

    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekday or Weekend', ['Weekday', 'Weekend'])
    month = st.select_slider(
        'Month',
        ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
         'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    )
    day = st.select_slider(
        'Day',
        ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'],
        value='Senin'
    )
    
    return period_day, weekday_weekend, month, day, item


def encode(x):
    return 1 if x >= 1 else 0


def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ', '.join(x)


def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered = data.loc[data["antecedents"] == item_antecedents]

    if filtered.empty:
        return None

    return list(filtered.iloc[0, :])


# ----------------------------
# MAIN
# ----------------------------

period_day, weekday_weekend, month, day, item = user_input_features()
data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

if type(data) != type("No Result"):
    item_count = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(
        index='Transaction', columns='Item', values='Count', aggfunc='sum'
    ).fillna(0)

    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1.0

    rules = association_rules(
        frequent_items, metric=metric, min_threshold=min_threshold
    )[["antecedents", "consequents", "support", "confidence", "lift"]]

    rules.sort_values("confidence", ascending=False, inplace=True)

    result = return_item_df(item)

    if result is None:
        st.warning(f"Tidak ditemukan kombinasi item untuk **{item}**.")
    else:
        st.success(
            f"Jika konsumen membeli **{item}**, maka mereka biasanya membeli "
            f"**{result[1]}** secara bersamaan."
        )

else:
    st.warning("Tidak ada data sesuai filter yang dipilih.")
