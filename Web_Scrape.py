# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:58:59 2019

@author: David Delgado
"""

# Import libraries
import pandas as pd
import requests
import sys

# Import plotting library to test
import matplotlib.pyplot as plt


def scrape_data(card_name:str, card_set:str, save_data:bool = True):
    """
        Takes in the card name as well as the card set and then returns a
        dataframe that has the date as the index and the price of the card
        on that date

        Inputs:
            card_name - Name of the card
            card_set - Name of the set the card is from
            save_data - If we want to save the data to the data folder
                (default=False)
        Returns:
            price_data - Dataframe of price data

    """

    input_card_name = card_name

    # Get the strings ready for the URL
    card_name = card_name.replace(" ", "+")
    card_set = card_set.replace(" ", "+")

    r = requests.get('https://www.mtggoldfish.com/price/'+ card_set + '/' +card_name + '#paper')

    html = r.text

    prices_start = html.find('var d = "Date,' + input_card_name +'";')
    prices_end = html.find('g = new Dygraph')

    # add the -1 so that we dont get a random None at the end
    prices = html[prices_start:prices_end-1]

    # Check if the dataframe is empty which means something went
    # wrong in the reading in of the card name or set
    if len(prices) == 0:
        sys.exit('Error reading in card, check spelling')
    # Convert to dataframe
    price_data = pd.DataFrame(prices.split('\n'), columns=['Data'])
    price_data = price_data["Data"].str.split(",", n=1, expand=True)

    # Rename the columns
    price_data.rename(columns={0:'Date', 1:'Price',}, inplace = True)

    # Get ride of the first line
    price_data = price_data.iloc[1:,:]

    # Regex the datab
    price_data.Date = price_data.Date.str.extract('(\d*\-\d+\-\d+)', expand=False)
    price_data.Price = price_data.Price.str.extract('(\d*\.\d+|\d+)', expand=False).astype(float)

    # Convert to datetime and make it the index
    price_data.index = pd.to_datetime(price_data.Date, format='%Y-%m-%d')
    price_data = price_data.drop('Date', axis = 1)

    # If we want to save the data to the data folder
    if save_data:
        CSV_name = input_card_name.replace(" ", "_")
        price_data.to_csv("Data/" + str(CSV_name)+ ".csv")

    return price_data

